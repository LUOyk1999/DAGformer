import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from utils import pad_batch, unpad_batch
BIG_CONSTANT = 1e8

def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):
    del is_query
    if projection_matrix is None:
        return F.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / torch.sqrt(
            torch.tensor(projection_matrix.shape[0], torch.float32)
        )
        data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
        return F.relu(data_dash) + numerical_stabilizer

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash

def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)

def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)

def kernelized_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None, tau=0.25, return_weight=True):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    z_num = numerator(query_prime, key_prime, value)
    z_den = denominator(query_prime, key_prime)

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den # [B, N, H, D]

    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output

def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None,
                                K=10, tau=0.25, return_weight=True):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    gumbels = (
        -torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).to(query.device) / tau # [N, B, H, K]
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]
    z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K]

    z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]

    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output

def add_conv_relational_bias(x, edge_index, b, trans='sigmoid'):
    '''
    compute updated result by the relational bias of input adjacency
    the implementation is similar to the Graph Convolution Network with a (shared) scalar weight for each edge
    '''
    row, col = edge_index
    d_in = degree(col, x.shape[1]).float()
    d_norm_in = (1. / d_in[col]).sqrt()
    d_out = degree(row, x.shape[1]).float()
    d_norm_out = (1. / d_out[row]).sqrt()
    conv_output = []
    for i in range(x.shape[2]):
        if trans == 'sigmoid':
            b_i = F.sigmoid(b[i])
        elif trans == 'identity':
            b_i = b[i]
        else:
            raise NotImplementedError
        value = torch.ones_like(row) * b_i * d_norm_in * d_norm_out
        adj_i = SparseTensor(row=col, col=row, value=value, sparse_sizes=(x.shape[1], x.shape[1]))
        conv_output.append( matmul(adj_i, x[:, :, i]) )  # [B, N, D]
    conv_output = torch.stack(conv_output, dim=2) # [B, N, H, D]
    return conv_output

import torch_geometric.nn as gnn
from einops import rearrange
import torch_geometric.utils as utils
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES

class StructureExtractor(nn.Module):

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=1,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.GELU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x = self.relu(gcn_layer(x, edge_index))
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x
        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x
    
class NodeFormerConv(gnn.MessagePassing):
    '''
    one layer of NodeFormer that attentive aggregates all nodes over a latent graph
    return: node embeddings for next layer, edge loss at this layer
    '''
    def __init__(self, in_channels, out_channels, num_heads, kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',
                 nb_random_features=10, use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
        super(NodeFormerConv, self).__init__(node_dim=0, aggr='add')
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        self.to_qk = nn.Linear(in_channels, out_channels * num_heads * 2)
        self.to_v = nn.Linear(in_channels, out_channels * num_heads)
        if rb_order >= 1:
            self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)
        self.scale = out_channels ** -0.5
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss
        self.structure_extractor = StructureExtractor(in_channels)
        self.attend = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(0.0)
    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention operation compute the dot-product attention """
        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        return v_j * attn.unsqueeze(-1)
    
    def self_attn(self, qk, v, ptr, return_attn=False):
        """ Self attention based on mask matrix""" 
       
        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None
    
    def reset_parameters(self):
        for conv in self.structure_extractor.gcn:
            conv.reset_parameters()
        self.structure_extractor.out_proj.reset_parameters()
        self.structure_extractor.bn.reset_parameters()
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        self.to_qk.reset_parameters()
        self.to_v.reset_parameters()
        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                torch.nn.init.constant_(self.b, 0.1)
            elif self.rb_trans == 'identity':
                torch.nn.init.constant_(self.b, 1.0)

    def forward(self, z, adjs, tau, sat=0):
        B, N = z.size(0), z.size(1)
        if (sat==1):
            z = self.structure_extractor(z.squeeze(), adjs[0]).unsqueeze(0)
        v = self.to_v(z).squeeze()
        qk = self.to_qk(z).squeeze().chunk(2, dim=-1)
        query = qk[0].reshape(-1, N, self.num_heads, self.out_channels)
        key = qk[1].reshape(-1, N, self.num_heads, self.out_channels)
        value = v.reshape(-1, N, self.num_heads, self.out_channels)
        
        ptr = torch.tensor([0,z.shape[1]], device=z.device)
        z_next, attn = self.self_attn(qk, v, ptr)
        
        z_next = z_next.reshape(-1, N, self.num_heads, self.out_channels)
        
        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
            projection_matrix = create_projection_matrix(
                self.nb_random_features, dim, seed=seed).to(query.device)
            
        start, end = adjs[0]
        query = query / math.sqrt(tau)
        key = key / math.sqrt(tau)
        query_prime = self.kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
        key_prime = self.kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
        query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
        key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
        
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]
        
        z_next = self.Wo(z_next.flatten(-2, -1))

        if self.use_edge_loss: # compute edge regularization loss on input adjacency
            row, col = adjs[0]
            d_in = degree(col, query.shape[1]).float()
            d_norm = 1. / d_in[col]
            d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, A_weight.shape[-1])
            link_loss = torch.mean(A_weight.log() * d_norm_)

            return z_next, link_loss

        else:
            return z_next

class SAT(nn.Module):
    '''
    NodeFormer model implementation
    return: predicted node labels, a list of edge losses at every layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=4, dropout=0.0,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True, use_gumbel=True,
                 use_residual=True, use_act=False, use_jk=False, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
        super(SAT, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                NodeFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel_transformation=kernel_transformation,
                              nb_random_features=nb_random_features, use_gumbel=use_gumbel, nb_gumbel_sample=nb_gumbel_sample,
                               rb_order=rb_order, rb_trans=rb_trans, use_edge_loss=use_edge_loss))
            self.bns.append(nn.LayerNorm(hidden_channels))

        if use_jk:
            self.fcs.append(nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels))
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk
        self.use_edge_loss = use_edge_loss

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, adjs, tau=1.0, sat=0):
        x = x.unsqueeze(0) # [B, N, H, D], B=1 denotes number of graph
        layer_ = []
        link_loss_ = []
        z = self.fcs[0](x)
        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)

        for i, conv in enumerate(self.convs):
            if self.use_edge_loss:
                z, link_loss = conv(z, adjs, tau, sat)
                link_loss_.append(link_loss)
            else:
                z = conv(z, adjs, tau)
            if self.use_residual:
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)

        if self.use_jk: # use jk connection for each layer
            z = torch.cat(layer_, dim=-1)

        x_out = self.fcs[-1](z).squeeze(0)

        if self.use_edge_loss:
            return x_out, link_loss_
        else:
            return x_out