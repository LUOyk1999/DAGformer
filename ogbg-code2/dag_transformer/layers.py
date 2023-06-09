# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from .utils import pad_batch, pad_batch2, unpad_batch
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
import torch.nn.functional as F


class Attention(gnn.MessagePassing):
    """Multi-head DAG attention using PyG interface
    accept Batch data given by PyG
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False, symmetric=False, gnn_type="gcn", **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.gnn_type = gnn_type
        self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x,
            SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=None,
            ptr=None,
            return_attn=False):
        
        # Compute value matrix

        v = self.to_v(x)
        
        if(SAT):
            # using DAG+SAT
            x_struct = self.structure_extractor(x, edge_index, edge_attr)
        else:
            # using DAG+transformer
            x_struct = x
        
        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)
        
        # Compute self-attention
        attn = None
        if dag_rr_edge_index is not None:
            # print(dag_rr_edge_index.shape)
            out = self.propagate(dag_rr_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    dag_rr_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)
            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, mask_dag_, return_attn=return_attn)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention based on MPNN """
        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, mask_dag_, return_attn=False):
        """ Self attention based on mask matrix""" 
       
        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        # DAG mask
        mask_dag_ = mask_dag_.reshape(dots.shape[0],mask_dag_.shape[1],mask_dag_.shape[1])
        mask_dag_ = mask_dag_[:,:dots.shape[2],:dots.shape[3]]
        dots = dots.masked_fill(
            mask_dag_.unsqueeze(1),
            float('-inf'),
        )
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None


class StructureExtractor(nn.Module):

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
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

        self.relu = nn.ReLU()
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



class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                activation="relu", batch_norm=True, pre_norm=False,
                gnn_type="gcn", **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
            bias=False, gnn_type=gnn_type, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
            edge_attr=None, ptr=None,
            return_attn=False,
        ):
        
        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )

        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x
