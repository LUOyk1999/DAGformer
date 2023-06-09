# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
from layers import TransformerEncoderLayer
from einops import repeat
import math
from gps_layer import GPSLayer
from torch_geometric.utils import *

from torch_geometric.utils import softmax


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, SAT, abs_pe_type, abs_pe, edge_index, mask_dag_, dag_rr_edge_index,
            edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = x
        for mod in self.layers:
            output = mod(output, SAT, abs_pe_type, abs_pe, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=edge_attr, degree=degree,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, gps=0, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe='dagpe', dag_attention=1, 
                 use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True,
                 global_pool='mean', SAT=0, **kwargs):
        super().__init__()
        self.SAT = SAT
        self.num_nodes = 8
        self.position = torch.arange(500).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.poe = torch.zeros(500, d_model)
        self.poe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.poe[:, 1::2] = torch.cos(self.position * self.div_term)
        self.dropout = nn.Dropout(0.1)
        self.abs_pe = abs_pe
        self.dag_attention = dag_attention
        self.eigvecs_embeding = nn.Linear(in_features=8,
                                       out_features=d_model,
                                       bias=False)
        self.embedding = nn.Linear(in_features=in_size,
                                    out_features=d_model,
                                    bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        
        self.gps = gps
        self.num_layers = num_layers
        if gps==0:
            encoder_layer = TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
            self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        elif gps==1:
            self.encoder = GPSLayer(dim_h=d_model, num_heads=num_heads, local_gnn_type='CustomGatedGCN', global_model_type='Transformer', dropout=dropout)
        else:
            self.encoder = GPSLayer(dim_h=d_model, num_heads=num_heads, local_gnn_type='CustomGatedGCN', global_model_type='DAG', dropout=dropout)
        
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        
        self.linear = nn.Linear(d_model,in_size)
        self.use_global_pool = use_global_pool

    def forward(self, data, return_attn=False):

        x, edge_index, edge_attr, mask_dag_, batch = data.x, data.edge_index, data.edge_attr, data.mask_rc, data.batch
        # print(data)
        data.ptr = torch.cat([data.batch.new_zeros(1),torch.bincount(data.batch).cumsum(dim=0)],dim=0)
        if(self.dag_attention==0):
            mask_dag_=None
        dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None
        degree = data.degree if hasattr(data, 'degree') else None
        output = self.embedding(x)

        self.poe = self.poe.to(x.device)
        abs_pe = data.abs_pe
        abs_pe = self.poe[:abs_pe.shape[0]][abs_pe]
        abs_pe = abs_pe.to(x.device)
        if self.abs_pe == 'dagpe':
            output = output + abs_pe
        elif self.abs_pe == 'none':
            output = output
        elif self.abs_pe == 'Eigvecs':
            output = output + self.eigvecs_embeding(data.Eigvecs)
        output = self.dropout(output)
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
        else:
            edge_attr = None

        if self.gps==0:
            output = self.encoder(
                output, 
                self.SAT,
                self.abs_pe,
                abs_pe,
                edge_index,
                mask_dag_,
                dag_rr_edge_index,
                edge_attr=edge_attr,
                ptr=data.ptr,
                return_attn=return_attn
            )
        else:
            data.x = output
            for i in range(self.num_layers):
                data = self.encoder(
                data
            )
            output = data.x

        return self.linear(output)
