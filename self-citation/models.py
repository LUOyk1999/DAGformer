import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GlobalAttention, GraphMultisetTransformer, Set2Set, GlobalAttention
from torch_geometric.nn import ResGatedGraphConv,ChebConv,SAGEConv,GCNConv,GATConv,TransformerConv,AGNNConv,EdgePooling,GraphConv,GCN2Conv,TopKPooling,SAGPooling
from torch_geometric.nn import GINConv,GATv2Conv,ASAPooling,LEConv,MFConv,SGConv,ARMAConv,TAGConv,PNAConv
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, ReLU, Sequential
from torch.nn import BatchNorm1d as BatchNorm
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch_geometric.transforms as T
from gtrick.pyg import VirtualNode

import matplotlib.pyplot as plt


class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(in_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=None))

        return out

    def message(self, x_j, edge_attr):
        if(edge_attr==None):
            return F.relu(x_j)
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

def get_conv(conv_name):
    if conv_name == 'GCNConv':
        return GCNConv
    elif conv_name == 'SAGEConv':
        return SAGEConv
    elif conv_name == 'GATConv':
        return GATConv
    elif conv_name == 'ARMAConv':
        return ARMAConv
    elif conv_name == 'GINConv':
        return GINConv



class Model(torch.nn.Module):
    def __init__(self, args, deg):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.conv_name = args.conv_name
        self.VN = args.VN
        Conv = get_conv(self.conv_name)

        self.convs = torch.nn.ModuleList()
        self.vns = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        if self.conv_name=='PNAConv':
            aggregators = ['mean']
            scalers = ['identity']
            self.convs.append(PNAConv(self.num_features, self.nhid,
                                aggregators=aggregators, scalers=scalers,
                                deg=deg, towers=2))
        else:
            self.convs.append(Conv(self.num_features, self.nhid))
        self.batch_norms.append(torch.nn.BatchNorm1d(self.nhid))
        if(self.VN==1):
            self.vns.append(VirtualNode(self.num_features, self.nhid, dropout=self.dropout_ratio))
        for layer in range(self.num_layers - 1):
            if self.conv_name=='PNAConv':
                self.convs.append(PNAConv(self.nhid, self.nhid,
                                    aggregators=aggregators, scalers=scalers,
                                    deg=deg, towers=2))
            else:
                self.convs.append(Conv(self.nhid, self.nhid))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.nhid))
            if(self.VN==1):
                self.vns.append(VirtualNode(self.nhid, self.nhid, dropout=self.dropout_ratio))

        self.lin = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        # if(self.conv_name=='GATConv'):
        #     edge_index = data.dag_rr_edge_index
        h_list = [x]
        for layer in range(self.num_layers):
            h = h_list[layer]
            if(self.VN==1):
                h, vx = self.vns[layer].update_node_emb(h_list[layer], edge_index, batch)
            h = self.convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout_ratio, training = self.training)
            if(self.VN==1):
                vx = self.vns[layer].update_vn_emb(h, batch, vx)

            h_list.append(h)

        node_representation = 0
        for layer in range(self.num_layers):
            node_representation += F.relu(h_list[layer + 1])

        x = node_representation

        x = self.lin(x)

        return x
