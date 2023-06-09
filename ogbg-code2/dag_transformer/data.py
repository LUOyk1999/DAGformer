# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch_cluster import neighbor_sampler
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_undirected
import networkx as nx
import os
import torch_geometric
import scipy.sparse as sp


class GraphDataset(object):
    def __init__(self, dataset, cache_path=None, use_mpnn=True, k=1000):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.cache_path = cache_path
        self.use_mpnn = use_mpnn
        self.k = k
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):


        data = self.dataset[index]
        data_new = Data(x=data.x, edge_index=data.edge_index_origin)

        DG = to_networkx(data_new)
        
        # Statistics
        # depth = nx.shortest_path_length(DG,0)
        # data.depth = np.max([depth[i] for i in depth])
        # data.maxdegree = np.max([i[1] for i in DG.out_degree()])

        # Compute DAG transitive closures
        TC = nx.transitive_closure_dag(DG)
        TC_copy = TC.copy()

        # Statistics
        # k_list = []
        # k_list.append(TC.number_of_edges()*2)
        # for i in range(8):
        #     for edge in TC_copy.edges():
        #         if(nx.shortest_path_length(DG,source=edge[0],target=edge[1])>(8-i)):
        #             TC.remove_edge(edge[0], edge[1])
        #     k_list.append(TC.number_of_edges()*2)
        #     TC_copy = TC.copy()
        # data.k_list = torch.tensor(k_list)

        # receptive fields k
        if(self.k<1000):
            for edge in TC_copy.edges():
                if(nx.shortest_path_length(DG,source=edge[0],target=edge[1])>self.k):
                    TC.remove_edge(edge[0], edge[1])

        # add k-hop neighborhood
        # for node_idx in range(data.num_nodes):
        #     sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
        #         node_idx, 
        #         2, 
        #         data.edge_index,
        #         relabel_nodes=True, 
        #         num_nodes=data.num_nodes
        #         )
        #     for node in sub_nodes:
        #         TC.add_edge(node_idx, node.item())
        
        data_new = from_networkx(TC)
        edge_index_dag = data_new.edge_index
        if self.use_mpnn:
            # using mpnn to implement DAG attention
            data.dag_rr_edge_index = to_undirected(edge_index_dag)
        else:
            # using mask to implement DAG attention
            num_nodes = data.num_nodes
            if(num_nodes<=1000):
                max_num_nodes = 1000
                mask_rc = torch.tensor([]).new_zeros(max_num_nodes, max_num_nodes).bool()
                for index1 in range(num_nodes):
                    ne_idx = edge_index_dag[0] == index1
                    le_idx = ne_idx.nonzero(as_tuple=True)[0]
                    lp_edge_index = edge_index_dag[1, le_idx]
                    ne_idx_inverse = edge_index_dag[1] == index1
                    le_idx_inverse = ne_idx_inverse.nonzero(as_tuple=True)[0]
                    lp_edge_index_inverse = edge_index_dag[0, le_idx_inverse]
                    mask_r=torch.tensor([]).new_zeros(max_num_nodes).bool()
                    mask_r[lp_edge_index] = True
                    mask_r[lp_edge_index_inverse] = True
                    # mask_r[index1] = True
                    mask_rc[index1] = ~ mask_r
            data.mask_rc = mask_rc
        
        if self.n_features == 1:
            data.x = data.x.squeeze(-1)
        if not isinstance(data.y, list):
            data.y = data.y.view(data.y.shape[0], -1)
            
        return data
