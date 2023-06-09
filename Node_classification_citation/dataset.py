from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, class_rand_splits
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import degree
import os


import networkx as nx
import scipy.sparse as sp

from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx, from_networkx


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname='', dag=0):
    if dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname, dag=dag)
    else:
        raise ValueError('Invalid dataname')
    return dataset




def index_to_mask(index, size):
    mask = torch.zeros(size, dtype = torch.bool, device = index.device)
    mask[index] = 1

    return mask

def load_planetoid_dataset(data_dir, name, dag):
    if dag==0:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                                name=name, transform=transform)
        # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
        data = torch_dataset[0]

        node_feat = data.x
        label = data.y
        num_nodes = data.num_nodes

        dataset = NCDataset(name)

        dataset.train_idx = torch.where(data.train_mask)[0]
        dataset.valid_idx = torch.where(data.val_mask)[0]
        dataset.test_idx = torch.where(data.test_mask)[0]

        dataset.graph = {
                        'edge_index': data.edge_index,
                        'node_feat': node_feat,
                        'edge_feat': None,
                        'num_nodes': num_nodes}
        dataset.label = label
    else:
        if(name=='cora'):
            data_dir = os.path.expanduser("./cora")
            edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
            edgelist["label"] = "cites"
            Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label", source='source', target='target', create_using=nx.DiGraph)
            nx.set_node_attributes(Gnx, "paper", "label")
            feature_names = ["w_{}".format(ii) for ii in range(1433)]
            column_names =  feature_names + ["subject"]
            node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
            node_data['subject'] =  node_data['subject'].replace({'Case_Based': 0,'Genetic_Algorithms':1,'Neural_Networks':2,'Probabilistic_Methods':3,'Reinforcement_Learning':4,'Rule_Learning':5,'Theory':6})
            new_nodes = {old_node: new_node for new_node, old_node in enumerate(node_data.index)}
            new_G = nx.DiGraph()
            for i in range(Gnx.number_of_nodes()):
                new_G.add_node(i)

            for old_edge in Gnx.edges:
                new_start = new_nodes[old_edge[0]]
                new_end = new_nodes[old_edge[1]]
                new_G.add_edge(new_start, new_end)
            print(new_G)
            new_G_undirected = new_G.to_undirected()
            edge_index_undirected = from_networkx(new_G_undirected).edge_index
            cycles = nx.simple_cycles(new_G)
            # print(len(list(cycles)))
            count=0 
            while True:
                try:
                    count+=1
                    cycle = nx.find_cycle(new_G, orientation='original')
                    # print(*cycle[0][:2])
                    new_G.remove_edge(*cycle[0][:2])
                except nx.exception.NetworkXNoCycle:
                    break
            # print(node_data.iloc[14])
            tx = torch.from_numpy(node_data.values[:,:1433])
            y = torch.from_numpy(node_data.values[:,-1])
            TC = nx.transitive_closure_dag(new_G)
            print(TC)
            edge_index = from_networkx(TC).edge_index
            print(edge_index)
            data = Data(x=tx, edge_index=edge_index, y=y)
            data.edge_index_undirected = edge_index_undirected
            print(data)
            indices = []
            num_classes = 7
            for i in range(num_classes):
                index = torch.nonzero(data.y == i).view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

            train_index = torch.cat([i[:20] for i in indices], dim=0)

            rest_index = torch.cat([i[20:] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]

            data.train_mask = index_to_mask(train_index, size=data.num_nodes)
            data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
            data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
            for index in range(data.x.shape[0]):
                value = data.x[index,:]
                value = value - value.min()
                value = value / value.max()
                data.x[index,:] = value
            edge_index = data.edge_index_undirected
            node_feat = data.x.float()
            label = data.y
            num_nodes = data.num_nodes

            dataset = NCDataset(name)

            dataset.train_idx = torch.where(data.train_mask)[0]
            dataset.valid_idx = torch.where(data.val_mask)[0]
            dataset.test_idx = torch.where(data.test_mask)[0]

            dataset.graph = {'edge_index_dag': data.edge_index,
                            'edge_index': data.edge_index_undirected,
                            'node_feat': node_feat,
                            'edge_feat': None,
                            'num_nodes': num_nodes}
            dataset.label = label
        elif(name=='citeseer'):
            data_dir = os.path.expanduser("./citeseer")
            edgelist = pd.read_csv(os.path.join(data_dir, "citeseer.cites"), sep='\t', header=None, names=["target", "source"])
            edgelist["label"] = "cites"
            Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label", source='source', target='target', create_using=nx.DiGraph)
            nx.set_node_attributes(Gnx, "paper", "label")
            feature_names = ["w_{}".format(ii) for ii in range(3703)]
            column_names =  feature_names + ["subject"]
            node_data = pd.read_csv(os.path.join(data_dir, "citeseer.content"), sep='\t', header=None, names=column_names)
            node_data['subject'] =  node_data['subject'].replace({'Agents': 0,'IR':1,'DB':2,'ML':3,'HCI':4,'AI':5})
            new_nodes = {str(old_node): new_node for new_node, old_node in enumerate(node_data.index)}
            # print(Gnx.number_of_edges())
            new_G = nx.DiGraph()
            for i in range(Gnx.number_of_nodes()):
                new_G.add_node(i)
            # print(new_nodes)
            for old_edge in Gnx.edges:
                try:
                    new_start = new_nodes[str(old_edge[0])]
                    new_end = new_nodes[str(old_edge[1])]
                    new_G.add_edge(new_start, new_end)
                except:
                    continue
            # print(new_G.number_of_edges())
            cycles = nx.simple_cycles(new_G)
            # print(len(list(cycles)))
            count=0 
            while True:
                try:
                    count+=1
                    cycle = nx.find_cycle(new_G, orientation='original')
                    # print(*cycle[0][:2])
                    new_G.remove_edge(*cycle[0][:2])
                except nx.exception.NetworkXNoCycle:
                    break
            print(new_G.number_of_edges())
            new_G_undirected = new_G.to_undirected()
            edge_index_undirected = from_networkx(new_G_undirected).edge_index
            # print(node_data.iloc[14])
            tx = torch.from_numpy(node_data.values[:,:3703])
            y = torch.from_numpy(node_data.values[:,-1])
            TC = nx.transitive_closure_dag(new_G)
            print(TC)
            edge_index = from_networkx(TC).edge_index
            # print(edge_index)
            data = Data(x=tx, edge_index=edge_index, y=y)
            data.edge_index_undirected = edge_index_undirected
            # print(data)
            indices = []
            num_classes = 6
            for i in range(num_classes):
                index = torch.nonzero(data.y == i).view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

            train_index = torch.cat([i[:20] for i in indices], dim=0)

            rest_index = torch.cat([i[20:] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]

            data.train_mask = index_to_mask(train_index, size=data.num_nodes)
            data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
            data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
            for index in range(data.x.shape[0]):
                value = data.x[index,:]
                value = value - value.min()
                value = value / value.max()
                data.x[index,:] = value
            edge_index = data.edge_index_undirected
            node_feat = data.x.float()
            label = data.y
            num_nodes = data.num_nodes

            dataset = NCDataset(name)

            dataset.train_idx = torch.where(data.train_mask)[0]
            dataset.valid_idx = torch.where(data.val_mask)[0]
            dataset.test_idx = torch.where(data.test_mask)[0]

            dataset.graph = {'edge_index_dag': data.edge_index,
                            'edge_index': data.edge_index_undirected,
                            'node_feat': node_feat,
                            'edge_feat': None,
                            'num_nodes': num_nodes}
            # print(data.edge_index_undirected.shape,data.edge_index.shape)
            dataset.label = label
        elif(name=='pubmed'):
            from stellargraph import datasets
            import pickle
            dataset = datasets.PubMedDiabetes()
            G, _subjects = dataset.load()
            if os.path.isfile('pubmed.pkl'):
                with open('pubmed.pkl', 'rb') as f:
                    new_G = pickle.load(f)
            else:
                dataset = datasets.PubMedDiabetes()
                G, _subjects = dataset.load()
                G = G.to_networkx()
                nx.set_node_attributes(G, dict(enumerate(_subjects)), "y")
                new_G = G.to_directed()
                count = 0
                while True:
                    try:
                        count+=1
                        cycle = nx.find_cycle(new_G, orientation='original')
                        print(*cycle[0][:2])
                        new_G.remove_edge(*cycle[0][:2])
                    except nx.exception.NetworkXNoCycle:
                        break
                print(new_G.number_of_edges())
                save_name = 'pubmed.pkl'
                with open(save_name, 'wb') as f:
                    pickle.dump(new_G, f, pickle.HIGHEST_PROTOCOL)
            new_G_undirected = nx.DiGraph(new_G).to_undirected()
            edge_index_undirected = from_networkx(new_G_undirected).edge_index
            print(edge_index_undirected.shape)
            new_G = nx.DiGraph(new_G)
            TC = new_G.copy()
            def descendants(G, source):
                des = {n for n, d in nx.shortest_path_length(G, source=source).items() if d <= 2}
                return des - {source}
            for v in new_G:
                TC.add_edges_from((v, u) for u in descendants(new_G, v) if u not in TC[v])
            # TC = nx.transitive_closure_dag(nx.DiGraph(new_G))
            print(TC)
            attr_dict = {node: {"x": data.pop("feature")} for node, data in TC.nodes(data=True)}
            nx.set_node_attributes(TC, attr_dict)
            attr_dict = {node: {"y": _subjects.loc[node]} for node, data in TC.nodes(data=True)}
            nx.set_node_attributes(TC, attr_dict)
            print(set([attr for n, attr_dict in TC.nodes(data=True) for attr in attr_dict.keys()]))
            for (e1, e2, feat_dict) in TC.edges(data=True):
                try:
                    feat_dict.pop("label")
                    feat_dict.pop("weight")
                except:
                    continue
            import copy
        
            # TC_copy = copy.deepcopy(TC)
            # for (e1, e2, feat_dict) in TC_copy.edges(data=True):
            #     if (e1 not in _subjects.index.values):
            #         TC.remove_edge(e1, e2)
            #         continue
            #     if (e2 not in _subjects.index.values):
            #         TC.remove_edge(e1, e2)
            #         continue
            data = from_networkx(TC)
            
            indices = []
            num_classes = 3
            for i in range(num_classes):
                index = torch.nonzero(data.y == i).view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

            train_index = torch.cat([i[:20] for i in indices], dim=0)

            rest_index = torch.cat([i[20:] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]

            data.train_mask = index_to_mask(train_index, size=data.num_nodes)
            data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
            data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
            
            for index in range(data.x.shape[0]):
                value = data.x[index,:]
                value = value - value.min()
                value = value / value.max()
                data.x[index,:] = value
            
            node_feat = data.x.float()
            label = data.y
            num_nodes = data.num_nodes

            dataset = NCDataset(name)

            dataset.train_idx = torch.where(data.train_mask)[0]
            dataset.valid_idx = torch.where(data.val_mask)[0]
            dataset.test_idx = torch.where(data.test_mask)[0]

            dataset.graph = {'edge_index_dag': data.edge_index,
                            'edge_index': edge_index_undirected,
                            'node_feat': node_feat,
                            'edge_feat': None,
                            'num_nodes': num_nodes}
            # print(data.edge_index_undirected.shape,data.edge_index.shape)
            dataset.label = label 

    return dataset


