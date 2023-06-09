
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import torch_geometric.utils as utils
import os
import re
import pandas as pd
import scipy as sc
import numpy
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils.convert import to_networkx, from_networkx
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
def node_iter(G):
   return G.nodes

def node_dict(G):
    node_dict = G.nodes
    return node_dict

def read_graphfile():
    graph_num=1000
    adj_list={i:[] for i in range(1,graph_num+1)}    
    index_graph={i:[] for i in range(1,graph_num+1)}
    adj_list_dag={i:[] for i in range(1,graph_num+1)}  
    node_attrs={i:[] for i in range(1,graph_num+1)} 
    Y_valid={i:[] for i in range(1,graph_num+1)} 
    graph_hindex=[]
    edge_weight=[]
    edge_proba=[]
    Name=[]
    authors_attributes=[]
    files_list=[]
    num_edges = 0
    index_i = 1
    for_i = -1
    for root, dirs, files in os.walk("./data/data_origin", topdown=False):
        for name in files:
            if(name[0]=='p'):
                files_list.append((name,int(re.findall(r"\d+\d*", name)[0])))
    # print(files_list)
    files_list = sorted(files_list, key=lambda x: x[1])
    # print(files_list)
    for name_pair in tqdm(files_list):
        name=name_pair[0]
        if(name[0]=='p'):
            for_i += 1
            path="./data/data_origin"+'/'+name
            idx_features_labels = np.genfromtxt("{}".format(path),
                                dtype=np.dtype(str))
            # build graph
            if(len(idx_features_labels.shape)==1):
                idx_features_labels=idx_features_labels[np.newaxis,:]
            idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt(path.replace("papers", "influence"),
                                            dtype=np.dtype(str))
            fea_labels=pd.DataFrame(idx_features_labels)
            y_valid = fea_labels[3].values.astype(int)
            fea_labels[1]=pd.to_numeric(fea_labels[1])-1965
            fea_labels=fea_labels[[0,1,2]]
            idx_features_labels=fea_labels.values

            if(edges_unordered.shape[0]==0):
                adj_list[index_i]=[]
                adj_list_dag[index_i]=[]
                edge_weight.append([])
                print("empty edge")
            else:
                if(len(edges_unordered.shape)==1):
                    edges_unordered=edges_unordered[np.newaxis,:]
                # print(edges_unordered)
                # print(edges_unordered.shape)
                edgeset=set()
                edges_=[]
                # print(edges_unordered)
                for j in range(edges_unordered.shape[0]):
                    # print(idx_features_labels[idx_map[edges_unordered[j][1]]])
                    if (idx_features_labels[idx_map[edges_unordered[j][1]]][1]>=idx_features_labels[idx_map[edges_unordered[j][0]]][1]):
                        continue
                    if((edges_unordered[j][0],edges_unordered[j][1]) in edgeset or (edges_unordered[j][1],edges_unordered[j][0]) in edgeset):
                        continue
                    else:
                        if(edges_unordered[j][0]==edges_unordered[j][1]):
                            continue
                        edgeset.add((edges_unordered[j][0],edges_unordered[j][1]))
                        edges_.append(edges_unordered[j])
                edges_unordered=np.array(edges_)
                # print(edges_unordered)
                if(edges_unordered.shape[0]==0):
                    adj_list[index_i]=[]
                    edge_weight.append([])
                else:
                    edge_w_temp = edges_unordered[:,2:].astype(np.float32)
                    edge_w=[]
                    for j in range(edge_w_temp.shape[0]):
                        edge_w.append(edge_w_temp[j,:])
                        edge_w.append(edge_w_temp[j,:])
                    edge_w=np.array(edge_w).astype(np.float32)
                    # print(edge_w,edge_w_temp)
                    edges_unordered = edges_unordered[:,:2]
                    # print(edges_unordered,idx_map)
                    # print(edges_unordered)
                    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                                    dtype=np.int32).reshape(edges_unordered.shape)
                    edge_weight.append(edge_w)
                    for line in edges:
                        e0,e1=(int(line[1]),int(line[0]))
                        if (idx_features_labels[e0][1]<idx_features_labels[e1][1]):
                            # print(index_i)
                            adj_list_dag[index_i].append([e0,e1])
                            adj_list[index_i].append([e0,e1])
                            adj_list[index_i].append([e1,e0])
                            num_edges += 1

            for line in idx_features_labels[:, 1:]:
                attrs = [float(attr) for attr in line]
                node_attrs[index_i].append(np.array(attrs))
            Name.append(name)
            Y_valid[index_i]=y_valid.tolist()
            index_i+=1

    return edge_weight,Y_valid,adj_list,node_attrs,Name,adj_list_dag

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs

def get_rw_landing_probs(edge_index, ksteps=range(1,21), edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing

def top_sort(edge_index, graph_size):

    node_ids = numpy.arange(graph_size, dtype=int)

    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        if(unevaluated_mask.shape==()):
            unevaluated_mask=np.array([unevaluated_mask])
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()


def add_order_info_01(graph):

    l0 = top_sort(graph.edge_index_dag, graph.num_nodes)
    ei2 = torch.LongTensor([list(graph.edge_index_dag[1]), list(graph.edge_index_dag[0])])
    l1 = top_sort(ei2, graph.num_nodes)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])

    graph.__setattr__("bi_layer_idx0", l0)
    graph.__setattr__("bi_layer_index0", ns)
    graph.__setattr__("bi_layer_idx1", l1)
    graph.__setattr__("bi_layer_index1", ns)
    assert_order(graph.edge_index_dag, l0, ns)
    assert_order(ei2, l1, ns)


def assert_order(edge_index, o, ns):
    # already processed
    proc = []
    for i in range(max(o)+1):
        # nodes in position i in order
        l = o == i
        l = ns[l].tolist()
        for n in l:
            # predecessors
            ps = edge_index[0][edge_index[1] == n].tolist()
            for p in ps:
                assert p in proc
        proc += l


# T.ToDense(300)
import re
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, DAG=True, transform=None, pre_transform=None):
        transform=T.Compose([
            # T.NormalizeFeatures(),
            # T.ToSparseTensor(),
            # T.ToSparseTensor(attr='edge_attr'),
        ])
        self.DAG = DAG
        super(MultiSessionsGraph, self).__init__(root, transform=transform)
    
        # super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return ['data.txt']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def get_idx_split(self, split_type = None):
        with open("test.csv", 'r') as f:
            test = [int(line.rstrip('\n')) for line in f]
        with open("valid.csv", 'r') as f:
            valid = [int(line.rstrip('\n')) for line in f]
        with open("train.csv", 'r') as f:
            train = [int(line.rstrip('\n')) for line in f]
        
        return {'train': torch.tensor(train, dtype = torch.long), 'valid': torch.tensor(valid, dtype = torch.long), 'test': torch.tensor(test, dtype = torch.long)}
    

    def process(self):
        
        data_list = []
        edge_weight,Y_valid,adj_list,node_attrs,Name,adj_list_dag=read_graphfile()
        mean_neighborhood = 0
        mean_nodes = 0
        max_depth = 0
        min_depth = 10000
        mean_depth = 0
        max_degree = 0
        min_degree = 10000
        mean_degree = 0
        real_graph_number = 0
        for i in range(len(Name)):
            number = re.findall(r"\d+\d*",Name[i])
            number=int(number[0])
            dag = nx.DiGraph(adj_list_dag[i+1])
            dag.add_nodes_from(range(len(node_attrs[i+1])))
            # print(nx.is_directed_acyclic_graph(dag))
            if nx.is_directed_acyclic_graph(dag)==False:
                print("not dag")
                raise ValueError("not dag")
            pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t(),name=torch.tensor(number,dtype=torch.int))
            pyg_graph.y_valid=torch.tensor(Y_valid[i+1],dtype=torch.int)
            edge_index_dag=torch.tensor(adj_list_dag[i+1],dtype=torch.long).t()
            
            rw_landing = get_rw_landing_probs(
                                            edge_index=edge_index_dag,
                                            num_nodes=pyg_graph.num_nodes)
            pyg_graph.RWPE = rw_landing
            undir_edge_index = to_undirected(edge_index_dag)
            L = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None,
                            num_nodes=pyg_graph.num_nodes)
            )
            evals, evects = np.linalg.eigh(L.toarray())
            _, pyg_graph.Eigvecs = get_lap_decomp_stats(
                evals=evals, evects=evects,
                max_freqs=8,
                eigvec_norm='L2')

            # calculate node depth
            graph_size=pyg_graph.num_nodes
            edge_index=edge_index_dag
            node_ids = numpy.arange(graph_size, dtype=int)
            node_order = numpy.zeros(graph_size, dtype=int)
            unevaluated_nodes = numpy.ones(graph_size, dtype=bool)
            # print(unevaluated_nodes,edge_index)
            # print(edge_index)
            if(edge_index.shape[0]==0):
                pe = torch.from_numpy(np.array(range(graph_size))).long()
                pyg_graph.abs_pe = pe
            else:
                real_graph_number+=1
                parent_nodes = edge_index[0]
                child_nodes = edge_index[1]

                n = 0
                while unevaluated_nodes.any():
                    # Find which parent nodes have not been evaluated
                    unevaluated_mask = unevaluated_nodes[parent_nodes]
                    if(unevaluated_mask.shape==()):
                        unevaluated_mask=np.array([unevaluated_mask])
                    # Find the child nodes of unevaluated parents
                    unready_children = child_nodes[unevaluated_mask]

                    # Mark nodes that have not yet been evaluated
                    # and which are not in the list of children with unevaluated parent nodes
                    nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

                    node_order[nodes_to_evaluate] = n
                    unevaluated_nodes[nodes_to_evaluate] = False

                    n += 1
                
                pe = torch.from_numpy(node_order).long()
                pyg_graph.abs_pe = pe

            pyg_graph.edge_index_dag = edge_index_dag
            data_new = Data(x=pyg_graph.x, edge_index=edge_index_dag)
            DG = to_networkx(data_new)

            # Statistics
            depth_ = np.max(node_order)
            maxdegree_ = np.max([i[1] for i in DG.degree()])
            mindegree_ = np.min([i[1] for i in DG.degree()])
            if(depth_>max_depth):
                max_depth=depth_
            if(depth_<min_depth):
                min_depth=depth_
            mean_depth += depth_
            max_degree = max(max_degree,maxdegree_)
            min_degree = min(min_degree,mindegree_)
            mean_degree += np.mean([i[1] for i in DG.degree()])
            
            # Compute DAG transitive closures
            TC = nx.transitive_closure_dag(DG)

            # TC_copy = TC.copy()
            # for edge in TC_copy.edges():
            #     if(nx.shortest_path_length(DG,source=edge[0],target=edge[1])>1000):
            #         TC.remove_edge(edge[0], edge[1])
                    
            # add k-hop-neighborhood
            # for node_idx in range(pyg_graph.num_nodes):
            #     sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
            #         node_idx, 
            #         3, 
            #         pyg_graph.edge_index,
            #         relabel_nodes=True, 
            #         num_nodes=pyg_graph.num_nodes
            #         )
            #     for node in sub_nodes:
            #         TC.add_edge(node_idx, node.item())
            
            data_new = from_networkx(TC)
            edge_index_dag = data_new.edge_index
            if(self.DAG):
                # DAG receptive fields
                pyg_graph.dag_rr_edge_index = to_undirected(edge_index_dag)
                mean_neighborhood+=pyg_graph.dag_rr_edge_index.shape[1]/pyg_graph.num_nodes
                mean_nodes+=pyg_graph.num_nodes
            else:
                # complete edges from all nodes
                n = pyg_graph.num_nodes
                s = torch.arange(n)
                pyg_graph.dag_rr_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
            
            # prepare for DAGNN 
            add_order_info_01(pyg_graph)

            print(pyg_graph)
            data_list.append(pyg_graph)
        print(mean_neighborhood/real_graph_number,mean_nodes/real_graph_number,real_graph_number)
        print(max_depth,min_depth,mean_depth,max_degree,min_degree,mean_degree)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])