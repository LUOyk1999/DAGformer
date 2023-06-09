import torch
import numpy
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx
import os
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
# see https://github.com/unbounce/pytorch-tree-lstm/blob/66f29a44e98c7332661b57d22501107bcb193f90/treelstm/util.py#L8
# assume nodes consecutively named starting at 0
#

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

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()


# to be able to use pyg's batch split everything into 1-dim tensors
def add_order_info_01(graph):

    l0 = top_sort(graph.edge_index, graph.num_nodes)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    l1 = top_sort(ei2, graph.num_nodes)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])

    graph.__setattr__("bi_layer_idx0", l0)
    graph.__setattr__("bi_layer_index0", ns)
    graph.__setattr__("bi_layer_idx1", l1)
    graph.__setattr__("bi_layer_index1", ns)
    assert_order(graph.edge_index, l0, ns)
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

neighborhood = 0

def add_order_info(graph):
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    layers = torch.stack([top_sort(graph.edge_index, graph.num_nodes), ns], dim=0)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    layers2 = torch.stack([top_sort(ei2, graph.num_nodes), ns], dim=0)

    graph_size=graph.num_nodes
    edge_index=graph.edge_index
    node_ids = numpy.arange(graph_size, dtype=int)
    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)
    # print(unevaluated_nodes,edge_index[0],edge_index[1])
    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1
    
    pe = torch.from_numpy(node_order).long()
    graph.abs_pe = pe
    data_new = Data(x=graph.x, edge_index=graph.edge_index)
    
    DG = to_networkx(data_new)
    # Compute DAG transitive closures
    TC = nx.transitive_closure_dag(DG)
    data_new = from_networkx(TC)
    edge_index_dag = data_new.edge_index
    num_nodes = graph.num_nodes
    graph.dag_rr_edge_index = to_undirected(edge_index_dag)
    undir_edge_index = to_undirected(graph.edge_index)
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=None,
                    num_nodes=graph.num_nodes)
    )
    evals, evects = np.linalg.eigh(L.toarray())
    _, graph.Eigvecs = get_lap_decomp_stats(
        evals=evals, evects=evects,
        max_freqs=8,
        eigvec_norm='L2')
    
    # Statistics
    # global neighborhood
    # neighborhood+=(edge_index_dag.shape[1]*2)/num_nodes
    # print(neighborhood)

    # DAG mask
    max_num_nodes = 8
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
        mask_rc[index1] = ~ mask_r
    graph.mask_rc = mask_rc

    graph.__setattr__("bi_layer_index", torch.stack([layers, layers2], dim=0))
