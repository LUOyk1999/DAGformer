from __future__ import print_function
import gzip
import pickle
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
import os
import collections
import igraph
import argparse
import sys
# import pygraphviz as pgv  # uncomment this for BO
from PIL import Image
import math
sys.path.append('%s/../scripts/' % os.path.dirname(os.path.realpath(__file__)))
import networkx as nx
from torch_geometric.data import Data
from src.utils_dag import add_order_info

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()


'''load and save objects'''
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def load_module_state(model, state_name):
    pretrained_dict0 = torch.load(state_name)   #, map_location=torch.device('cpu')
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict0.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return


'''Data preprocessing'''
def load_ENAS_graphs(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    max_n = 0  # maximum number of nodes
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            # if i == burn_in + 50: break
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
             # DAGNN
            if fmt == 'pyg':
                g, n = decode_ENAS_to_pygraph(row)
            elif fmt == 'igraph':
                g, n = decode_ENAS_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            g_list.append((g, y))
    graph_args.num_vertex_type = n_types + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args


def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def decode_ENAS_to_tensor(row, n_types):
    n_types += 2  # add start_type 0, end_type 1
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    g = []
    # ignore start vertex
    for i, node in enumerate(row):
        node_type = node[0] + 2  # assign 2, 3, ... to other types
        type_feature = one_hot(node_type, n_types)
        if i == 0:
            edge_feature = torch.zeros(1, n+1)  # a node will have at most n+1 connections
        else:
            edge_feature = torch.cat([torch.FloatTensor(node[1:]).unsqueeze(0), 
                                     torch.zeros(1, n+1-i)], 1)  # pad zeros
        edge_feature[0, i] = 1 # ENAS node always connects from the previous node
        g.append(torch.cat([type_feature, edge_feature], 1))
    # process the output node
    node_type = 1
    type_feature = one_hot(node_type, n_types)
    edge_feature = torch.zeros(1, n+1)
    edge_feature[0, n] = 1  # output node only connects from the final node in ENAS
    g.append(torch.cat([type_feature, edge_feature], 1))
    return torch.cat(g, 0).unsqueeze(0), n+2


def decode_ENAS_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n+2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        g.vs[i+1]['type'] = node[0] + 2  # assign 2, 3, ... to other types
        g.add_edge(i, i+1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i+1)
    g.vs[n+1]['type'] = 1  # output node
    g.add_edge(n, n+1)
    # note that the nodes 0, 1, ... n+1 are in a topological order
    return g, n+2


def flat_ENAS_to_nested(row, n_nodes):
    # transform a flattened ENAS string to a nested list of ints
    if type(row) == str:
        row = [int(x) for x in row.split()]
    cnt = 0
    res = []
    for i in range(1, n_nodes+1):
        res.append(row[cnt:cnt+i])
        cnt += i
        if cnt == len(row):
            break
    return res


def decode_igraph_to_ENAS(g):
    # decode an igraph to a flattend ENAS string
    n = g.vcount()
    res = []
    adjlist = g.get_adjlist(igraph.IN)
    for i in range(1, n-1):
        res.append(int(g.vs[i]['type'])-2)
        row = [0] * (i-1)
        for j in adjlist[i]:
            if j < i-1:
                row[j] = 1
        res += row
    return ' '.join(str(x) for x in res)

'''
# some code to test format transformations
row = '[[4], [0, 1], [3, 1, 0], [3, 0, 1, 1], [1, 1, 1, 1, 1], [2, 1, 1, 0, 1, 1], [5, 1, 1, 1, 1, 1, 0], [2, 0, 0, 1, 0, 0, 1, 0]]'
row = '[[2], [2, 0], [4, 0, 0], [0, 1, 0, 0], [2, 1, 0, 0, 1], [3, 1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 1, 0], [4, 0, 0, 0, 0, 0, 0, 0], [4, 1, 0, 0, 1, 0, 0, 0, 0], [3, 0, 1, 1, 0, 0, 1, 0, 0, 0], [5, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]'
g, _ = decode_ENAS_to_igraph(row)
string = decode_igraph_to_ENAS(g)
print(row, string)
pdb.set_trace()
pwd = os.getcwd()
os.chdir('software/enas/')
os.system('./scripts/custom_cifar10_macro_final.sh ' + '"' + string + '"')
os.chdir(pwd)
'''


def load_BN_graphs(name, n_types=8, fmt='igraph', rand_seed=0, with_y=True):
    # load raw Bayesian network strings to igraphs or tensors
    g_list = []
    max_n = 0
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            # if i == 50: break
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            # DAGNN
            if fmt == 'pyg':
                g, n = decode_BN_to_pygraph(row)
            elif fmt == 'igraph':
                g, n = decode_BN_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_BN_to_tensor(row, n_types)
            max_n = max(max_n, n)
            assert(max_n == n)  # all BNs should have the same node number
            g_list.append((g, y)) 

    graph_args.num_class = 1  # how many classes of graphs
    graph_args.num_vertex_type = n_types + 2  # how many vertex types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# classes: %d' % graph_args.num_class)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args


def decode_BN_to_tensor(row, n_types):
    n_types += 2  # add start_type 0, end_type 1
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    g = []
    end_vertices = [True] * n
    # ignore start vertex
    for i, node in enumerate(row):
        node_type = node[0] + 2  # convert 0, 1, 2... to 2, 3, 4...
        type_feature = one_hot(node_type, n_types)
        edge_feature = torch.zeros(1, n+1)  # a node will have at most n+1 connections
        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            edge_feature[0, 0] = 1
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    edge_feature[0, j+1] = 1
                    end_vertices[j] = False
        g.append(torch.cat([type_feature, edge_feature], 1))
    # output node
    node_type = 1
    type_feature = one_hot(node_type, n_types)
    edge_feature = torch.zeros(1, n+1)
    for j, flag in enumerate(end_vertices):  # connect all loose-end vertices to the output node
        if flag == True:
            edge_feature[0, j+1] = 1
    g.append(torch.cat([type_feature, edge_feature], 1))
    return torch.cat(g, 0).unsqueeze(0), n+2


def decode_BN_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n+2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        cur_type = node[0] + 2
        g.vs[i+1]['type'] = cur_type
        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            g.add_edge(0, i+1)
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    g.add_edge(j+1, i+1)
    g.vs[n+1]['type'] = 1  # output node
    end_vertices = [v.index for v in g.vs.select(_outdegree_eq=0) if v.index != n+1]
    for j in end_vertices:  # connect all loose-end vertices to the output node
        g.add_edge(j, n+1)
    return g, n+2


# DAGNN
def decode_BN_to_pygraph(row, n_types=8):
    n_types += 2  # add start_type 0, end_type 1

    adj = np.zeros((n_types, n_types))
    x = []
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    end_vertices = [True] * n
    x += [one_hot(0, n_types)]
    # ignore start vertex
    node_types2 = []
    for i, node in enumerate(row):
        node_type = node[0] + 2  # convert 0, 1, 2... to 2, 3, 4...
        node_types2 += [node_type]
        x += [one_hot(node_type, n_types)]
        # edge_feature = torch.zeros(1, n+1)  # a node will have at most n+1 connections
        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            adj[0, i + 1] = 1
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    adj[j + 1, i + 1] = 1
                    end_vertices[j] = False
    # output node
    node_type = 1
    x += [one_hot(node_type, n_types)]
    for j, flag in enumerate(end_vertices):  # connect all loose-end vertices to the output node
        if flag == True:
            adj[j + 1, n + 1] = 1

    nx_graph = nx.DiGraph(adj)
    x = torch.cat(x, dim=0).float()

    # edge_type = []
    # for s, t, d in list(nx_graph.edges(data=True)):
    #     etype = [k for k, v in d.items() if v == 1][0]
    #     edge_type += [EDGE_TYPES[etype]]
    # edge_type = torch.LongTensor(edge_type)

    edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
    # need "index" in name to be recognized during batch collation
    graph = Data(x=x,
                 edge_index=edge_index)  # , edge_type=edge_type, maxy=m, y_index_0=torch.LongTensor(yi0), yi1=torch.LongTensor(yi1), y=torch.LongTensor(y))

    add_order_info(graph)
    # to be able to use igraph methods in DVAE models
    graph.vs = [{'type': 0}] + [{'type': t} for t in node_types2] + [{'type': 1}]

    return graph, n_types


# DAGNN
def decode_ENAS_to_pygraph(row, n_types=6):
    n_types += 2  # add start_type 0, end_type 1

    adj = np.zeros((n_types, n_types))
    x = []
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    x += [one_hot(0, n_types)]
    # ignore start vertex
    node_types2 = []
    for i, node in enumerate(row):
        node_type = node[0] + 2  # convert 0, 1, 2... to 2, 3, 4...
        node_types2 += [node_type]
        x += [one_hot(node_type, n_types)]
        adj[i, i+1] = 1
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                adj[j, i + 1] = 1

    # output node
    node_type = 1
    x += [one_hot(node_type, n_types)]
    adj[n, n + 1] = 1

    nx_graph = nx.DiGraph(adj)
    x = torch.cat(x, dim=0).float()

    edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
    # need "index" in name to be recognized during batch collation
    graph = Data(x=x, edge_index=edge_index)  # , edge_type=edge_type, maxy=m, y_index_0=torch.LongTensor(yi0), yi1=torch.LongTensor(yi1), y=torch.LongTensor(y))

    add_order_info(graph)

    # o = list(range(8))   #list(nx.topological_sort(nx_graph))
    # layers = torch.LongTensor([list(range(8)), o])
    # o.reverse()
    # layers2 = torch.LongTensor([list(range(8)), o])

    # to be able to use igraph methods in DVAE models
    graph.vs = [{'type': 0}] + [{'type': t} for t in node_types2] + [{'type': 1}]

    return graph, n_types


def decode_igraph_to_BN_adj(g):
    # decode an BN igraph to its flattened adjacency matrix string
    types = g.vs['type'][1:-1]
    real_order = np.argsort(types).tolist()
    adj = np.array(g.get_adjacency().data)[1:-1, 1:-1]
    adj = adj[real_order][:, real_order]
    return ' '.join(str(x) for x in adj.reshape(-1))


def adjstr_to_BN(row):
    # input: '0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    # output: [[0], [1, 1], [2, 0, 0], [3, 0, 0, 0], [4, 0, 1, 0, 0], [5, 1, 1, 0, 0, 0], [6, 0, 1, 0, 0, 1, 0], [7, 0, 0, 0, 1, 1, 1, 0]]
    matrix = np.array([int(x) for x in row.split()]).reshape(8, 8)
    res = [[0]]
    for i in range(1, 8):
        cur = [i] + matrix[:i, i].tolist()
        res.append(cur)
    return res


def decode_from_latent_space(
        latent_points, model, decode_attempts=500, n_nodes='variable', return_igraph=False, 
        data_type='ENAS'):
    # decode points from the VAE model's latent space multiple attempts
    # and return the most common decoded graphs
    if n_nodes != 'variable':
        check_n_nodes = True  # check whether the decoded graphs have exactly n nodes
    else:
        check_n_nodes = False
    decoded_arcs = []  # a list of lists of igraphs
    pbar = tqdm(range(decode_attempts))
    for i in pbar:
        current_decoded_arcs = model.decode(latent_points)
        decoded_arcs.append(current_decoded_arcs)
        pbar.set_description("Decoding attempts {}/{}".format(i, decode_attempts))

    # We see which ones are decoded to be valid architectures
    valid_arcs = []  # a list of lists of strings
    if return_igraph:
        str2igraph = {}  # map strings to igraphs
    pbar = tqdm(range(latent_points.shape[0]))
    for i in pbar:
        valid_arcs.append([])
        for j in range(decode_attempts):
            arc = decoded_arcs[j][i]  # arc is an igraph
            if data_type == 'ENAS':
                if is_valid_ENAS(arc, model.START_TYPE, model.END_TYPE):
                    if not check_n_nodes or check_n_nodes and arc.vcount() == n_nodes:
                        cur = decode_igraph_to_ENAS(arc)  # a flat ENAS string
                        if return_igraph:
                            str2igraph[cur] = arc
                        valid_arcs[i].append(cur)
            elif data_type == 'BN':  
                if is_valid_BN(arc, model.START_TYPE, model.END_TYPE, nvt=model.nvt):
                    cur = decode_igraph_to_BN_adj(arc)  # a flat BN adjacency matrix string
                    if return_igraph:
                        str2igraph[cur] = arc
                    valid_arcs[i].append(cur)
        pbar.set_description("Check validity for {}/{}".format(i, latent_points.shape[0]))

    # select the most common decoding as the final architecture
    final_arcs = []  # a list of lists of strings
    pbar = tqdm(range(latent_points.shape[ 0 ]))
    for i in pbar:
        valid_curs = valid_arcs[i]
        aux = collections.Counter(valid_curs)
        if len(aux) > 0:
            arc, num_arc = list(aux.items())[np.argmax(aux.values())]
        else:
            arc = None
            num_arc = 0
        final_arcs.append(arc)
        pbar.set_description("Latent point {}'s most common decoding ratio: {}/{}".format(
                             i, num_arc, len(valid_curs)))

    if return_igraph:
        final_arcs_igraph = [str2igraph[x] if x is not None else None for x in final_arcs]
        return final_arcs_igraph, final_arcs
    return final_arcs


'''Network visualization'''
def plot_DAG(g, res_dir, name, backbone=False, data_type='ENAS', pdf=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name+'.png')
    if pdf:
        file_name = os.path.join(res_dir, name+'.pdf')
    if data_type == 'ENAS':
        draw_network(g, file_name, backbone)
    elif data_type == 'BN':
        draw_BN(g, file_name)
    return file_name


def draw_network(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 0:  
        label = 'input'
        color = 'skyblue'
    elif label == 1:
        label = 'output'
        color = 'pink'
    elif label == 2:
        label = 'conv3'
        color = 'yellow'
    elif label == 3:
        label = 'sep3'
        color = 'orange'
    elif label == 4:
        label = 'conv5'
        color = 'greenyellow'
    elif label == 5:
        label = 'sep5'
        color = 'seagreen3'
    elif label == 6:
        label = 'avg3'
        color = 'azure'
    elif label == 7:
        label = 'max3'
        color = 'beige'
    else:
        label = ''
        color = 'aliceblue'
    #label = f"{label}\n({node_id})"
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)


def draw_BN(g, path):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    label_dict = dict(zip(range(2, 10), 'ASTLBEXD'))
    pos_dict = dict(zip(range(2, 10), ['0, 3!', '2.75, 3!', '0, 2!', '2, 2!', '3.5, 1!', '1.5, 1!', '1.5, 0!', '3.5, 0!']))

    def add_node(graph, node_id, label, shape='circle', style='filled'):
        if label in {0, 1}:
            return
        else:
            label, pos = label_dict[label], pos_dict[label]
        graph.add_node(
                node_id, label=label, color='black', fillcolor='white',
                shape=shape, style=style, pos=pos, fontsize=27,
        )
        return

    if g is None:
        graph.add_node(
                0, label='invalid', color='black', fillcolor='white', 
                shape='box', style='filled', 
        )
        graph.layout(prog='dot')
        graph.draw(path)
        return

    for idx in range(1, g.vcount()-1):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(1, g.vcount()-1):
        for node in g.get_adjlist(igraph.IN)[idx]:
            #if node != g.vcount()-1 and node != 0:  # we don't draw input/output nodes for BN
            node_type = g.vs[node]['type']
            if node_type != 0 and node_type != 1:  # we don't draw input/output nodes for BN
                graph.add_edge(node, idx)

    graph.layout()
    graph.draw(path)
    return path


'''Validity and novelty functions'''
def is_same_DAG(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True


def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    res = g.is_dag()
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        if v.indegree() == 0 and v['type'] != START_TYPE:
            return False
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1


def is_valid_ENAS(g, START_TYPE=0, END_TYPE=1):
    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)
    # in addition, node i must connect to node i+1
    for i in range(g.vcount()-2):
        res = res and g.are_connected(i, i+1)
        if not res:
            return res
    # the output node n must not have edges other than from n-1
    res = res and (g.vs[g.vcount()-1].indegree() == 1)
    return res
    

def is_valid_BN(g, START_TYPE=0, END_TYPE=1, nvt=10):
    # nvt: number of vertex types in this BN
    # first need to be a DAG
    res = g.is_dag()
    # check whether start and end types only appear once
    # BN nodes need not be connected
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
    res = res and n_start == 1 and n_end == 1
    # in addition, every type must appear exactly once
    res = res and (len(set(g.vs['type'])) == nvt) and g.vcount() == nvt
    return res


'''Other util functions'''
def combine_figs_horizontally(names, new_name):
    images = list(map(Image.open, names))
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(new_name)



class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))


