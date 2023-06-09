import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, get_gpu_memory_map
from eval import evaluate, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args
import time

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
if args.method[:3] == 'dag' or args.method == 'transformer':
    print(args.method[:3])
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset, dag=1)
else:
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset, dag=0)
    
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, dataset, n, c, d, device)

### Loss function (Single-class, Multi-class) ###

criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
# print('MODEL:', model)

### Adj storage for relational bias ###
adjs = []
adj, _ = remove_self_loops(dataset.graph['edge_index'].to(device))
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
print(adj.shape)
from torch_geometric.utils import to_undirected
for i in range(args.rb_order - 1): # edge_index of high order adjacency
    if args.method[:3] == 'dag':
        adj = to_undirected(dataset.graph['edge_index_dag']).to(device)
    else:
        adj = adj_mul(adj, adj, n)
    print(adj.shape)
    adjs.append(adj)
dataset.graph['adjs'] = adjs
def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        if args.method == 'nodeformer':
            out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
        elif args.method == 'dag_nodeformer' or 'transformer' or 'sat':
            out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau, sat=args.sat)
            # print(model)
            # print("Total number of parameters: {}".format(count_parameters(model)))
        else:
            out = model(dataset)
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        if args.method == 'nodeformer' or args.method == 'dag_nodeformer':
            loss -= args.lamda * sum(link_loss_) / len(link_loss_)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

    logger.print_statistics(run)
    

results = logger.print_statistics()
