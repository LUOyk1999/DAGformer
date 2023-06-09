import argparse
import glob
import os
import time
import math
from dataset import MultiSessionsGraph
import torch
import torch_geometric.utils as utils
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import random
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
from torch.autograd import Variable
from models import Model
from sklearn.model_selection import KFold,StratifiedKFold
from dag_transformer import GraphTransformer
from dagnn import DAGNN
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100, help='batch size') # batch_size should >= 50
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--conv_name', type=str, default='NO', help='conv')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--DAG_attention', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4, help="number of heads")
parser.add_argument('--pe', type=str, default='dagpe')
parser.add_argument('--gps', type=int, default=0)
parser.add_argument('--SAT', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--VN', type=int, default=0)

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

args = parser.parse_args()

if(args.DAG_attention==1):
    dataset = MultiSessionsGraph('./data',args.DAG_attention)
else:
    dataset = MultiSessionsGraph('./data_tf',args.DAG_attention)

args.num_classes = 2

args.num_features = dataset.num_features

print(args)

def train(train_loader,val_loader,test_loader,deg):
    if(args.conv_name=='NO'):
        model = GraphTransformer(in_size=args.num_features,
                                num_class=args.num_classes,
                                d_model=args.nhid,
                                gps=args.gps,
                                abs_pe=args.pe,
                                dim_feedforward=4*args.nhid,
                                dropout=args.dropout_ratio,
                                num_heads=args.num_heads,
                                num_layers=args.num_layers,
                                batch_norm=True,
                                in_embed=False,
                                SAT=args.SAT,
                                deg=deg,
                                edge_embed=False).to(args.device)
    else:
        if(args.conv_name=='DAGNN'):
            model = DAGNN(emb_dim=args.nhid,num_layers=args.num_layers,hidden_dim=args.nhid,num_class=args.num_classes,dropout=args.dropout_ratio).to(args.device)
        else:
            model = Model(args,deg).to(args.device)
    # print(model)
    print("Total number of parameters: {}".format(count_parameters(model)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.27, 0.73]).to(args.device))
    ap=0
    roc=0
    f1=0
    f1_test_final=0
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0

        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        
        for i, data in enumerate(train_loader):
            
            optimizer.zero_grad()

            data = data.to(args.device)
            out = model(data)
            # print(data.x,data.y_valid)
            y = data.y_valid
            y = y.type(torch.LongTensor).to(args.device)
            mask = (y != -2)
            # print(out[mask])
            loss = criterion(out[mask], y[mask])
            # print(out[mask].squeeze(),y[mask])
            # print(loss)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        train_loss = loss_train / len(train_loader)
        
        acc_val, loss_val, preds, labels, y_proba = compute_test(val_loader, model, epoch, criterion)
        
        ap_temp = average_precision_score(
            labels,
            y_proba[:,1]
        )
        roc_temp = roc_auc_score(labels, y_proba[:,1])
        f1_temp = f1_score(labels,preds)

        f1_valid_last = f1_temp
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),'loss_val: {:.6f}'.format(loss_val),'val_ap: {:.6f}'.format(ap_temp),'val_roc: {:.6f}'.format(roc_temp),'val_f1: {:.6f}'.format(f1_temp))
        

        acc_val, loss_val, preds, labels, y_proba = compute_test(test_loader,model,epoch,criterion)
        f1_test = f1_score(labels,preds)
        ap_test = average_precision_score(labels, y_proba[:,1])
        roc_test = roc_auc_score(labels, y_proba[:,1])
        
        if(ap_temp>ap):
            ap = ap_temp
            ap_test_final = ap_test
        if(f1_temp>f1):
            f1 = f1_temp
            f1_test_final = f1_test
        if(roc_temp>roc):
            roc = roc_temp
            roc_test_final = roc_test
        print('Epoch: {:04d}'.format(epoch + 1), 'test_ap: {:.6f}'.format(ap_test_final), 'test_roc: {:.6f}'.format(roc_test_final), 'test_f1: {:.6f}'.format(f1_test_final))
        
        lr_scheduler.step()

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return ap_test_final, roc_test_final, f1_test_final, f1_valid_last

def compute_test(loader,model,epoch,criterion):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    Preds=[]
    Labels=[]
    Y_proba=[]
    Loss_test=[]
    for data in loader:

        data = data.to(args.device)
        out = model(data)
        y = data.y_valid
        y = y.type(torch.LongTensor).to(args.device)
        mask = (y != -2)
        loss = criterion(out[mask], y[mask])
        loss_test += loss.item()
        pred = out[mask].max(dim=1)[1]
        y = y[mask]
        correct += pred.eq(y).sum().item()
        preds=pred.cpu().long().numpy()
        labels=torch.squeeze(y).cpu()
        labels=labels.long().numpy()
        y_proba=F.softmax(out[mask],-1).cpu().detach().numpy()
        
        Preds.append(preds)
        Labels.append(labels)
        Y_proba.append(y_proba)

    preds = np.concatenate(Preds)
    labels = np.concatenate(Labels)
    y_proba = np.concatenate(Y_proba)
    return correct, loss_test, preds, labels, y_proba



if __name__ == '__main__':
    # Model training
    y_=[]
    AP=[]
    ROC=[]
    F1=[]
    for r in range(args.runs):
        torch.manual_seed(r)
        np.random.seed(r)
        split_idx = dataset.get_idx_split()
        print(len(split_idx["valid"]))
        print(dataset[0])
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = 2)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = 2)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = 2)
        deg = torch.cat([
            utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in dataset])
        ap, roc, f1, f1_valid_last = train(train_loader,val_loader,test_loader, deg)
        if (f1_valid_last==0):
            print("failed")
            continue
            # print("AP,ROC,F1 on average:",np.mean(AP),np.std(AP),np.mean(ROC),np.std(ROC),np.mean(F1),np.std(F1))
            # raise ValueError('training failed')
        ROC.append(roc)
        AP.append(ap)
        F1.append(f1)
        print(AP,ROC,F1)
        
    print("10 times AP,ROC,F1 for Class 1:",np.mean(AP),np.std(AP),np.mean(ROC),np.std(ROC),np.mean(F1),np.std(F1))
