# -*- coding: utf-8 -*-
from param_parser import parameter_parser
from models import StatusGCN,pnGCN
from src.check import Task
import check

from config import kTupleconfig as config
import models
import TupleData
import torch
from torch.autograd import Variable
import os
'''在python3.X中，cpickle已被别的包替换，使用以下语句即可：'''
import _pickle as pickle
import time
from datetime import datetime
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from dgl import DGLGraph
import networkx as nx
import platform

from ssne.pygcn.utils import load_data, accuracy
from ssne.pygcn.models import GCN


def train(**kwargs):
    """use the triplet like transE
    """
    config.parse(kwargs)
    dataset_name = 'kTupleDataV1'
    train_data = TupleData.TupleData(config.filename, split_ratio=config.split_ratio, neg_num=config.neg_num)
    
    features,features_pos,features_neg,adj,adj_pos,adj_neg,labels = check.preprocessing(train_data)
    
    idx_train = range(2) #训练集
    idx_val = range(2) #评估集
    idx_test = range(500, 1500) #测试集、

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #符号预测
    #使用one-hot encode 来获取标签，如果没有标签，默认fromnode,to_node来进行表示，
    #在matlab进行实验分类时使用find去除了背景类0，所以所有的类别从1开始，在matlab进行分类的时候没问题
    #但是Pytorch有个要求，在使用CrossEntropyLoss这个函数进行验证时label必须是以0开始的，所以会报错
    best_eval_dict = {'f1-micro': 0.0, 'f1-macro': 0.0}

    model = pnGCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass= 4,
            dropout=args.dropout)

    optimizer = torch.optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    task = Task(train_data.G, config)
    #cuda config

    #'''  
    for epoch in range(config.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features_pos,adj_pos,features_neg,adj_neg)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        #print(loss_train)
        optimizer.step()


        #if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            #model.eval()
            #output = model(features, adj)

        #对节点符号进行符号预测
        #如ou何对节点符号进行embedding：
        
        eval_dict = task.link_sign_pre_pn(output,idx_train,idx_val,idx_test,method='concatenate')
        
        #eval_dict = task.link_sign_pre(check.cat_neighbor_new(
            #train_data.G.g, output, method='cat_neg'),check.cat_neighbor_new(
            #train_data.G.g, output, method='cat_neg'),idx_train,idx_val,idx_test,method='concatenate')
        #print(np.all(model.get_embedding()))
        # task.link_sign_prediction_ktuple(model.get_embedding())
        #print(eval_dict)
        if config.snapshoot:
            #print('epoch {0}, loss: {1}, time: {2}'.format(epoch, total_loss, train_time), file=fout)
            print("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
                eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']))
            for key in best_eval_dict:
                if eval_dict[key] > best_eval_dict[key]:
                    for key in best_eval_dict:
                        best_eval_dict[key] = eval_dict[key]

                    #model.save(snap_root + '/{}.model'.format(config.model))
                    #model.save('/{}.model'.format(config.model))
                    break

        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        #acc_val = accuracy(output[idx_val], labels[idx_val])
        #print('Epoch: {:04d}'.format(epoch+1),
        #      'loss_train: {:.4f}'.format(loss_train.item()),
        #      'acc_train: {:.4f}'.format(acc_train.item()),
        #      'loss_val: {:.4f}'.format(loss_val.item()),
        #      'acc_val: {:.4f}'.format(acc_val.item()),
        #      'time: {:.4f}s'.format(time.time() - t))

    #pygcn
    #'''


'''
def val(model, dataloader):

'''

    #计算模型在验证集上的准确率等信息，用以辅助训练

'''

pass
'''


def test(**kwargs):
    snap_root = kwargs['snap_root']
    config_file = snap_root + '/config.pkl'
    config = pickle.load(file(config_file))
    model_file = snap_root + '/{}.model'.format(config.model)
    dataset_name = 'kTupleDataV1'
    if os.path.exists(config.filename + '_' + str(config.split_ratio)
                      + '_{}.pkl'.format(dataset_name)):
        train_data = pickle.load(file(snap_root + '/data.pkl'))
        print ('exists {}.pkl, load it!'.format(dataset_name))
        print (train_data.G.g.number_of_nodes(), train_data.G.g.number_of_edges())
    else:
        raise Exception('Data Module not exists!')
    model = getattr(models, config.model)(config)   # .eval()
    if torch.cuda.is_available():
        model.cuda()
        config.CUDA = True
    model.load_state_dict(torch.load(model_file))
    task = Task(train_data.G, config)
    task.link_sign_prediction_split(utils.cat_neighbor(
            train_data.G.g, model.get_embedding(), method='null'), method='concatenate')

def read_graph(**kwargs):
    def number_of_sign_edges(g, sign):
        count = 0
        for edge in g.edges:
            if g[edge[0]][edge[1]]['sign'] == sign:
                count += 1
        return 1.0 * count / g.number_of_edges()
    file_name = kwargs['file_name']
    from utils.SignGraph import SignGraph
    G = SignGraph(file_name, split_ratio=1.0, directed=True)
    print ("nodes: {}\nedges: {}\npos_ratio: {:.3f}\nneg_ratio: {:.3f}\n".
           format(G.g.number_of_nodes(), G.g.number_of_edges(),
                  number_of_sign_edges(G.g, 1), number_of_sign_edges(G.g, -1)))


def help():
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example:
            python {0} train --env='env0701' --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(config.__class__))
    print(source)




if __name__=='__main__':

    args = parameter_parser()
    #args.cuda = not args.no_cuda and torch.cuda.is_available()

    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #if args.cuda:
        #torch.cuda.manual_seed(args.seed)

    #sysstr = platform.system()
    #print(sysstr)
    #在本地运行程序
    train()

    #import fire
    #fire.Fire()

