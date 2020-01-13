# -*- coding: utf-8 -*-
import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')

gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

#use dgl to build dcn
class GCN(nn.Module): 
    #'''
    # define the model 
    def __init__(self, 
                 #g, 图 
                 in_feats, 
                 n_hidden, #out_feats
                 #n_classes, 
                 #n_layers, 分层
                 activation #激活函数
                 #, 
                 #dropout 
                 ): 
        super(GCN, self).__init__() 
        #self.g = g 
        #
        '''
        if dropout: 
            self.dropout = nn.Dropout(p=dropout) 
        else: self.dropout = 0.
        '''
        self.layers = nn.ModuleList() # input layer 
        self.layers.append(NodeApplyModule(in_feats, n_hidden, activation))# hidden layers 
        self.apply_mod = NodeApplyModule(in_feats, n_hidden, activation)
        '''
        for i in range(n_layers - 1): 
            self.layers.append(NodeApplyModule(n_hidden, n_hidden, activation)) # output layer 
            self.layers.append(NodeApplyModule(n_hidden, n_classes))
        '''
    #forward函数设计的有问题
    #'''
    def forward(self, g, features): 
        self.g = g
        self.g.ndata['h'] = features 
        g.ndata['h'] = features 
        self.g.update_all(gcn_msg, gcn_reduce) 
        g.update_all(gcn_msg, gcn_reduce) 
        g.apply_nodes(func=self.apply_mod)
        
        return g.ndata.pop('h')
        #for idx, layer in enumerate(self.layers): # apply dropout 
            #if idx > 0 and self.dropout: 
                #self.g.ndata['h'] = self.dropout(self.g.ndata['h']) 
                #self.g.ndata['h'] = self.g.ndata['h'] 
                #self.g.update_all(gcn_msg, gcn_reduce, layer) 
            #return self.g.ndata.pop('h')
    #'''
    '''
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
    '''
    '''
    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
    '''

#输入参数
    #nfeat：底层节点的参数，feature的个数
    #nhid：隐层节点个数
    #nclass：最终的分类数    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #激活函数relu
        #nfeat：底层节点的参数，feature的个数
        #nhid：隐层节点个数
        #nclass：最终的分类数
        self.gcn1 = GCN(4, 4, F.relu)
        self.gcn2 = GCN(4, 2, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x

net = Net()

print(net)
'''
Net(
  (gcn1): GCN(
    (apply_mod): NodeApplyModule(
      (linear): Linear(in_features=1433, out_features=16, bias=True)
    )
  )
  (gcn2): GCN(
    (apply_mod): NodeApplyModule(
      (linear): Linear(in_features=16, out_features=7, bias=True)
    )
  )
)
'''