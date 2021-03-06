# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GraphConvolution
from src.layers import StatusGraphConvolution
from src.layers import TransELayer


class StatusGCN(nn.Module):
    #输入参数
    #nfeat：底层节点的参数，feature的个数
    #nhid：隐层节点个数
    #nclass：最终的分类数
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(StatusGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        
        self.gc2 = GraphConvolution(nhid, nclass)
        #两层gcn层后加全连接层试试看
        self.transegc = TransELayer(nclass,nclass)
        self.dropout = dropout


    def forward(self, x , adj):
        #get feature and adj
        #x_pos, adj_pos = data.features_pos, data.adj_pos
        #x_neg, adj_neg = data.features_neg, data.adj_neg
        #x, adj = data.features, data.adj
        #using gcn
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.transegc(x,adj)
        return F.log_softmax(x, dim=1)

    #单局gcn的效果较差，使用多层gcn试试看
    '''
    def __init__(self, n_total_features, n_latent, p_drop=0.): 
        super(GCN, self).__init__() 
        self.n_total_features = n_total_features 
        self.conv1 = GCNConv(self.n_total_features, 11) 
        self.act1=nn.Sequential(nn.ReLU(), nn.Dropout(p_drop)) 
        self.conv2 = GCNConv(11, 11) 
        self.act2 = nn.Sequential(nn.ReLU(), nn.Dropout(p_drop)) 
        self.conv3 = GCNConv(11, n_latent) 
    
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
        x = self.act1(self.conv1(x, edge_index)) 
        x = self.act2(self.conv2(x, edge_index)) 
        x = self.conv3(x, edge_index) 
        return x
    '''

class pnGCN(nn.Module):
    #输入参数
    #nfeat：底层节点的参数，feature的个数
    #nhid：隐层节点个数
    #nclass：最终的分类数
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(pnGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        
        self.gc2 = GraphConvolution(nhid, nclass)
        #两层gcn层后加全连接层试试看
        self.transegc = TransELayer(nclass,nclass)
        self.dropout = dropout

    #data
    def forward(self, x_pos, adj_pos,x_neg, adj_neg):
        #get feature and adj
        #x_pos, adj_pos = data.features_pos, data.adj_pos
        #x_neg, adj_neg = data.features_neg, data.adj_neg
        #x, adj = data.features, data.adj
        #pos_embedding = get_gcn_embedding(self,x_pos,adj_pos)
        
        #using gcn
        x_pos = F.tanh(self.gc1(x_pos,adj_pos))
        x_neg = F.tanh(self.gc1(x_neg,adj_neg))
        #x = F.relu(self.gc1(x, adj))
        x_pos = F.dropout(x_pos, self.dropout, training=self.training)
        x_neg = F.dropout(x_neg, self.dropout, training=self.training)
        x_pos = self.gc2(x_pos, adj_pos)
        x_neg = self.gc2(x_neg, adj_neg)
        #拼接试一下：
        x = torch.cat((x_pos,x_neg),1)
        
        #__import__('pdb').set_trace()
        #x = self.gc2(x, adj)
        #x = self.transegc(x,adj)
        return F.log_softmax(x, dim=1)
    
    def get_gcn_embedding(self,x,adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.transegc(x,adj)
        return F.log_softmax(x, dim=1)
    
    #单局gcn的效果较差，使用多层gcn试试看
    '''
    def __init__(self, n_total_features, n_latent, p_drop=0.): 
        super(GCN, self).__init__() 
        self.n_total_features = n_total_features 
        self.conv1 = GCNConv(self.n_total_features, 11) 
        self.act1=nn.Sequential(nn.ReLU(), nn.Dropout(p_drop)) 
        self.conv2 = GCNConv(11, 11) 
        self.act2 = nn.Sequential(nn.ReLU(), nn.Dropout(p_drop)) 
        self.conv3 = GCNConv(11, n_latent) 
    
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
        x = self.act1(self.conv1(x, edge_index)) 
        x = self.act2(self.conv2(x, edge_index)) 
        x = self.conv3(x, edge_index) 
        return x
    '''
