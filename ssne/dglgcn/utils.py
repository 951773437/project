# -*- coding: utf-8 -*-
from dgl.data import citation_graph as citegrh
import torch
import dgl
from dgl import DGLGraph

#加载数据
def load_cora_data():
    data = citegrh.load_cora()
    print(data.features)
    features = torch.FloatTensor(data.features)
    #print(data.features)
    labels = torch.LongTensor(data.labels)
    #print(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    print(data.train_mask)
    g = data.graph
    # add self loop
    #g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

def load_data(self):
    
    
    
    features = torch.FloatTensor(self.G)
    #print(data.features)
    labels = torch.LongTensor(data.labels)
    #print(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    #g = data.graph
    # add self loop
    #g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask