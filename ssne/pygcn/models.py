import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    #输入参数
    #nfeat：底层节点的参数，feature的个数
    #nhid：隐层节点个数
    #nclass：最终的分类数
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout


    def forward(self, x, adj):
        #print('222')
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
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