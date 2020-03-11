# -*- coding: utf-8 -*-
import dgl
import math
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #__import__('pdb').set_trace()
        #print(input)
        #print('333')
        support = torch.mm(input, self.weight)
        #print(adj)
        output = torch.spmm(adj, support) + 1
        print(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class StatusGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        === status ranking ===
        === F function ===
        torch.cat是将两个张量（tensor）拼接在一起
        """
        emb = self.embedding_manager
        emb_rank_r = emb.ru_embeddings(rank[0])
        emb_rank_a = emb.au_embeddings(rank[1])
        emb_rank_acc = emb.au_embeddings(rank[2])
        rank_q, rank_q_len = rank[3], rank[4]

        rank_q_output, _ = emb.ubirnn(rank_q, emb.init_hc(rank_q.size(0)))
        rank_q_pad = Variable(torch.zeros(
            rank_q_output.size(0)
            , 1
            , rank_q_output.size(2))).cuda()
        rank_q_output = torch.cat(
            (rank_q_pad, rank_q_output)
            , 1)

        rank_q_len = rank_q_len.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)

        emb_rank_q = rank_q_output.gather(1, rank_q_len.detach())

        low_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q.squeeze(), emb_rank_a]
            , dim=1) \
            .unsqueeze(1)
        high_rank_mat = torch.stack(
            [emb_rank_r, emb_rank_q.squeeze(), emb_rank_acc]
            , dim=1) \
            .unsqueeze(1)

        low_score = torch.cat([
            self.convnet1(low_rank_mat)
            , self.convnet2(low_rank_mat)
            , self.convnet3(low_rank_mat)]
            , dim=2).squeeze()
        high_score = torch.cat([
            self.convnet1(high_rank_mat)
            , self.convnet2(high_rank_mat)
            , self.convnet3(high_rank_mat)]
            , dim=2).squeeze()

        low_score = self.fc_new_2(
            self.fc_new_1(low_score.squeeze()).squeeze()).squeeze()
        high_score = self.fc_new_2(
            self.fc_new_1(high_score.squeeze()).squeeze()).squeeze()

        rank_loss = torch.sum(F.sigmoid(low_score - high_score))
        # rank_loss = F.sigmoid(rank_loss)
        print("Rank loss: {:.6f}".format(rank_loss.data[0]))

        return rank_loss

        
        
        
        
        #print(input)
        #print('333')
        support = torch.mm(input, self.weight)
        #print(adj)
        output = torch.spmm(adj, support) + 1
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TransELayer(Module):
    '''h,r,t'''
    #h:embedding
    #r:sign
    #t:embedding
    #添加损失函数。
    """ TransE """
    def __init__(self, in_features, out_features, bias=True):
        super(TransELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #定义h，r,t
        #self.h_embedding = nn.Em
        #self.r_embedding = nn.Embedding(3, config.dimension)
        #self.t_embedding = in_features
        #self.init_model_weight()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        #self.reset_parameters()
    
    def calc(self, h, r, t, sign):
        return torch.abs(h + sign.view(-1, 1, 1) * r - t)
    
    
    
        
    def forward(self,input,adj):
        #h, r, t, sign, negs_r, negs_t = data
        h = input
        r = adj
        '''
        针对词向量有一个专门的层nn.Embedding，
        用来实现词与词向量的映射。
        nn.Embedding具有一个权重，形状是(num_embeddings,embedding_dimension)。
        例如，输入10个词，每个词用2维向量表征，对应的权重就是一个10x2的矩阵。
        如果Embedding层的输入形状为NxM（N是batch size,M是序列的长度），
        则输出的形状是N*M*embedding_dimension.
        '''  
        #transe的计算：h,r,t
        #符号从
        #h_emb = self.h_embedding(h).view(-1, 1, self.config.dimension)
        #r_emb = self.r_embedding(r).view(-1, 1, self.config.dimension)
        # r_emb = torch.abs(self.r_embedding(r).view(-1, 1, self.config.dimension))
        #t_emb = self.t_embedding(t).view(-1, 1, self.config.dimension)
        #negs_r_emb = self.r_embedding(negs_r)
        #negs_t_emb = self.t_embedding(negs_t)
        
        #torch.sum用法：计算score
        #torch.sum(input, dim, out=None) → Tensor
        #input (Tensor) – 输入张量
        #dim (int) – 缩减的维度
        #out (Tensor, optional) – 结果张量
        #torch sum 按行求和
        __import__('pdb').set_trace()
        pos_score = torch.sum(self.calc(input, adj, input), 2)
        neg_score = torch.sum(self.calc(input, adj, input), 2)
        
        margin_loss = self.get_margin_split_loss(pos_score, neg_score, sign, negs_r)
        # print margin_loss
        return margin_loss
    
    #添加损失函数值
    def get_margin_split_loss(self, pos_loss, neg_loss, sign, neg_r):
        margin_tensor = self.config.pos_margin * (neg_r == 1).float() \
            + self.config.neg_margin * (neg_r == 0).float() + self.config.zero_margin * (neg_r == 2).float()
        # y = sign.view(-1, 1) * 100
        y = Variable(torch.Tensor([1]).cuda())
        return torch.sum(torch.max(Variable(torch.FloatTensor([0]).cuda()), margin_tensor + y*(pos_loss - neg_loss)))

    def init_model_weight(self):
        #均匀分布 ~ U(−a,a)
        nn.init.xavier_uniform(self.h_embedding.weight.data)
        nn.init.xavier_uniform(self.r_embedding.weight.data)
        nn.init.xavier_uniform(self.t_embedding.weight.data)

    def init_model_weight_by_uniform(self):
        nn.init.uniform(self.h_embedding.weight, a=-(6.0 / self.config.dimension), b=(6.0 / self.config.dimension))
        nn.init.uniform(self.r_embedding.weight, a=0, b=(6.0 / self.config.dimension))

    def calc(self, h, r, t):#, sign
        #return torch.abs(h + sign.view(-1, 1, 1) * r - t)
        return torch.abs(h + r - t)

    def regular_loss(self, embedding):
        #ReLU的有效性体现在两个方面：
        #克服梯度消失的问题
        #加快训练速度
        return torch.sum(F.relu((-1 * embedding)))

    def get_embedding(self):
        return torch.cat((self.h_embedding.weight, self.t_embedding.weight), dim=1).cpu().data.numpy()
        # return self.h_embedding.weight.cpu().data.numpy()
        


class kTupleV3(Module):
    """triplet use TransE
    (h,l,t)
    """

    def __init__(self, config):
        super(kTupleV3, self).__init__()
        self.config = config
        self.model_name = 'kTupleV3'
        self.h_embedding = nn.Embedding(config.N, config.dimension)
        self.r_embedding = nn.Embedding(3, config.dimension)
        self.t_embedding = nn.Embedding(config.N, config.dimension)
        self.dropout = nn.Dropout(0.5)
        self.init_model_weight()

    def get_margin_loss(self, pos_loss, neg_loss, sign):
        #criterion：计算输入x1，x2（2个1D张量）与y（1或-1）的损失
        #计算两个向量之间的相似度，
        #当两个向量之间的距离大于 margin,则 loss 为正，小于margin，loss 为 0
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        # loss = criterion(pos_loss, neg_loss, Variable(torch.Tensor([-1])).cuda())
        sign_w = sign - (sign == -1).float() * 1
        y = Variable(torch.Tensor([-1]).cuda())
        sign_mask_neg = sign + (sign == -1).float()
        sign_mask_pos = sign - (sign == 1).float()
        loss = criterion(pos_loss, neg_loss, y)
        return loss

    def get_margin_split_loss(self, pos_loss, neg_loss, sign, neg_r):
        margin_tensor = self.config.pos_margin * (neg_r == 1).float() \
            + self.config.neg_margin * (neg_r == 0).float() + self.config.zero_margin * (neg_r == 2).float()
        # y = sign.view(-1, 1) * 100
        y = Variable(torch.Tensor([1]).cuda())
        return torch.sum(torch.max(Variable(torch.FloatTensor([0]).cuda()), margin_tensor + y*(pos_loss - neg_loss)))

    def init_model_weight(self):
        #均匀分布 ~ U(−a,a)
        nn.init.xavier_uniform(self.h_embedding.weight.data)
        nn.init.xavier_uniform(self.r_embedding.weight.data)
        nn.init.xavier_uniform(self.t_embedding.weight.data)

    def init_model_weight_by_uniform(self):
        nn.init.uniform(self.h_embedding.weight, a=-(6.0 / self.config.dimension), b=(6.0 / self.config.dimension))
        nn.init.uniform(self.r_embedding.weight, a=0, b=(6.0 / self.config.dimension))

    def calc(self, h, r, t, sign):
        return torch.abs(h + sign.view(-1, 1, 1) * r - t)
        # return torch.abs(h + r - t)

    def regular_loss(self, embedding):
        #ReLU的有效性体现在两个方面：
        #克服梯度消失的问题
        #加快训练速度
        return torch.sum(F.relu((-1 * embedding)))

    def forward(self, data):
        h, r, t, sign, negs_r, negs_t = data
        sign = sign.float()
        h_emb = self.h_embedding(h).view(-1, 1, self.config.dimension)
        r_emb = self.r_embedding(r).view(-1, 1, self.config.dimension)
        # r_emb = torch.abs(self.r_embedding(r).view(-1, 1, self.config.dimension))
        t_emb = self.t_embedding(t).view(-1, 1, self.config.dimension)
        negs_r_emb = self.r_embedding(negs_r)
        negs_t_emb = self.t_embedding(negs_t)
        pos_score = torch.sum(self.calc(h_emb, r_emb, t_emb, sign), 2)
        neg_score = torch.sum(self.calc(h_emb, negs_r_emb, negs_t_emb, sign), 2)
        # __import__('pdb').set_trace()
        margin_loss = self.get_margin_split_loss(pos_score, neg_score, sign, negs_r)
        # print margin_loss
        return margin_loss

    def get_embedding(self):
        return torch.cat((self.h_embedding.weight, self.t_embedding.weight), dim=1).cpu().data.numpy()
        # return self.h_embedding.weight.cpu().data.numpy()
