# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .BasicModule import BasicModule



class kTuple(BasicModule):
    """triplet use TransE
    """

    def __init__(self, config):
        super(kTuple, self).__init__()
        self.config = config
        self.model_name = 'kTuple'
        self.h_embedding = nn.Embedding(config.N + 2, config.dimension)
        self.r_embedding = nn.Embedding(2, config.dimension)
        self.t_embedding = nn.Embedding(config.N + 2, config.dimension)
        self.dropout = nn.Dropout(0.5)
        self.init_model_weight()

    def get_margin_loss(self, pos_loss, neg_loss, sign):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        # loss = criterion(pos_loss, neg_loss, Variable(torch.Tensor([-1])).cuda())
        sign_w = sign - (sign == -1).float() * 1
        y = Variable(torch.Tensor([-1]).cuda())
        sign_mask_neg = sign + (sign == -1).float()
        sign_mask_pos = sign - (sign == 1).float()
        loss = criterion(pos_loss, neg_loss, y)
        return loss

    def init_model_weight(self):
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
        return torch.sum(F.relu((-1 * embedding)))

    def forward(self, data, negs):
        h, r, t, sign = data
        sign = sign.float()
        # __import__('pdb').set_trace()
        h_emb = self.h_embedding(h).view(-1, 1, self.config.dimension)
        r_emb = self.r_embedding(r).view(-1, 1, self.config.dimension)
        # r_emb = torch.abs(self.r_embedding(r).view(-1, 1, self.config.dimension))
        t_emb = self.t_embedding(t).view(-1, 1, self.config.dimension)
        negs_emb = self.t_embedding(negs)
        pos_score = torch.sum(torch.mean(self.calc(h_emb, r_emb, t_emb, sign), 1), 1)
        neg_score = torch.sum(torch.mean(self.calc(h_emb, r_emb, negs_emb, sign), 1), 1)
        # pos_loss = (r == 1).float() * torch.sum(torch.abs(
        #   self.h_embedding(h) + self.r_embedding(r) - self.h_embedding(t)), 1)
        # neg_loss = (r == 0).float() * torch.sum(torch.abs(
        #   self.h_embedding(h) - self.r_embedding(r) - self.h_embedding(t)), 1)
        # return torch.sum(F.relu(pos_loss + neg_loss)) + torch.sum(F.relu(-1 * self.r_embedding.weight))
        margin_loss = self.get_margin_loss(pos_score, neg_score, sign)
        regular_r_loss = torch.sum(F.relu(-1 * self.r_embedding(Variable(torch.LongTensor([0]).cuda())))) \
            + torch.sum(F.relu(self.r_embedding(Variable(torch.LongTensor([1]).cuda()))))
        return margin_loss

    def get_embedding(self):
        return torch.cat((self.h_embedding.weight, self.t_embedding.weight), dim=1).cpu().data.numpy()
        # return self.h_embedding.weight.cpu().data.numpy()


class kTupleV1(BasicModule):
    """triplet use TransR
    """

    def __init__(self, config):
        super(kTupleV1, self).__init__()
        self.config = config
        self.model_name = 'kTupleV1'
        self.rel_dim = config.dimension
        self.ent_dim = config.dimension
        self.h_embedding = nn.Embedding(config.N, self.ent_dim)
        self.r_embedding = nn.Embedding(3, self.rel_dim)
        self.t_embedding = nn.Embedding(config.N, self.ent_dim)
        self.transfer_matrix = nn.Embedding(3, self.ent_dim * self.rel_dim)
        self.dropout = nn.Dropout(0.5)
        self.init_model_weight()

    def get_margin_loss(self, pos_loss, neg_loss, sign):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        # loss = criterion(pos_loss, neg_loss, Variable(torch.Tensor([-1])).cuda())
        sign_w = sign - (sign == -1).float() * 9
        y = Variable(torch.Tensor([-1]).cuda())
        sign_mask_neg = sign + (sign == -1).float()
        sign_mask_pos = sign - (sign == 1).float()
        loss = criterion(pos_loss, neg_loss, y)
        return loss

    def get_margin_split_loss(self, pos_loss, neg_loss, sign, neg_r):
        margin_tensor = self.config.pos_margin * (neg_r == 1).float() \
            + self.config.neg_margin * (neg_r == 0).float() + self.config.zero_margin * (neg_r == 2).float()
        return torch.sum(torch.max(Variable(torch.FloatTensor([0]).cuda()), margin_tensor + pos_loss - neg_loss))

    def init_model_weight(self):
        nn.init.xavier_uniform(self.h_embedding.weight.data)
        nn.init.xavier_uniform(self.r_embedding.weight.data)
        nn.init.xavier_uniform(self.t_embedding.weight.data)
        nn.init.xavier_uniform(self.transfer_matrix.weight.data)

    def init_model_weight_by_uniform(self):
        nn.init.uniform(self.h_embedding.weight, a=-(6.0 / self.config.dimension), b=(6.0 / self.config.dimension))
        nn.init.uniform(self.r_embedding.weight, a=0, b=(6.0 / self.config.dimension))

    def _transfer(self, transfer_matrix, embeddings):
        return torch.matmul(transfer_matrix, embeddings)

    def calc(self, h, r, t, sign):
        return torch.abs(h + sign.view(-1, 1, 1) * r - t)
        # return torch.abs(h + r - t)

    def regular_loss(self, embedding):
        return torch.sum(F.relu((-1 * embedding)))

    def forward(self, data):
        h, r, t, sign, negs_r, negs_t = data
        # __import__('pdb').set_trace()
        sign = sign.float()
        neg_num = int(negs_r.shape[1])
        p_h_emb = self.h_embedding(h).view(-1, self.ent_dim, 1)
        p_r_emb = self.r_embedding(r).view(-1, self.rel_dim)
        p_t_emb = self.t_embedding(t).view(-1, self.ent_dim, 1)
        n_r_emb = self.r_embedding(negs_r).view(-1, self.rel_dim)
        n_t_emb = self.t_embedding(negs_t).view(-1, self.ent_dim, 1)
        p_matrix = self.transfer_matrix(r).view(-1, self.rel_dim, self.ent_dim)
        n_matrix = self.transfer_matrix(negs_r).view(-1, self.rel_dim, self.ent_dim)
        p_h = self._transfer(p_matrix, p_h_emb).view(-1, 1, self.rel_dim)
        p_t = self._transfer(p_matrix, p_t_emb).view(-1, 1, self.rel_dim)
        p_r = p_r_emb.view(-1, 1, self.rel_dim)
        n_h = p_h.view(-1, 1, self.rel_dim)
        n_t = self._transfer(n_matrix, n_t_emb).view(-1, neg_num, self.rel_dim)
        n_r = n_r_emb.view(-1, neg_num, self.rel_dim)

        # __import__('pdb').set_trace()
        pos_score = torch.sum(torch.mean(self.calc(p_h, p_r, p_t, sign).view(-1, 1, self.rel_dim), 1), 1)
        neg_score = torch.sum(torch.mean(self.calc(n_h, n_r, n_t, sign), 1), 1)
        # pos_loss = (r == 1).float() * torch.sum(torch.abs(
        #   self.h_embedding(h) + self.r_embedding(r) - self.h_embedding(t)), 1)
        # neg_loss = (r == 0).float() * torch.sum(torch.abs(
        #   self.h_embedding(h) - self.r_embedding(r) - self.h_embedding(t)), 1)
        # return torch.sum(F.relu(pos_loss + neg_loss)) + torch.sum(F.relu(-1 * self.r_embedding.weight))
        margin_loss = self.get_margin_loss(pos_score, neg_score, sign)
        regular_r_loss = torch.sum(F.relu(-1 * self.r_embedding(Variable(torch.LongTensor([0]).cuda())))) \
            + torch.sum(F.relu(self.r_embedding(Variable(torch.LongTensor([1]).cuda()))))
        return margin_loss

    def get_embedding(self):
        return torch.cat((self.h_embedding.weight, self.t_embedding.weight), dim=1).cpu().data.numpy()
        # return self.h_embedding.weight.cpu().data.numpy()


class kTupleV2(BasicModule):
    """triplet use TransE and split pos&neg node
    """

    def __init__(self, config):
        super(kTupleV2, self).__init__()
        self.config = config
        self.model_name = 'kTupleV2'
        self.rel_dim = config.dimension
        self.ent_dim = config.dimension
        self.pos_embedding = nn.Embedding(config.N, self.ent_dim)
        self.neg_embedding = nn.Embedding(config.N, self.ent_dim)
        self.r_embedding = nn.Embedding(2, self.rel_dim)
        self.dropout = nn.Dropout(0.5)
        self.init_model_weight()

    def get_margin_loss(self, pos_loss, neg_loss, sign):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        # loss = criterion(pos_loss, neg_loss, Variable(torch.Tensor([-1])).cuda())
        sign_w = sign - (sign == -1).float() * 9
        y = Variable(torch.Tensor([-1]).cuda())
        sign_mask_neg = sign + (sign == -1).float()
        sign_mask_pos = sign - (sign == 1).float()
        loss = criterion(pos_loss, neg_loss, y)
        return loss

    def init_model_weight(self):
        nn.init.xavier_uniform(self.pos_embedding.weight.data)
        nn.init.xavier_uniform(self.neg_embedding.weight.data)
        nn.init.xavier_uniform(self.r_embedding.weight.data)

    def init_model_weight_by_uniform(self):
        nn.init.uniform(self.h_embedding.weight, a=-(6.0 / self.config.dimension), b=(6.0 / self.config.dimension))
        nn.init.uniform(self.r_embedding.weight, a=0, b=(6.0 / self.config.dimension))

    def _transfer(self, transfer_matrix, embeddings):
        return torch.matmul(transfer_matrix, embeddings)

    def calc(self, h, r, t, sign):
        return torch.abs(h + sign.view(-1, 1, 1) * r - t)
        # return torch.abs(h + r - t)

    def regular_loss(self, embedding):
        return torch.sum(F.relu((-1 * embedding)))

    def where(self, cond):
        cond = cond.long()
        return torch.nonzero(cond).squeeze(1)

    def forward(self, data, negs):
        h, r, t, sign = data
        sign = sign.float()
        index = self.where(r == 1)
        # __import__('pdb').set_trace()
        p_h_emb = self.pos_embedding(h[index]).view(-1, 1, self.config.dimension)
        r_emb = self.r_embedding(r[index]).view(-1, 1, self.config.dimension)
        p_t_emb = self.pos_embedding(t[index]).view(-1, 1, self.config.dimension)
        p_neg_emb = self.pos_embedding(negs[index])
        p_pos_score = torch.sum(torch.mean(self.calc(p_h_emb, r_emb, p_t_emb, sign[index]), 1), 1)
        p_neg_score = torch.sum(torch.mean(self.calc(p_h_emb, r_emb, p_neg_emb, sign[index]), 1), 1)
        p_margin_loss = self.get_margin_loss(p_pos_score, p_neg_score, sign[index])
        if int(index.size()[0]) == 32:
            return p_margin_loss
        index = self.where(r == 0)
        n_h_emb = self.neg_embedding(h[index]).view(-1, 1, self.config.dimension)
        n_t_emb = self.neg_embedding(t[index]).view(-1, 1, self.config.dimension)
        n_neg_emb = self.neg_embedding(negs[index])
        r_emb = self.r_embedding(r[index]).view(-1, 1, self.config.dimension)
        n_pos_score = torch.sum(torch.mean(self.calc(n_h_emb, r_emb, n_t_emb, sign[index]), 1), 1)
        n_neg_score = torch.sum(torch.mean(self.calc(n_h_emb, r_emb, n_neg_emb, sign[index]), 1), 1)
        n_margin_loss = self.get_margin_loss(n_pos_score, n_neg_score, sign[index])

        # pos_loss = (r == 1).float() * torch.sum(torch.abs(
        #   self.h_embedding(h) + self.r_embedding(r) - self.h_embedding(t)), 1)
        # neg_loss = (r == 0).float() * torch.sum(torch.abs(
        #   self.h_embedding(h) - self.r_embedding(r) - self.h_embedding(t)), 1)
        # return torch.sum(F.relu(pos_loss + neg_loss)) + torch.sum(F.relu(-1 * self.r_embedding.weight))
        regular_r_loss = torch.sum(F.relu(-1 * self.r_embedding(Variable(torch.LongTensor([0]).cuda())))) \
            + torch.sum(F.relu(self.r_embedding(Variable(torch.LongTensor([1]).cuda()))))
        return p_margin_loss + n_margin_loss

    def get_embedding(self):
        # return torch.cat((self.h_embedding.weight, self.t_embedding.weight), dim=1).cpu().data.numpy()
        return torch.cat((self.pos_embedding.weight, self.neg_embedding.weight), dim=1).cpu().data.numpy()


class kTupleV3(BasicModule):
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


class kTuple_R(BasicModule):
    """triplet use TransE - zero relation
    """

    def __init__(self, config):
        super(kTuple_R, self).__init__()
        self.config = config
        self.model_name = 'kTuple_R'
        self.h_embedding = nn.Embedding(config.N, config.dimension)
        self.r_embedding = nn.Embedding(3, config.dimension)
        self.t_embedding = nn.Embedding(config.N, config.dimension)
        self.dropout = nn.Dropout(0.5)
        self.init_model_weight()

    def get_margin_loss(self, pos_loss, neg_loss, sign):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        # loss = criterion(pos_loss, neg_loss, Variable(torch.Tensor([-1])).cuda())
        sign_w = sign - (sign == -1).float() * 1
        y = Variable(torch.Tensor([-1]).cuda())
        sign_mask_neg = sign + (sign == -1).float()
        sign_mask_pos = sign - (sign == 1).float()
        loss = criterion(pos_loss, neg_loss, y)
        return loss

    def get_margin_split_loss(self, pos_loss, neg_loss, sign, r):
        margin_tensor = self.config.pos_margin * (r == 1).float() \
            + self.config.neg_margin * (r == 0).float() + self.config.zero_margin * (r == 2).float()
        # y = sign.view(-1, 1) * 100
        y = Variable(torch.Tensor([1]).cuda())
        return torch.sum(torch.max(Variable(torch.FloatTensor([0]).cuda()), margin_tensor + y*(pos_loss - neg_loss)))

    def init_model_weight(self):
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
        neg_score = torch.sum(self.calc(h_emb, r_emb, negs_t_emb, sign), 2)
        # __import__('pdb').set_trace()
        margin_loss = self.get_margin_split_loss(pos_score, neg_score, sign, r.view(-1, 1))
        # print margin_loss
        return margin_loss

    def get_embedding(self):
        return torch.cat((self.h_embedding.weight, self.t_embedding.weight), dim=1).cpu().data.numpy()
        # return self.h_embedding.weight.cpu().data.numpy()
