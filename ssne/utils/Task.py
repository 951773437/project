# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics


class Task(object):
    def __init__(self, Graph, config):
        self.G = Graph
        self.g = Graph.g
        self.config = config

    def get_link_embedding(self, embedding, src, tgt, method):
        if method == 'concatenate':
            #数组拼接
            return np.concatenate((embedding[src, :], embedding[tgt, :]), axis=1)
        if method == 'concatenate_direct':
            return np.concatenate(
                (embedding[src, :self.config.dimension], embedding[tgt, self.config.dimension:]), axis=1)
        if method == '-':
            return embedding[tgt, :] - embedding[src, :]
        if method == '-_relu':
            emb = embedding[tgt, :] - embedding[src, :]
            return (abs(emb) + emb)/2
        if method == 'average':
            return (embedding[src, :] + embedding[tgt, :]) / 2
        if method == 'hadamard':
            return (embedding[src, :] * embedding[tgt, :])
        if method == 'l1':
            return np.abs(embedding[src, :] - embedding[tgt, :])
        if method == 'l2':
            return np.power(embedding[src, :] - embedding[tgt, :], 2.0)


    def link_sign_pre_con(self, output_pos , output_neg , idx_train, idx_val , idx_test, method='concatenate_gcn'):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        src, tgt, sign = zip(*self.G.train_edges)
        x_train = np.concatenate((output_pos[src, :], output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
        y_train = list(sign)
        src, tgt, sign = zip(*self.G.test_edges)
        #Join a sequence of arrays along an existing axis.（按轴axis连接array组成一个新的array）
        x_postest = np.array(output_pos[src, :])
        #x_postest = np.concatenate((output_pos[src, :]), axis=1)
        x_negtest = np.concatenate((output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
        x_test = np.concatenate((output_pos[src, :], output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
        y_test = list(sign)
        
        
        #np.savetxt("y_test.txt",y_test)
        #print(y_test)
        # y_test = (-1,1)
        # clf = OneVsRestClassifier(LogisticRegression())
        #predict()预测。利用训练得到的模型对数据集进行预测，返回预测结果。
        clf = LogisticRegression()
        clf1 = LogisticRegression()
        clf2 = LogisticRegression()
        clf.fit(x_train, y_train)
        clf1.fit(x_postest, y_train)
        clf2.fit(x_negtest, y_train)
        y_pred = clf.predict(x_test)
        y_pred_pos = clf1.predict(x_postest)
        y_pred_neg = clf2.predict(x_negtest)
        
        np.savetxt("y_pred.txt",y_pred)
        np.savetxt("y_pred_pos.txt",y_pred_pos)
        np.savetxt("y_pred_neg.txt",y_pred_neg)
        __import__('pdb').set_trace()
        # y_score = clf.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        eval_dict = {'auc': metrics.auc(fpr, tpr),
                     'f1': metrics.f1_score(y_test, y_pred),
                     'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                     'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
        '''
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                acc += 1
        '''
        print ("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
            eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']))
        return eval_dict




    #embedding符号预测
    def link_sign_pre(self, output_pos , output_neg , idx_train, idx_val , idx_test, method='concatenate_gcn'):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        src, tgt, sign = zip(*self.G.train_edges)
        x_train = np.concatenate((output_pos[src, :], output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
        #print(x_train)
        #x_train = self.get_link_embedding(output_pos, idx_train, tgt, method) + self.get_link_embedding(output_neg, src, tgt, method)
        #y_train = list(sign)
        y_train = list(sign)
        src, tgt, sign = zip(*self.G.test_edges)
        x_test = np.concatenate((output_pos[src, :], output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
        # x_test = np.concatenate((embedding[src, :], embedding[tgt, :]), axis=1)
        y_test = list(sign)
        #np.savetxt("y_test.txt",y_test)
        #print(y_test)
        # y_test = (-1,1)
        # clf = OneVsRestClassifier(LogisticRegression())
        #predict()预测。利用训练得到的模型对数据集进行预测，返回预测结果。
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        #np.savetxt("y_pred.txt",y_pred)
        # y_score = clf.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        eval_dict = {'auc': metrics.auc(fpr, tpr),
                     'f1': metrics.f1_score(y_test, y_pred),
                     'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                     'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
        '''
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                acc += 1
        '''
        print ("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
            eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']))
        return eval_dict


    def link_sign_prediction_split(self, embedding, method='concatenate'):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        src, tgt, sign = zip(*self.G.train_edges)
        x_train = self.get_link_embedding(embedding, src, tgt, method)
        y_train = list(sign)
        src, tgt, sign = zip(*self.G.test_edges)
        x_test = self.get_link_embedding(embedding, src, tgt, method)
        # x_test = np.concatenate((embedding[src, :], embedding[tgt, :]), axis=1)
        y_test = list(sign)
        # clf = OneVsRestClassifier(LogisticRegression())
        #predict()预测。利用训练得到的模型对数据集进行预测，返回预测结果。
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # y_score = clf.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        eval_dict = {'auc': metrics.auc(fpr, tpr),
                     'f1': metrics.f1_score(y_test, y_pred),
                     'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                     'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
        '''
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                acc += 1
        '''
        print ("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
            eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']))
        return eval_dict

    def link_sign_prediction_ktuple(self, embedding):
        src, tgt, y_true = zip(*self.G.test_edges)
        src_emb = embedding[src, :]
        tgt_emb = embedding[src, :]
        y_pred = (np.sum(np.abs(src_emb), axis=1) - np.sum(np.abs(tgt_emb), axis=1)) > 0
        print (metrics.f1_score(y_true, 1-y_pred, average='micro'))

    def link_sign_prediction_SneaeV4(self, embedding):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        src, tgt, sign = zip(*self.G.train_edges)
        x_train = np.concatenate((
            embedding[src, :2 * self.config.dimension], embedding[tgt, 2 * self.config.dimension:]), axis=1)
        y_train = list(sign)
        src, tgt, sign = zip(*self.G.test_edges)
        x_test = np.concatenate((
            embedding[src, :2 * self.config.dimension], embedding[tgt, 2 * self.config.dimension:]), axis=1)
        y_test = list(sign)
        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        '''
        y_score = clf.predict_proba(x_test)
        acc = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                acc += 1
        '''
        print ("link_sign_prediction  f1-micro: {:.3f}, f1-macro: {:.3f}".format(
            metrics.f1_score(y_test, y_pred, average='micro'),
            metrics.f1_score(y_test, y_pred, average='macro')))

