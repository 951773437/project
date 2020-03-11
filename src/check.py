# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import dok_matrix
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics



#data preprocessing
def preprocessing(train_data):
    
        feature = sp.csr_matrix(train_data.G.pos_adjmatrix, dtype=np.float32).T
        feature_pos = sp.csr_matrix(train_data.G.pos_adjmatrix, dtype=np.float32).T
        feature_neg = sp.csr_matrix(train_data.G.neg_adjmatrix, dtype=np.float32).T
        
        adj = train_data.G.all_matrix
        adj_pos = train_data.adj_pos
        adj_neg = train_data.adj_neg
        
        adj_pos = adj_pos + adj_pos.T.multiply(adj_pos.T > adj_pos) - adj_pos.multiply(adj_pos.T > adj_pos)
        adj_neg = adj_neg + adj_neg.T.multiply(adj_neg.T > adj_neg) - adj_neg.multiply(adj_neg.T > adj_neg)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #特征进行归一化处理
        feature = normalize(feature)
        feature_pos = normalize(feature_pos)
        feature_neg = normalize(feature_neg)
        #邻接矩阵进行归一化处理
        adj_pos = normalize(adj_pos + sp.eye(adj_pos.shape[0]))
        adj_neg = normalize(adj_neg + sp.eye(adj_neg.shape[0]))
        adj = normalize(adj + sp.eye(adj.shape[0]))
        
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj_pos = sparse_mx_to_torch_sparse_tensor(adj_pos)
        adj_neg = sparse_mx_to_torch_sparse_tensor(adj_neg)
        
        features = torch.FloatTensor(np.array(feature.todense()))
        features_pos = torch.FloatTensor(np.array(feature_pos.todense()))
        features_neg = torch.FloatTensor(np.array(feature_neg.todense()))
        
        labels = [0,1]
        labels = torch.LongTensor(labels)
        
        return features,features_pos,features_neg,adj,adj_pos,adj_neg,labels
        

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# 在邻接矩阵中加入自连接
def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # 对加入自连接的邻接矩阵进行对称归一化处理
    adj = normalize_adj(adj, symmetric)
    return adj

class SignGraph:
    def __init__(self, filename, seq='\t', split_ratio=0.8, weight=False, directed=False,
                 remove_self_loop=False, train_with_only_trainset=False):
        self.adj_matrix = None
        self.train_edges = None
        self.test_edges = None
        self.g = nx.DiGraph()
        self.weight = weight
        self.directed = directed
        #get labels
        print(filename)
        edge_pd = pd.read_csv(filename, sep='\t', header=None, comment='%')
        edges = np.array(edge_pd)
        #print(edges)
        self.labels = list(edge_pd.columns.values)
        # edges = [map(int, line.strip().split(seq)) for line in edgefile]
        #print(edges)
        self.new_all_edges = edges
        self.all_edges = edges

        source_node, targt_node, sign = zip(*edges)
        nodes = list(set(source_node) | set(targt_node))
        #print(len(nodes)-1)
        #print(nodes)
        if len(nodes)-1 != max(nodes):
            print ('======= len(nodes)-1 != max(nodes)! ==========')
        # print len(nodes)
        for node in range(max(nodes)+1):
            self.g.add_node(node)
        self.getGraph(weight=self.weight, directed=self.directed)
        if remove_self_loop:
            self.remove_self_loop()
        self.g, self.mapping = self.convert_node_labels_to_integers(self.g)
        self.all_edges = [(u, v, d.get('sign', 1)) for u, v, d in self.g.edges(list(self.g), data=True)]
        #print(self.all_edges)
        edges = shuffle(self.all_edges)
        #print(edges)
        training_size = int(split_ratio * len(edges))
        self.train_edges = edges[:training_size]
        self.test_edges = edges[training_size:]
        if split_ratio == 1.0:
            self.test_edges = self.train_edges
        if train_with_only_trainset:
            self.all_edges = self.train_edges
            self.g.remove_edges_from(list(self.g.edges()))
            self.getGraph(weight=self.weight, directed=self.directed)
        self.adj_matrix = self.to_adjmatrix()
        self.poslist = self.get_poslist()
        self.neglist = self.get_neglist()
        self.pos_adjmatrix = self.to_posadjmatrix()
        self.neg_adjmatrix = self.to_negadjmatrix()
        self.all_matrix = self.to_allmatrix()
        #print(self.poslist)
        #print(self.neglist)

    #得到正边集合
    def get_poslist(self):
        pos_edges = []
        for edge in self.g.edges():
            if self.g[edge[0]][edge[1]]['sign'] == 1:
                pos_edges.append(edge)
        return pos_edges
    #得到负边集合
    def get_neglist(self):
        neg_edges = []
        for edge in self.g.edges():
            if self.g[edge[0]][edge[1]]['sign'] == -1:
                neg_edges.append(edge)
        return neg_edges
    
    def getGraph(self, weight, directed):
        for line in self.all_edges:
            if directed:
                if not weight:
                    src, tgt, sign = line
                    self.g.add_edge(src, tgt)
                    self.g[src][tgt]['weight'] = 1.0
                    self.g[src][tgt]['sign'] = sign
                else:
                    src, tgt, sign, weight = line
                    self.g.add_edge(src, tgt)
                    self.g[src][tgt]['weight'] = float(weight)
                    self.g[src][tgt]['sign'] = sign
            else:
                if not weight:
                    src, tgt, sign = line
                    self.g.add_edge(src, tgt)
                    self.g.add_edge(tgt, src)
                    self.g[src][tgt]['weight'] = 1.0
                    self.g[tgt][src]['weight'] = 1.0
                    self.g[src][tgt]['sign'] = sign
                    self.g[tgt][src]['sign'] = sign
                else:
                    src, tgt, sign, weight = line
                    self.g.add_edge(src, tgt)
                    self.g.add_edge(tgt, src)
                    self.g[src][tgt]['weight'] = float(weight)
                    self.g[tgt][src]['weight'] = float(weight)
                    self.g[src][tgt]['sign'] = sign
                    self.g[tgt][src]['sign'] = sign
        print(('node: {}, edges: {}').format(self.g.number_of_nodes(), self.g.number_of_edges()))

    def to_posadjmatrix(self):
        self.pos_adjmatrix = dok_matrix((len(self.g.nodes()), len(self.g.nodes())), np.float)
        for edge in self.g.edges():
            if self.g[edge[0]][edge[1]]['sign']==1:
                self.pos_adjmatrix[edge[0], edge[1]] = self.g[edge[0]][edge[1]]['sign']
        self.pos_adjmatrix = self.pos_adjmatrix.tocsr()
        return self.pos_adjmatrix
    
    def to_negadjmatrix(self):
        self.neg_adjmatrix = dok_matrix((len(self.g.nodes()), len(self.g.nodes())), np.float)
        for edge in self.g.edges():
            if self.g[edge[0]][edge[1]]['sign']== -1:
                #print(edge)
                self.neg_adjmatrix[edge[0], edge[1]] = self.g[edge[0]][edge[1]]['sign']
        self.neg_adjmatrix = self.neg_adjmatrix.tocsr()
        return self.neg_adjmatrix
    
    def to_adjmatrix(self):
        self.adj_matrix = dok_matrix((len(self.g.nodes()), len(self.g.nodes())), np.float)
        for edge in self.g.edges():
            self.adj_matrix[edge[0], edge[1]] = self.g[edge[0]][edge[1]]['sign']
        self.adj_matrix = self.adj_matrix.tocsr()
        return self.adj_matrix

    def to_allmatrix(self):
        self.all_matrix = dok_matrix((len(self.g.nodes()), len(self.g.nodes())), np.float)
        for edge in self.g.edges():
            if self.g[edge[0]][edge[1]]['sign']== -1:
                self.adj_matrix[edge[0], edge[1]] = -self.g[edge[0]][edge[1]]['sign']
            else:
                self.adj_matrix[edge[0], edge[1]] = self.g[edge[0]][edge[1]]['sign']
        self.all_matrix = self.all_matrix.tocsr()
        return self.all_matrix
    
    
    def remove_self_loop(self):
        self_loop_node = []
        for node in self.g.nodes():
            if node in self.g[node]:
                self.g.remove_edge(node, node)
        for node in self.g.nodes():
            if self.g.in_degree(node) == 0 and self.g.out_degree(node) == 0:
                self_loop_node.append(node)
        if len(self_loop_node) > 0:
            self.g.remove_nodes_from(self_loop_node)
            print ('Remove {} self loop'.format(self_loop_node))
        print(('node: {}, edges: {}').format(self.g.number_of_nodes(), self.g.number_of_edges()))

    def convert_node_labels_to_integers(self, G, first_label=0, ordering="default",
                                        label_attribute=None):
        """Return a copy of the graph G with the nodes relabeled using
        consecutive integers and the mapping dict.

        Parameters
        ----------
        G : graph
        A NetworkX graph

        first_label : int, optional (default=0)
        An integer specifying the starting offset in numbering nodes.
        The new integer labels are numbered first_label, ..., n-1+first_label.

        ordering : string
        "default" : inherit node ordering from G.nodes()
        "sorted"  : inherit node ordering from sorted(G.nodes())
        "increasing degree" : nodes are sorted by increasing degree
        "decreasing degree" : nodes are sorted by decreasing degree

        label_attribute : string, optional (default=None)
        Name of node attribute to store old label.  If None no attribute
        is created.

        Returns
        -------
        G : Graph
        A NetworkX graph
        mapping : dict
        A dict of {node: id}
        Notes
        -----
        Node and edge attribute data are copied to the new (relabeled) graph.
        """
        N = G.number_of_nodes() + first_label
        if ordering == "default":
            mapping = dict(zip(G.nodes(), range(first_label, N)))
        elif ordering == "sorted":
            nlist = sorted(G.nodes())
            mapping = dict(zip(nlist, range(first_label, N)))
        elif ordering == "increasing degree":
            dv_pairs = [(d, n) for (n, d) in G.degree()]
            dv_pairs.sort()  # in-place sort from lowest to highest degree
            mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
        elif ordering == "decreasing degree":
            dv_pairs = [(d, n) for (n, d) in G.degree()]
            dv_pairs.sort()  # in-place sort from lowest to highest degree
            dv_pairs.reverse()
            mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
        else:
            raise nx.NetworkXError('Unknown node ordering: %s' % ordering)
        H = nx.relabel_nodes(G, mapping)
        # create node attribute with the old label
        if label_attribute is not None:
            nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
        return H, mapping

    # def status_triplet_number(self)

# -*- coding: utf-8 -*-


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

    def link_sign_pre_pn(self,output,idx_train,idx_val,idx_test, method='concatenate_gcn'):
        
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        src, tgt, sign = zip(*self.G.train_edges)
        #__import__('pdb').set_trace()
        #numpy.concatenate((a1, a2, ...), axis=0)
        #Join a sequence of arrays along an existing axis.（按轴axis连接array组成一个新的array）
        x_train = output[src, :].cpu().detach().numpy()
        #x_train = np.concatenate(output[src, :], axis=1)
        #x_train = np.concatenate((output_pos[src, :], output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
        #print(x_train)
        #x_train = self.get_link_embedding(output_pos, idx_train, tgt, method) + self.get_link_embedding(output_neg, src, tgt, method)
        #y_train = list(sign)
        y_train = list(sign)
        src, tgt, sign = zip(*self.G.test_edges)
        x_test = output[src, :].cpu().detach().numpy()
        #x_test = np.concatenate((output[src, :]), axis=1)
        #x_test = np.concatenate((output_pos[src, :], output_neg[src, :], output_neg[tgt, :], output_neg[tgt, :]), axis=1)
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


#根据embedding去拼接节点：
def cat_neighbor_new(g, embedding, method='null'):
    """concatenate node i neighbor's embedding to node i

    Parameter
    ---------
    g: Graph
    a networkx graph

    embedding: ndarray
    a numpy ndarray which represent nodes embedding

    method: str
    "null": default, use original embedding
    "cat_pos": use positive out edges as neighbor embedding, and concatenate it with original embedding
    "cat_pos_self": like "cat_pos"
    "cat_pos_extend": like "cat_pos", but use in and out edges

    Return
    ------
    emb: ndarray
    the embedding of nodes while concatenating neighbor nodes' embedding

    Notes
    -----
    ===2018.09.25
    only concatenate positive neighbor
    1. negative link?
    2. no out link?
    cat_pos_neg > cat_pos_self = cat_pos_extend > cat_pos
    """
    embedding = embedding.data.numpy()
    neighbor_emb = np.zeros_like(embedding)
    for node in g.nodes():
        neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
        if len(neighbor_node) == 0:
            neighbor_emb[node] = embedding[node]
        else:
            neighbor_emb[node] = np.sum(embedding[neighbor_node], axis=0)
    #print(np.concatenate((embedding, neighbor_emb), axis=1))
    return np.concatenate((embedding, neighbor_emb), axis=1)


def cat_neighbor(g, embedding, method='null'):
    """concatenate node i neighbor's embedding to node i

    Parameter
    ---------
    g: Graph
    a networkx graph

    embedding: ndarray
    a numpy ndarray which represent nodes embedding

    method: str
    "null": default, use original embedding
    "cat_pos": use positive out edges as neighbor embedding, and concatenate it with original embedding
    "cat_pos_self": like "cat_pos"
    "cat_pos_extend": like "cat_pos", but use in and out edges

    Return
    ------
    emb: ndarray
    the embedding of nodes while concatenating neighbor nodes' embedding

    Notes
    -----
    ===2018.09.25
    only concatenate positive neighbor
    1. negative link?
    2. no out link?
    cat_pos_neg > cat_pos_self = cat_pos_extend > cat_pos
    """
    neighbor_emb = np.zeros_like(embedding)
    if method == 'null':
        return embedding
    elif method == 'cat_pos':
        for node in g.nodes():
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                continue
            neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_self':
        for node in g.nodes():
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] = np.sum(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_extend':
        in_neighbor_emb = np.zeros_like(embedding)
        out_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes():
            out_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            in_neighbor_node = [src for src, tgt in g.in_edges(node) if g[src][node]['sign'] == 1]
            neighbor_node = list(set(out_neighbor_node) | set(in_neighbor_node))
            # if len(in_neighbor_node) == 0:
                # neighbor_emb[node] = embedding[node]
            # else:
                # neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
            # if len(out_neighbor_node) == 0:
                # neighbor_emb[node] = embedding[node]
            # else:
                # neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_attention':
        # def softmax(x):
            # x = x - np.max(x)
            # exp_x = np.exp(x)
            # return exp_x / np.sum(exp_x)
        for node in g.nodes():
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                # neighbor_emb[node] = embedding[node]
                continue
            else:
                # __import__('pdb').set_trace()
                relevance = np.sum(embedding[node] * embedding[neighbor_node], axis=1)
                # relevance = relevance / np.sum(relevance)
                relevance = softmax(relevance)
                # relevance = np.ones(embedding.shape[1], dtype=np.float)
                neighbor_emb[node] = np.sum(relevance * embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_degree_attention':
        for node in g.nodes():
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
                continue
            else:
                # relevance = np.sum(embedding[node] * embedding[neighbor_node], axis=1).reshape(len(neighbor_node), 1)
                relevance = np.array([1.0 * g.degree(i) for i in neighbor_node], dtype=np.float)
                relevance = softmax(relevance).reshape(len(neighbor_node), 1)
                # relevance = (relevance / np.sum(relevance)).reshape(len(neighbor_node), 1)
                # relevance = np.ones(embedding.shape[1], dtype=np.float)
                neighbor_emb[node] = np.sum(relevance * embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_neg':
        pos_neighbor_emb = np.zeros_like(embedding)
        neg_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes:
            pos_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            neg_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
            if len(pos_neighbor_node) == 0:
                pos_neighbor_emb[node] = embedding[node]
            else:
                pos_neighbor_emb[node] = np.mean(embedding[pos_neighbor_node], axis=0)
            if len(neg_neighbor_node) == 0:
                neg_neighbor_emb[node] = embedding[node]
            else:
                neg_neighbor_emb[node] = -1.0 * np.mean(embedding[neg_neighbor_node], axis=0)
        return np.concatenate((embedding, pos_neighbor_emb, neg_neighbor_emb), axis=1)
    elif method == 'cat_pos_neg_extend':
        in_pos_neighbor_emb = np.zeros_like(embedding)
        in_neg_neighbor_emb = np.zeros_like(embedding)
        out_pos_neighbor_emb = np.zeros_like(embedding)
        out_neg_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes():
            in_pos_neighbor_node = [src for src, tgt in g.in_edges(node) if g[src][tgt]['sign'] == 1]
            out_pos_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == 1]
            in_neg_neighbor_node = [src for src, tgt in g.in_edges(node) if g[src][tgt]['sign'] == -1]
            out_neg_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == -1]
            if len(in_pos_neighbor_node) == 0:
                in_pos_neighbor_emb[node] = embedding[node]
            else:
                in_pos_neighbor_emb[node] = np.mean(embedding[in_pos_neighbor_node], axis=0)
            if len(in_neg_neighbor_node) == 0:
                in_neg_neighbor_emb[node] = embedding[node]
            else:
                in_neg_neighbor_emb[node] = np.mean(embedding[in_neg_neighbor_node], axis=0)
            if len(out_neg_neighbor_node) == 0:
                out_neg_neighbor_emb[node] = embedding[node]
            else:
                out_neg_neighbor_emb[node] = np.mean(embedding[out_neg_neighbor_node], axis=0)
            if len(out_pos_neighbor_node) == 0:
                out_pos_neighbor_emb[node] = embedding[node]
            else:
                out_pos_neighbor_emb[node] = np.mean(embedding[out_pos_neighbor_node], axis=0)
        return np.concatenate((
            embedding, out_pos_neighbor_emb, in_pos_neighbor_emb, out_neg_neighbor_emb, in_neg_neighbor_emb), axis=1)
    elif method == 'cat_pos_neg_attention':
        pos_neighbor_emb = np.zeros_like(embedding)
        neg_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes():
            pos_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            neg_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
            if len(pos_neighbor_node) == 0:
                pos_neighbor_emb[node] = embedding[node]
            else:
                relevance = np.sum(embedding[node] * embedding[pos_neighbor_node], axis=1).reshape(1, -1)
                relevance = softmax(relevance).reshape(-1, 1)
                pos_neighbor_emb[node] = np.sum(relevance * embedding[pos_neighbor_node], axis=0)
            if len(neg_neighbor_node) == 0:
                neg_neighbor_emb[node] = embedding[node]
            else:
                neg_neighbor_emb[node] = -1.0 * np.mean(embedding[neg_neighbor_node], axis=0)
        return np.concatenate((embedding, pos_neighbor_emb, neg_neighbor_emb), axis=1)
    elif method == 'cat_neg':
        for node in g.nodes():
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] = np.sum(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    else:
        raise Exception('no method named: ' + method)

#判断用户使用系统，如果是linux用指令的方式，如果是windows则是变得方式
def get_System(self):
    import platform
    self.platform = platform.system()
    