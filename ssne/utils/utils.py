# -*- coding: utf-8 -*-
import torch
import numpy as np


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def init_snapshoot(data_name, snap_time):
    #服务器配置
    import os
    snap_root = 'checkpoints/' + data_name.split('/')[-1] + '/' + str(snap_time)
    if not os.path.exists(snap_root):
        os.makedirs(snap_root)
    fout = open(snap_root + '/result.log', 'w')
    return fout, snap_root
    #本地配置
    


def preprocess_Sneae(dataset, data):
    '''
    dataset: Dataset class
    data: mini-batch
    '''
    adj = torch.FloatTensor(dataset.adj[data.numpy(), :])
    adj_T = torch.FloatTensor(dataset.adj.T[data.numpy(), :])
    return data, adj, adj_T


def get_tuple(adj, threshold=200):
        tuple_list = []
        for i in xrange(adj.shape[0]):
            poss = np.where(adj[i, :] == 1)[0]
            negs = np.where(adj[i, :] == -1)[0]
            tuples = []
            for pos in poss:
                for neg in negs:
                    tuples.append((pos + 1, neg + 1))
            if len(negs) == 0:
                for pos in poss:
                    tuples.append((pos + 1, 0))

            if len(tuples) > threshold:
                rand = np.random.permutation(len(tuples))[0:threshold]
                newTuples = []
                for ind in rand:
                    newTuples.append(tuples[ind])
                tuples = newTuples
            for tup in tuples:
                tuple_list.append([i + 0, tup[0], tup[1]])
        return tuple_list

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
    