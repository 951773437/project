import numpy as np
import scipy.sparse as sp
import torch

#标签转化成one-hot label形式，在符号网络中暂时使用不到这个标签，
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

#加载数据
def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    #print(dataset)
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # frame：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    print(idx_features_labels)
    #总结feature：节点
    #label 标签
    #adj：节点和关系构建的邻接矩阵
    
    #需要修改输入的代码。
    #常常需要把一个稀疏的np.array压缩，这时候就用到scipy库中的sparse.csr_matrix
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    #这是个二维数组  取所有行  列从下标为1的列开始取  一直到倒数第二个
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    
    #数据预处理阶段
    #对结果进行处理-》执行gcn
    # build symmetric adjacency matrix 构建一个对称的邻接矩阵
    # 目的将有向图的邻接矩阵变成无向图的邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #特征进行归一化处理
    features = normalize(features)
    #邻接矩阵进行归一化处理
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    #这三个参数自己调试
    idx_train = range(140) #训练集
    idx_val = range(200, 500) #评估集
    idx_test = range(500, 1500) #测试集、
    
    #输入转化成tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    #np.savetxt('features.txt',features)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #np.savetxt('adj.txt',adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


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
