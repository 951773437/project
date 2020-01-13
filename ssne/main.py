# -*- coding: utf-8 -*-
from config import kTupleconfig as config
import models
from data import kTupleDataV1
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dglgcn.utils import load_data
from utils import Task
from utils import utils
import os
'''在python3.X中，cpickle已被别的包替换，使用以下语句即可：'''
import _pickle as pickle
import time
from datetime import datetime
import numpy as np
import dgl
import scipy.sparse as sp
import torch.nn.functional as F
from dglgcn.models import Net,GCN,net
from dgl import DGLGraph
import networkx as nx

import platform

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

#pygat
from torch.autograd import Variable
from pygat.utils import load_data, accuracy
from pygat.models import GAT, SpGAT

import argparse



# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)






def train(**kwargs):
    """use the triplet like transE
    """
    config.parse(kwargs)
    dataset_name = 'kTupleDataV1'
    train_data = kTupleDataV1(config.filename, split_ratio=config.split_ratio, neg_num=config.neg_num)

    #feature?
    feature = sp.csr_matrix(train_data.G.pos_adjmatrix, dtype=np.float32).T
    #feature = train_data.pos_feature
    #feature = sp.csr_matrix(np.array(train_data.feature), dtype=np.float32).T
    #print(feature)
    #feature_pos = sp.csr_matrix(np.array(train_data.pos_feature), dtype=np.float32).T
    feature_pos = sp.csr_matrix(train_data.G.pos_adjmatrix, dtype=np.float32).T
    #print(feature_pos.shape())
    #feature_neg = sp.csr_matrix(np.array(train_data.neg_feature), dtype=np.float32).T
    
    feature_neg = sp.csr_matrix(train_data.G.neg_adjmatrix, dtype=np.float32).T
    print(feature_neg)
    #print(feature_neg.shape())
    #feature = train_data.G.adj_matrix
    #print(feature.todense())
    '''
    feature = np.matrix([
            [i, -i]
            for i in range(train_data.adj_pos.shape[0])
        ], dtype=float)
    '''
    #adj = train_data.adj_matrix
    adj = train_data.G.all_matrix
    adj_pos = train_data.adj_pos
    #print(adj_pos.shape)
    adj_neg = train_data.adj_neg
    graph_pos = nx.from_scipy_sparse_matrix(adj_pos, create_using=nx.DiGraph())
    graph_neg = nx.from_scipy_sparse_matrix(adj_neg, create_using=nx.DiGraph())
    #数据预处理
    #数据预处理
    #对结果进行处理-》执行gcn
    # build symmetric adjacency matrix 构建一个对称的邻接矩阵
    # 目的将有向图的邻接矩阵变成无向图的邻接矩阵
    adj_pos = adj_pos + adj_pos.T.multiply(adj_pos.T > adj_pos) - adj_pos.multiply(adj_pos.T > adj_pos)
    adj_neg = adj_neg + adj_neg.T.multiply(adj_neg.T > adj_neg) - adj_neg.multiply(adj_neg.T > adj_neg)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj_pos.todense())
    #特征进行归一化处理
    feature = normalize(feature)
    feature_pos = normalize(feature_pos)
    feature_neg = normalize(feature_neg)
    #print(np.array(feature.todense()).shape)
    #print(feature.todense())
    #邻接矩阵进行归一化处理
    adj_pos = normalize(adj_pos + sp.eye(adj_pos.shape[0]))
    adj_neg = normalize(adj_neg + sp.eye(adj_neg.shape[0]))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    #print(adj_pos)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_pos = sparse_mx_to_torch_sparse_tensor(adj_pos)
    #print(adj_pos)
    adj_neg = sparse_mx_to_torch_sparse_tensor(adj_neg)
    #这三个参数自己调试
    idx_train = range(2) #训练集
    idx_val = range(2) #评估集
    idx_test = range(500, 1500) #测试集、

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    features = torch.FloatTensor(np.array(feature.todense()))
    features_pos = torch.FloatTensor(np.array(feature_pos.todense()))
    features_neg = torch.FloatTensor(np.array(feature_neg.todense()))
    #符号预测
    #使用one-hot encode 来获取标签，如果没有标签，默认fromnode,to_node来进行表示，
    #在matlab进行实验分类时使用find去除了背景类0，所以所有的类别从1开始，在matlab进行分类的时候没问题
    #但是Pytorch有个要求，在使用CrossEntropyLoss这个函数进行验证时label必须是以0开始的，所以会报错
    labels = [0,1]
    #labels = encode_onehot(labels)
    #features = torch.FloatTensor(feature)

    #print(data.features)
    labels = torch.LongTensor(labels)
    #print(data.labels)
    #mask = torch.ByteTensor(data.train_mask)
    #g = data.graph
    # add self loop
    #g.remove_edges_from(g.selfloop_edges())
    g_pos = DGLGraph(graph_pos)
    g_neg = DGLGraph(graph_neg)
    #g.add_edges(g.nodes(), g.nodes())

    #train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    #shujuxunlian
    #g, features, labels, mask = load_data(train_data)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    best_eval_dict = {'f1-micro': 0.0, 'f1-macro': 0.0}

    model = GCN(nfeat=features_pos.shape[1],
            nhid=args.hidden,
            nclass= 64,
            dropout=args.dropout)

    model2 = GCN(nfeat=features_neg.shape[1],
            nhid=args.hidden,
            nclass= 64,
            dropout=args.dropout)
    
    model3 = GAT(nfeat=features_pos.shape[1], 
                nhid=args.hidden, 
                nclass=64, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)

    optimizer = torch.optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    optimizer1 = torch.optim.Adam(model2.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(model3.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    task = Task(train_data.G, config)
    features,features_pos,features_neg,adj,labels = Variable(features),Variable(features_pos),Variable(features_neg), Variable(adj), Variable(labels)
    '''
    #cuda config
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    '''
    '''
    #pygat
    for epoch in range(config.epochs):
        t = time.time()
        model3.train()
        optimizer2.zero_grad()
        output = model3(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
    
    '''
    #'''
    #pytorch   
    for epoch in range(config.epochs):
        #for epoch in range(60):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        optimizer1.zero_grad()
        #__import__('pdb').set_trace()
        output_pos = model(features_pos, adj)
        #var.detach().numpy().savetxt("output_pos.txt",output_pos)
       # __import__('pdb').set_trace()

        #print(feature_neg)
        #print(adj_neg)
        #output_neg = model2(features_neg, adj)
        output_neg = model2(features_neg, adj)
        #__import__('pdb').set_trace()
        #var.detach().numpy().savetxt("output_neg.txt",output_neg)
        # print(output_pos)
        # print(output_neg)
        #output = model(features,adj_neg)
        loss_train = F.nll_loss(output_pos[idx_train], labels[idx_train])
        loss_neg_train = F.nll_loss(output_neg[idx_train], labels[idx_train])
        #acc_train = accuracy(output_pos[idx_train], labels[idx_train])
        loss_train.backward()
        loss_neg_train.backward()
        #print(loss_train)
        optimizer.step()
        optimizer1.step()


        #if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            #model.eval()
            #output = model(features, adj)

        #对节点符号进行符号预测
        #如ou何对节点符号进行embedding：

        eval_dict = task.link_sign_pre(utils.cat_neighbor_new(
            train_data.G.g, output_pos, method='cat_neg'),utils.cat_neighbor_new(
            train_data.G.g, output_neg, method='cat_neg'),idx_train,idx_val,idx_test,method='concatenate')
        #print(np.all(model.get_embedding()))
        # task.link_sign_prediction_ktuple(model.get_embedding())
        #print(eval_dict)
        if config.snapshoot:
            #print('epoch {0}, loss: {1}, time: {2}'.format(epoch, total_loss, train_time), file=fout)
            print("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
                eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']))
            for key in best_eval_dict:
                if eval_dict[key] > best_eval_dict[key]:
                    for key in best_eval_dict:
                        best_eval_dict[key] = eval_dict[key]

                    #model.save(snap_root + '/{}.model'.format(config.model))
                    #model.save('/{}.model'.format(config.model))
                    break




        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        #acc_val = accuracy(output[idx_val], labels[idx_val])
        #print('Epoch: {:04d}'.format(epoch+1),
        #      'loss_train: {:.4f}'.format(loss_train.item()),
        #      'acc_train: {:.4f}'.format(acc_train.item()),
        #      'loss_val: {:.4f}'.format(loss_val.item()),
        #      'acc_val: {:.4f}'.format(acc_val.item()),
        #      'time: {:.4f}s'.format(time.time() - t))

    #pygcn
    #'''
    '''
    #dgl
    dur = []
    for epoch in range(30):
        if epoch >=3:
            t0 = time.time()
        logits_pos = net(g_pos, features)
        logits_neg = net(g_neg, features)
        #print(logits_pos)
        #logits_neg = net(g_neg, features)
        logp = F.log_softmax(logits_pos, 1)
        logn = F.log_softmax(logits_neg, 1)
        print('logp:{}'.format(logp))
        print('logn:{}'.format(logn))
        #logp = F.log_softmax(logits, 1)
        #np.savetxt("logp.txt",logits_pos.get_embedding())

        loss_train = F.nll_loss(logits_pos[idx_train], labels[idx_train])
        acc_train = accuracy(logits_pos[idx_train], labels[idx_train])
        loss = F.nll_loss(logits_pos[idx_val], labels[idx_val])
        optimizer.zero_grad()
        #print(logits)
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        eval_dict = task.link_sign_prediction_split(utils.cat_neighbor(
            train_data.G.g, logp, method='cat_pos'), method='concatenate')
        #print(np.all(model.get_embedding()))
        # task.link_sign_prediction_ktuple(model.get_embedding())
        #print(eval_dict)
        if config.snapshoot:
            #print('epoch {0}, loss: {1}, time: {2}'.format(epoch, total_loss, train_time), file=fout)
            #print("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
                #eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']), file=fout)
            for key in best_eval_dict:
                if eval_dict[key] > best_eval_dict[key]:
                    for key in best_eval_dict:
                        best_eval_dict[key] = eval_dict[key]

                    #model.save(snap_root + '/{}.model'.format(config.model))
                    output.save('/{}.model'.format(config.model))
                    break
        loss_val = F.nll_loss(logits_pos[idx_val], labels[idx_val])
        acc_val = accuracy(logits_pos[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}| acc {:.4f} ".format(
                epoch,
                loss.item(),
                np.mean(dur)))


    '''



    #train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    #print(train_dataloader)
    #pytorch traindataloader
    #创建DataLoader，batch_size设置为2，shuffle=False不打乱数据顺序，num_workers= 4使用4个子进程：
    #train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    #保存train_data文件
    '''
    if os.path.exists(config.filename + '_' + str(config.split_ratio)
                      + '_{}.pkl'.format(dataset_name)) and not config.save_dataset:
        #将数据编译成pkl格式
        train_data = pickle.load(open(config.filename + '_' + str(config.split_ratio) + '_{}.pkl'.format(dataset_name), 'rb'))
        print(config.filename + '_' + str(config.split_ratio))
        print(train_data) #<data.kTupleDataV1.kTupleDataV1 object at 0x7f96e2ddc5c0>
        print('exists {}.pkl, load it!'.format(dataset_name))
        print(train_data.G.g.number_of_nodes(), train_data.G.g.number_of_edges())
    else:
        train_data = kTupleDataV1(config.filename, split_ratio=config.split_ratio, neg_num=config.neg_num)
        #list数据写入
        with open('train_data_ktuple','w') as f:
            f.write(str(train_data.sign_tuple))
        pickle.dump(train_data, open(
            config.filename + '_' + str(config.split_ratio) + '_{}.pkl'.format(dataset_name), 'wb'))
        print('success save {}.pkl'.format(dataset_name))
    '''
    '''
    #3333
    config.N = train_data.G.g.number_of_nodes()
    #print(config.N)
    model = getattr(models, config.model)(config)   # .eval()
    #print(model)

    if torch.cuda.is_available():
        model.cuda()
        config.CUDA = True
    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    #print(train_dataloader)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95)
    #print(optimizer)
    task = Task(train_data.G, config)

    best_eval_dict = {'f1-micro': 0.0, 'f1-macro': 0.0}
    
    '''

    '''
    #if config.snapshoot:
        #snapshoot_time = datetime.strftime(datetime.now(),  '%y-%m-%d_%H:%M:%S')
        #best_eval_dict = {'f1-micro': 0.0, 'f1-macro': 0.0}
        #fout, snap_root = utils.init_snapshoot(config.filename, snapshoot_time)
        #config.save(fout)
    '''
    '''
    #config.show()

    # model.train()
    #print(range(config.epochs))
    for epoch in range(config.epochs):
        total_loss = 0.0
        start_time = datetime.now()
        for idx, data in enumerate(train_dataloader):
            negs = data[-1]
            data = data[:-1]
            neg_r, neg_t = zip(*negs)
            #transpose只能操作2D矩阵的转置
            neg_r = torch.cat(neg_r).view(len(negs), -1).transpose(0, 1)
            neg_t = torch.cat(neg_t).view(len(negs), -1).transpose(0, 1)
            data.extend([neg_r, neg_t])
            data = map(lambda x: Variable(x), data)
            if config.CUDA:
                data = map(lambda x: Variable(x.cuda()), data)
            else:
                data = map(lambda x: Variable(x), data)
            #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            optimizer.zero_grad()
            loss = model(data)

            loss.backward()
            #optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，可以根据具体的需求来做。
            #只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整
            #更新模型
            optimizer.step()
            #计算损失函数-》2
            if config.CUDA:
                total_loss += loss.cpu().data.numpy()
            else:
                total_loss += loss.data.numpy()
        train_time = (datetime.now() - start_time).seconds
        print('epoch {0}, loss: {1}, time: {2}'.format(epoch, total_loss, train_time))
        if (epoch > 30 or config.speedup):
            if config.speedup:
                if epoch % config.speedup != 0:
                    continue
            if epoch % 5 != 0:
                continue
        eval_dict = task.link_sign_prediction_split(utils.cat_neighbor(
            train_data.G.g, model.get_embedding(), method='cat_neg'), method='concatenate')
        print(np.all(model.get_embedding()))
        # task.link_sign_prediction_ktuple(model.get_embedding())
        print(eval_dict)
        if config.snapshoot:
            #print('epoch {0}, loss: {1}, time: {2}'.format(epoch, total_loss, train_time), file=fout)
            #print("link_sign_prediction auc: {:.3f}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(
                #eval_dict['auc'], eval_dict['f1'], eval_dict['f1-micro'], eval_dict['f1-macro']), file=fout)
            for key in best_eval_dict:
                if eval_dict[key] > best_eval_dict[key]:
                    for key in best_eval_dict:
                        best_eval_dict[key] = eval_dict[key]

                    #model.save(snap_root + '/{}.model'.format(config.model))
                    model.save('/{}.model'.format(config.model))
                    break
    #333333
    '''
    '''
    if config.snapshoot:
        fout.write('best result:' + str(best_eval_dict) + '\n')
        fout.close()
        config.save(open(snap_root + '/{:.3f}.config'.format(best_eval_dict['f1-micro']), 'w'))
        pickle.dump(config, open(snap_root + '/config.pkl', 'wb'))
        if config.save_dataset:
            pickle.dump(train_data, open(snap_root + '/data.pkl', 'wb'))
    '''
    '''
    f = open("out.txt", "w")    # 打开文件以便写入
    print(model.get_embedding())
    np.savetxt("out.txt",np.array(model.get_embedding()))

    #print(np.save(np.array(model.get_embedding())),file=f)
    f.close  #  关闭文件
    '''

'''
def val(model, dataloader):

'''

    #计算模型在验证集上的准确率等信息，用以辅助训练

'''

pass
'''
# 在邻接矩阵中加入自连接
def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # 对加入自连接的邻接矩阵进行对称归一化处理
    adj = normalize_adj(adj, symmetric)
    return adj





def test(**kwargs):
    snap_root = kwargs['snap_root']
    config_file = snap_root + '/config.pkl'
    config = pickle.load(file(config_file))
    model_file = snap_root + '/{}.model'.format(config.model)
    dataset_name = 'kTupleDataV1'
    if os.path.exists(config.filename + '_' + str(config.split_ratio)
                      + '_{}.pkl'.format(dataset_name)):
        train_data = pickle.load(file(snap_root + '/data.pkl'))
        print ('exists {}.pkl, load it!'.format(dataset_name))
        print (train_data.G.g.number_of_nodes(), train_data.G.g.number_of_edges())
    else:
        raise Exception('Data Module not exists!')
    model = getattr(models, config.model)(config)   # .eval()
    if torch.cuda.is_available():
        model.cuda()
        config.CUDA = True
    model.load_state_dict(torch.load(model_file))
    task = Task(train_data.G, config)
    task.link_sign_prediction_split(utils.cat_neighbor(
            train_data.G.g, model.get_embedding(), method='null'), method='concatenate')


def read_graph(**kwargs):
    def number_of_sign_edges(g, sign):
        count = 0
        for edge in g.edges:
            if g[edge[0]][edge[1]]['sign'] == sign:
                count += 1
        return 1.0 * count / g.number_of_edges()
    file_name = kwargs['file_name']
    from utils.SignGraph import SignGraph
    G = SignGraph(file_name, split_ratio=1.0, directed=True)
    print ("nodes: {}\nedges: {}\npos_ratio: {:.3f}\nneg_ratio: {:.3f}\n".
           format(G.g.number_of_nodes(), G.g.number_of_edges(),
                  number_of_sign_edges(G.g, 1), number_of_sign_edges(G.g, -1)))


def help():
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example:
            python {0} train --env='env0701' --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(config.__class__))
    print(source)

#normalize
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def print_Tensor_encoded(self, epoch, i, tensors):
    message = '(epoch: %d, iters: %d)' % (epoch, i)
    for k, v in tensors.items():
        with open(self.log_tensor_name, "a") as log_file:
            v_cpu = v.cpu()
            log_file.write('%s: ' % message)
            np.savetxt(log_file,  v_cpu.detach().numpy())
#准确率判断
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    print(preds)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
#标签转化成one-hot label形式，在符号网络中暂时使用不到这个标签，
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



if __name__=='__main__':

    #sysstr = platform.system()
    #print(sysstr)
    #在本地运行程序
    train()

    #import fire
    #fire.Fire()

