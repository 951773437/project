from torch.utils import data
from src.check import SignGraph
from math import pow
import random
import json
import networkx as nx
import numpy as np

#数据预处理
#目标返回正向节点和负向节点的邻接矩阵
class TupleData(data.Dataset):
    """Return the triplet: [h, r, t, sign, [negs]]
    """

    def __init__(self, filename, seq='\t', split_ratio=0.8, remove_self_loop=True, neg_num=2):
        self.G = SignGraph(filename, seq=seq, split_ratio=split_ratio,
                           remove_self_loop=remove_self_loop, directed=True, train_with_only_trainset=True)
        self.neg_num = neg_num
        self.init_neg_tabel()
        #self.sign_tuple = self.get_SignTuple()
        self.ktuple = self.get_Ktuple()
        #self.newkTuple = self.change_kTuple()
        #self.posTuple = self.get_posTuple()
        #self.negTuple = self.get_negTuple()
        #self.new_neg_pos_sample()
        #self.new_neg_neg_sample()
        self.poslist = self.G.poslist
        self.neglist = self.G.neglist
        self.pos_feature = self.G.pos_adjmatrix
        self.neg_feature = self.G.neg_adjmatrix
        #self.pos_feature = self.get_posFeature()
        #self.neg_feature = self.get_negFeature()
        self.feature = [self.pos_feature,self.neg_feature]
        #print('poslist:{}'.format(self.poslist))
        self.adj_matrix = self.G.adj_matrix
        print(self.adj_matrix.todense())
        #print('adj_matrix:{}'.format(self.adj_matrix.todense()))
        #self.new_neg_sample()
        #self.neg_sample()
        #self.pos_sample()
        self.adj_pos = self.get_pos_adj()
        self.adj_neg = self.get_neg_adj()
        self.nega_sample()
        #self.neg_feature = self.
        #print(self.posTuple)
        #print(self.negTuple)
        
        del self.neg_sample_table
        
        
    
    #获取特征正子集特征，
    def get_posFeature(self):
        pos_feature = []
        for i in self.G.g.nodes():
            num = 0;
            for edge in self.poslist:
                #__import__('pdb').set_trace()
                if i == edge[0]:
                    #for edge2 in self.poslist:
                        #pos
                        #if j == edge[1]:
                            
                    #num = num + 1
                    num = 1
                    pos_feature.append(num)
                    break
            if num == 0:
                pos_feature.append(0)
        #for i in self.G.g.nodes():
            #if self.G.g.has_edge()
        #print(pos_feature)
        return pos_feature
    # 获取负子集特征：
    def get_negFeature(self):
        #基于status theory 如果节点有负向边，为0，-1
        #for edge in self.ktuple:
            #print(edge)
            
        
        
        
        neg_feature = []
        for i in self.G.g.nodes():
            num = 0;
            for edge in self.neglist:
                if i == edge[0]:
                    num = num - 1
                    #num = 1
                    neg_feature.append(num)
                    
                    break
            if num == 0:
                neg_feature.append(0)
        #for i in self.G.g.nodes():
            #if self.G.g.has_edge()
        #print(pos_feature)
        return neg_feature
    
    
    def get_posTuple(self):
        pos_kTuple = []
        for i in self.sign_tuple.keys():
            if self.sign_tuple[i][0]:
                for edge in self.sign_tuple[i][0]:
                     pos_kTuple.append([edge[0], 1, edge[1], 1])
            #if self.sign_tuple[i][1]:
                #for edge in self.sign_tuple[i][1]:
                     #new_kTuple.append([edge[0], 0, edge[1], -1])
        #print('pos_kTuple:{}'.format(pos_kTuple))         
        return pos_kTuple
    
    def get_negTuple(self):
        neg_kTuple = []
        for i in self.sign_tuple.keys():
            #if self.sign_tuple[i][0]:
                #for edge in self.sign_tuple[i][0]:
                     #neg_kTuple.append([edge[0], 1, edge[1], 1])
            if self.sign_tuple[i][1]:
                for edge in self.sign_tuple[i][1]:
                     neg_kTuple.append([edge[0], 0, edge[1], -1])
        print('neg_kTuple:{}'.format(neg_kTuple))         
        return neg_kTuple
    
    def get_poslist(self):
        pos_edges = []
        for edge in self.G.g.edges():
            sou_num = edge[0] #源节点
            if self.G.g[edge[0]][edge[1]]['sign'] == 1:
                pos_edges.append(edge)
        return pos_edges

    #生成正向集合带采样的矩阵
    def get_pos_adj(self):
        pos_G = nx.Graph()
        #print(self.G.g.nodes())
        pos_G.add_nodes_from(self.G.g.nodes())
        #print(self.G.poslist)
        pos_G.add_edges_from(self.G.poslist)
        adj_pos = nx.adj_matrix(pos_G)
        #print(adj_pos.todense())
        return(adj_pos)
    #生成负向集合带采样的矩阵
    def get_neg_adj(self):
        pos_G = nx.Graph()
        pos_G.add_nodes_from(self.G.g.nodes())
        pos_G.add_edges_from(self.G.poslist)
        adj_pos = nx.adj_matrix(pos_G)
        #print(adj_pos.todense())
        return(adj_pos)
    
    #改用字典进行代码处理，可能需要加快代码运行速度。
    def get_SignTuple(self):
        #使用字典存储数据，每一个节点会成为
        sign_dict = {}
        triplet = []
        #一个对象能不能作为字典的key，就取决于其有没有__hash__方法。
        #所以所有python自带类型中，除了list、dict、set和内部至少带有上述三种类型之一的tuple之外，其余的对象都能当key。
        #在python的函数中和全局同名的变量，如果你有修改变量的值就会变成局部变量，在修改之前对该变量的引用自然就会出现没定义这样的错误了，
        #如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
        for edge in self.G.g.edges():
            sou_num = edge[0] #源节点
            #print(sou_num)
            pos_edges = []
            neg_edges = []
            if self.G.g[edge[0]][edge[1]]['sign'] == 1:
                new_pos_edge = [edge]
                pos_edges.append(new_pos_edge)
                pos_tuple = []
                if sou_num in sign_dict.keys():
                    sign_list = sign_dict[sou_num]
                    new_pos_edges = sign_list[0] #postive数据
                    new_neg_edges = sign_list[1] #negative数据
                    new_pos_edges.append(edge)
                    sign_list = [new_pos_edges,new_neg_edges]
                    sign_dict[sou_num] = sign_list
                else:
                    #不在键值中
                    sign_dict.setdefault(sou_num,[]).append(new_pos_edge)
                    sign_dict.setdefault(sou_num,[]).append(neg_edges)
                #判断triplet内是否含有这个元素
                #if edge[0] in triplet
                #triplet.append([edge[0],pos_edges,neg_edges])
            else:
                new_neg_edge = [edge]
                neg_edges.append(new_neg_edge)
                #判断triplet是否含有该节点，如果没有该节点，添加，如果有该节点，修改
                if sou_num in sign_dict.keys():
                    sign_list = sign_dict[sou_num]
                    new_pos_edges = sign_list[0] #postive数据
                    new_neg_edges = sign_list[1] #negative数据
                    new_neg_edges.append(edge)
                    sign_list = [new_pos_edges,new_neg_edges]
                    sign_dict[sou_num] = sign_list
                else:
                    #不在键值中
                    sign_dict.setdefault(sou_num,[]).append(pos_edges)
                    sign_dict.setdefault(sou_num,[]).append(new_neg_edge)
                #判断triplet内是否含有这个元素
                #triplet.append([edge[0],pos_edges,neg_edges])       
        #print(sign_dict)
        #jsObj = json.dumps(sign_dict)
        #fileObject = open('jsonFile.json', 'w')
        #fileObject.write(jsObj)
        #fileObject.close()
        
        #输出的数据格式用pickle存储出来，便于之后读取和计算
        #list_file = open('signTuple.pickle','wb')
        #pickle.dump(sign_dict,list_file)
        #list_file.close()
        return sign_dict

    def get_Ktuple(self):
        triplet = []
        for edge in self.G.g.edges():
            #print(edge)
            if self.G.g[edge[0]][edge[1]]['sign'] == 1:
                triplet.append([edge[0], 1, edge[1], 1])
            else:
                triplet.append([edge[0], 0, edge[1], -1])
        return triplet

    def change_kTuple(self):
        new_kTuple = []
        for i in self.sign_tuple.keys():
            if self.sign_tuple[i][0]:
                for edge in self.sign_tuple[i][0]:
                     new_kTuple.append([edge[0], 1, edge[1], 1])
            if self.sign_tuple[i][1]:
                for edge in self.sign_tuple[i][1]:
                     new_kTuple.append([edge[0], 0, edge[1], -1])
        #print('new_kTuple:{}'.format(new_kTuple))         
        return new_kTuple
                 

    #修改neg_sample输出，在于使自身修改的数据预处理能够输出neg_sample 格式的数据
    #negative_sample 负采样
    def sign_neg_sample(self):
        for i in range(len(self.sign_tuple)):#13
            neg_sign_triple_list = []
            for _ in range(self.neg_num):#2
                new_neighbor = [tgt for src, tgt in self.G.g.out_edges(
                    [self.sign_tuple[i][0]]) if self.G.g[src][tgt]['sign'] == (self.sign_tuple[i][3] * -1)]
                #计算set之间的差集
                new_neighbor = list(set(new_neighbor) - set ([neg[1] for neg in neg_triplet_list]))
                neg_r = self.sign_tuple[i][1]
                if len(new_neighbor) > 0:
                    neg_t = random.choice(new_neighbor)
                else:
                    neg_r = 2
                    #random.randint()方法里面的取值区间是前闭后闭区间
                    #用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b。c是步幅。
                    neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                    while self.G.g.has_edge(self.ktuple[i][0], neg_t):
                        neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                # neg_triplet_list.append([self.ktuple[i][0], self.ktuple[i][1], neg_t])
                neg_triplet_list.append([neg_r, neg_t])
            # self.ktuple[i].append([[self.ktuple[i][0], self.ktuple[i][1], t] for t in neg_triplet_list])
            self.ktuple[i].append(neg_triplet_list)
    
    def new_neg_pos_sample(self):
        for i in range(len(self.posTuple)):#16
            neg_pos_triplet_list = []
            for _ in range(self.neg_num):
                neighbor = [tgt for src, tgt in self.G.g.out_edges(
                    [self.posTuple[i][0]]) if self.G.g[src][tgt]['sign'] == (self.posTuple[i][3] * -1)]
                neighbor = list(set(neighbor) - set([neg[1] for neg in neg_pos_triplet_list]))
                neg_r = self.posTuple[i][1]
                #print(neg_r)
                #print(len(neighbor))
                if len(neighbor) > 0:
                    neg_t = random.choice(neighbor)
                else:
                    neg_r = 2
                    #print(self.neg_tabel_size - 1)
                    neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                    while self.G.g.has_edge(self.newkTuple[i][0], neg_t):
                        neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                # neg_triplet_list.append([self.ktuple[i][0], self.ktuple[i][1], neg_t])
                #print('neg_t:{}'.format(neg_t))
                neg_pos_triplet_list.append([neg_r, neg_t])
            # self.ktuple[i].append([[self.ktuple[i][0], self.ktuple[i][1], t] for t in neg_triplet_list])
            self.posTuple[i].append(neg_pos_triplet_list)
            #print(self.posTuple)
            
    def new_neg_neg_sample(self):
         for i in range(len(self.negTuple)):#16
            neg_neg_triplet_list = []
            for _ in range(self.neg_num):
                neighbor = [tgt for src, tgt in self.G.g.out_edges(
                    [self.negTuple[i][0]]) if self.G.g[src][tgt]['sign'] == (self.negTuple[i][3] * -1)]
                neighbor = list(set(neighbor) - set([neg[1] for neg in neg_neg_triplet_list]))
                neg_r = self.posTuple[i][1]
                #print(neg_r)
                #print(len(neighbor))
                if len(neighbor) > 0:
                    neg_t = random.choice(neighbor)
                else:
                    neg_r = 2
                    #print(self.neg_tabel_size - 1)
                    neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                    while self.G.g.has_edge(self.newkTuple[i][0], neg_t):
                        neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                # neg_triplet_list.append([self.ktuple[i][0], self.ktuple[i][1], neg_t])
                #print('neg_t:{}'.format(neg_t))
                neg_neg_triplet_list.append([neg_r, neg_t])
            # self.ktuple[i].append([[self.ktuple[i][0], self.ktuple[i][1], t] for t in neg_triplet_list])
            self.negTuple[i].append(neg_neg_triplet_list)
            #print(self.negTuple)
    
    
    def new_neg_sample(self):
        for i in range(len(self.newkTuple)):#16
            neg_triplet_list = []
            for _ in range(self.neg_num):
                neighbor = [tgt for src, tgt in self.G.g.out_edges(
                    [self.newkTuple[i][0]]) if self.G.g[src][tgt]['sign'] == (self.newkTuple[i][3] * -1)]
                neighbor = list(set(neighbor) - set([neg[1] for neg in neg_triplet_list]))
                neg_r = self.newkTuple[i][1]
                #print(neg_r)
                #print(len(neighbor))
                if len(neighbor) > 0:
                    neg_t = random.choice(neighbor)
                else:
                    neg_r = 2
                    #print(self.neg_tabel_size - 1)
                    neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                    while self.G.g.has_edge(self.newkTuple[i][0], neg_t):
                        neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                # neg_triplet_list.append([self.ktuple[i][0], self.ktuple[i][1], neg_t])
                #print('neg_t:{}'.format(neg_t))
                neg_triplet_list.append([neg_r, neg_t])
            # self.ktuple[i].append([[self.ktuple[i][0], self.ktuple[i][1], t] for t in neg_triplet_list])
            self.newkTuple[i].append(neg_triplet_list)
            #print(self.newkTuple)
    
    #出现的问题，has_edge产生的结果容易与前面重复。
    #对正向边进行采样
    def pos_sample(self):
        for i in range(len(self.G.poslist)):
            sou_node = self.G.poslist[i][0]
            neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
            #print(self.G.g.edges)
            while self.G.g.has_edge(sou_node, neg_t) | sou_node == neg_t:
                neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                #print(sou_node,neg_t)
            self.poslist.append((sou_node, neg_t))
        #print(self.poslist)
    #对负向边进行采样        
    def neg_sample(self):
        for i in range(len(self.G.neglist)):
            sou_node = self.G.poslist[i][0]
            neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
            while self.G.g.has_edge(sou_node, neg_t) | sou_node == neg_t:
                neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                #print(sou_node,neg_t)
            self.poslist.append((sou_node, neg_t))
        #print(self.poslist)        
            
    
    
    
    #实验采样选择的随机选取网络节点中的与其不连接的节点
    #如果网络中与其有正向连接或者负向连接的节点全部排除
    #剩下的节点在进行随机采样。
            
        
    
    def nega_sample(self):
        #print('sign_tuple:{}'.format(self.sign_tuple))
        #print('ktuple:{}'.format(self.ktuple))
        for i in range(len(self.ktuple)):#16
            neg_triplet_list = []
            for _ in range(self.neg_num):
                neighbor = [tgt for src, tgt in self.G.g.out_edges(
                    [self.ktuple[i][0]]) if self.G.g[src][tgt]['sign'] == (self.ktuple[i][3] * -1)]
                neighbor = list(set(neighbor) - set([neg[1] for neg in neg_triplet_list]))
                #print(neighbor)
                neg_r = self.ktuple[i][1]
                #print(neg_r)
                #print(len(neighbor))
                if len(neighbor) > 0:
                    neg_t = random.choice(neighbor)
                else:
                    neg_r = 2
                    #print(self.neg_tabel_size - 1)
                    neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                    while self.G.g.has_edge(self.ktuple[i][0], neg_t):
                        neg_t = self.neg_sample_table[random.randint(0, self.neg_tabel_size - 1)]
                # neg_triplet_list.append([self.ktuple[i][0], self.ktuple[i][1], neg_t])
                #print('neg_t:{}'.format(neg_t))
                neg_triplet_list.append([neg_r, neg_t])
            # self.ktuple[i].append([[self.ktuple[i][0], self.ktuple[i][1], t] for t in neg_triplet_list])
            self.ktuple[i].append(neg_triplet_list)
            #print(neg_triplet_list)
            #print(self.ktuple)

    #构建负采样表，使用分别对正向节点和负向节点进行负采样。
    def init_neg_tabel(self):
        table_size = 1e8
        self.neg_tabel_size = table_size
        NEG_SAMPLE_POWER = 0.75
        degree = self.G.g.degree()
        norm = sum([pow(degree[i], NEG_SAMPLE_POWER) for i in self.G.g.nodes()])
        self.neg_sample_table = ['' for i in range(int(table_size))]
        p = 0
        i = 0
        for node in self.G.g.nodes():
            p += 1.0 * pow(degree[node], NEG_SAMPLE_POWER) / norm
            while i < table_size and (1.0 * i) / table_size < p:
                self.neg_sample_table[i] = node
                i += 1

    def __getitem__(self, index):
        #return self.newkTuple[index]
        return self.ktuple[index]

    def __len__(self):
        #return len(self.newkTuple)
        return len(self.ktuple)