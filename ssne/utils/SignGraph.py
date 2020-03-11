# -*- coding: utf-8 -*-
# coding: utf-8
import networkx as nx
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import dok_matrix
import pandas as pd


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

