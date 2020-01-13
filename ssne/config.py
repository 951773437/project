# -*- coding: utf-8 -*-

import warnings

'''Python3.5中：iteritems变为items'''
def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)


class BasicConfig():
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def show(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

    def save(self, file_out=None):
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                if file_out:
                    print((k, getattr(self, k)), file=file_out)



class kTupleConfig(BasicConfig):
    filename = 'data/dataset/wiki-Vote.txt'
    directed = True
    weighted = False
    dimension = 128
    margin = 1
    pos_margin = 1
    neg_margin = 1
    zero_margin = 1
    weight_decay = 1e-2
    batch_size = 32
    lr = 0.01
    epochs = 60
    model = 'kTupleV3'
    CUDA = False
    num_workers = 4
    split_ratio = 1.0  #分流比; 拆分比例; 分路比; 分光比; 分割率; 
    snapshoot = True
    neg_num = 2
    save_dataset = False
    speedup = False


kTupleconfig = kTupleConfig()
