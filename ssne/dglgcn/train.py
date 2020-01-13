# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from dglgcn.utils import load_cora_data
from dglgcn.models import Net,GCN,net
warnings.filterwarnings('ignore')

#模型处理
g, features, labels, mask = load_cora_data()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(30):
    if epoch >=3:
        t0 = time.time()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch >=3:
        dur.append(time.time() - t0)
    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))