# model.py for gae
import torch
import torch.nn as nn
import torch.nn.functional as F
from gae.layers import GraphConvolution

class GCN_AE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCN_AE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj):
        return self.dc(self.encode(x, adj))

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        # self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        # adj = self.act(torch.mm(z, z.t()))
        adj = torch.mm(z, z.t())
        return adj