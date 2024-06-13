import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import nn


class model_nn(nn.Module):
    def __init__(self):
        super(model_nn, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(200, 2048), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(2048, 17))

    def forward(self, x_nn):
        x_1 = self.layer1(x_nn)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_4, x_3


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2048)
        # self.linear = nn.Linear(2048, 2048)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        # x = self.linear(x)
        return x
