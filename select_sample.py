import numpy as np
import torch
from Dataset import *
from torch.utils.data import DataLoader
from function import *
from model import *
from sklearn.metrics.pairwise import pairwise_distances

device = torch.device("cuda:0")


# 寻找原型
def prototypeFinding(feats, labels):
    return


# 加载已标记数据，获取预测标签及特征
labeledData = np.load('./data/ip_train_data.npy')
labeledLabel = np.load('./data/ip_train_label.npy')
trainSet = data_set(labeledData, labeledLabel)
train_loader = DataLoader(trainSet)
model_nn = torch.load('model/model_train_1k.pth')
labeledFeats = get_features(model_nn, train_loader)

print('~')

# data_new = torch.cat((cdc_feats, features), 0)
# label_0 = torch.zeros(len(cdc_feats), dtype=int)
# label_1 = torch.ones(len(features), dtype=int)
# label_new = torch.cat((label_0, label_1), 0)
# choose_ds = data_set2(data_new, label_new)
# choose_dl = DataLoader(choose_ds)
# # GCN挑选样本
# adj = aff_to_adj(choose_ds.data)
# model = GCN(2048, 2048, 0.5).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_loss = []
# # 训练
# print('training begin')
# for epoch in range(200):
#     print(epoch+1)
#     input = choose_ds.data.to(device)
#     label = choose_ds.label.to(device)
#     optimizer.zero_grad()
#     outputs = model(input, adj)
#     loss = criterion(outputs,label)
#     loss.backward()
#     optimizer.step()
# print('training finished')
# # 获取关系特征
# model.eval()
# inputs = choose_ds.data.cuda()
# labels = choose_ds.label.cuda()
# g_feat = model(inputs, adj)
# g_mean = torch.mean(g_feat[:5024], dim=0)
# new_feat = torch.zeros(cdc_feats.shape)
# for i in range(len(cdc_feats)):
#     new_feat[i] = 0.5 * cdc_feats[i] + 0.5 * g_mean
# new = torch.max(new_feat, dim=1).indices
# cdc = torch.max(cdc_feats, dim=1).indices
# arg = []
# for x in range(len(cdc_feats)):
#     if new[x] != cdc[x]:
#         arg.append(x)
# score_1 = torch.zeros((len(choose_ds.data), 1))
# for i in range(len(choose_ds.data)):
#     score_1[i] = torch.max(scores[i])
# scores_median = np.squeeze(torch.abs(score_1[:len(cdc_data)] - 0).detach().cpu().numpy())
# arg = np.argsort(scores_median)
