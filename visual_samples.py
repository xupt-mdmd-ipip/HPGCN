import numpy as np

from Dataset import *
from torch.utils.data import DataLoader
from function import *
from model import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch.optim as optim

device = torch.device("cuda:0")
# 加载已标记数据，获取预测标签及特征
labeled_data = np.load('./data/ip_train_data.npy')
labeled_label = np.load('./data/ip_train_label.npy')
train_set = data_set(labeled_data, labeled_label)
train_loader = DataLoader(train_set)
model_nn = torch.load('model_train_1k.pth')
labeled_feats, labeled_pred = get_features(model_nn, train_loader)
# # 加载未标记数据，获取预测标签及特征
# cdc_data = np.load('./data/ip_cdc_data.npy')
# cdc_label = np.load('./data/ip_cdc_label.npy')
# cdc_set = data_set(cdc_data, cdc_label)
# cdc_loader = DataLoader(cdc_set)
# cdc_feats, cdc_pred = get_features(model_nn, cdc_loader)

# 找到未标记数据中每一类预测标签的中心样本序号
labeled_feats = labeled_feats.detach().cpu().numpy()
labeled_pred = labeled_pred.detach().cpu().numpy()
labeled_pred = labeled_pred.astype(int)
labeled_max = []
labeled_center = []
labeled_range = np.max(labeled_pred).astype(int) + 1
for x in range(labeled_range):
    class_index = []
    for y in range(len(labeled_pred)):
        if labeled_pred[y] == x:
            class_index.append(y)
    dist = pairwise_distances(labeled_feats[class_index], labeled_feats[class_index], metric='euclidean')
    dist_delete0 = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
    dist_mean = np.mean(dist_delete0, 1)
    min_point = np.argmin(dist_mean)
    labeled_center.append(class_index[min_point])
    labeled_max.append(np.max(dist[min_point]))

# 找出每一小组
center_dist = pairwise_distances(labeled_feats[labeled_center], labeled_feats[labeled_center], metric='euclidean')
center_delete0 = center_dist[~np.eye(center_dist.shape[0], dtype=bool)].reshape(center_dist.shape[0], -1)
labeled_adjmin = np.zeros(labeled_range)
group_include_one = []
for i in range(len(center_delete0)):
    labeled_adjmin[i] = np.argmin(center_delete0[i])
    labeled_mean1 = np.min(center_delete0[i])
    if labeled_mean1 < labeled_max[i]:
        if labeled_adjmin[i] > i - 1:
            labeled_adjmin[i] = labeled_adjmin[i] + 1
    else:
        labeled_adjmin[i] = i
        group_include_one.append(i)
c_pair = np.zeros((labeled_range, 2))
for ix in range(len(c_pair)):
    c_pair[ix][0] = ix
    c_pair[ix][1] = labeled_adjmin[ix]
    c_pair[ix] = np.sort(c_pair[ix])
    if set(c_pair[ix]) & set(group_include_one):
        c_pair[ix][0] = ix
        c_pair[ix][1] = ix
        group_include_one.append(ix)
group_include_one = np.unique(group_include_one)
c_pair1 = np.delete(c_pair, group_include_one, axis=0)
c_pair1 = np.unique(c_pair1, axis=0)
c = []
count = len(c_pair1)
for iix in range(count):
    c.append(list(c_pair1[iix]))
    for iiy in range(len(c_pair1)):
        if list(set(c[iix]) & set(c_pair1[iiy])):
            c[iix] = list(set(c[iix]) | set(c_pair1[iiy]))
    if list(set(c[iix - 1]) & set(c[iix])):
        c[iix - 1] = list(set(c[iix - 1]) | set(c[iix]))
        c[iix] = c[iix - 1]
group_more = np.unique(c)

# 求组内均值点
listnum = len(group_more)
center_mean = np.zeros((listnum, 2048))
for listindex in range(listnum):
    list_now = group_more[listindex]
    list_now = np.array(list_now).astype(int)
    center_featnow = np.zeros((len(list_now), 2048))
    for ingroup in range(len(list_now)):
        center_featnow[ingroup] = labeled_feats[labeled_center[list_now[ingroup]]]

# 画图
pca = PCA(n_components=2)
labeled_feats = pca.fit_transform(labeled_feats)
center_mean = pca.fit_transform(center_mean)
colors = ['blue', 'yellow', 'violet', 'red', 'green', 'pink', 'orange', 'magenta', 'gray', 'deeppink', 'coral',
          'blueviolet', 'brown', 'aqua', 'cornsilk', 'chartreuse']
# markers = ['.', ',', 'o', 'v',  '^', '<', '>', 's', 'h', 'p', 'x', '+', 'D', 'd', '|', '_']
plt.rcParams["figure.dpi"] = 300
for ic in range(len(labeled_pred)):
    plt.scatter(labeled_feats[ic][0], labeled_feats[ic][1], c=colors[labeled_pred[ic]], alpha=0.3)
for iic in range(len(labeled_center)):
    plt.scatter(labeled_feats[labeled_center[iic]][0], labeled_feats[labeled_center[iic]][1], c=colors[iic], marker='*',
                alpha=0.8)
for iiic in range(len(center_mean)):
    plt.scatter(center_mean[iiic][0], center_mean[iiic][1], marker='x', alpha=0.8)
plt.savefig('fig/l_center3.png')
plt.show()
print('~')
