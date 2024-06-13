from train_test import *
from Dataset import *
from config import *

device = torch.device("cuda:0")


# model pretraining
def model_pretrain(data_path_train, label_path_train):
    data_train = np.load(data_path_train)
    label_train = np.load(label_path_train)
    train_ds = data_set(data_train, label_train)
    train_nn(train_ds, epoch)


model_pretrain('data/ip_train_data.npy', 'data/ip_train_label.npy')

# labeled_data = np.load('./data/ip_train_data.npy')
# labeled_label = np.load('./data/ip_train_label.npy')
# train_set = data_set(labeled_data, labeled_label)
# train_loader = DataLoader(train_set)
# model_nn = torch.load('model_train_1k.pth')
# features = get_features(model_nn, train_loader)

# cdc_data = np.load('./data/ip_cdc_data.npy')
# cdc_label = np.load('./data/ip_cdc_label.npy')
# cdc_set = data_set(cdc_data, cdc_label)
# cdc_loader = DataLoader(cdc_set)
# cdc_feats = get_features(model_nn, cdc_loader)
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
# # score_1 = torch.zeros((len(choose_ds.data), 1))
# # for i in range(len(choose_ds.data)):
# #     score_1[i] = torch.max(scores[i])
# # scores_median = np.squeeze(torch.abs(score_1[:len(cdc_data)] - 0).detach().cpu().numpy())
# # arg = np.argsort(scores_median)
# np.save('arg', arg)
# arg = np.load('arg.npy')
# qurey_data = cdc_data[arg]
# dist = cdist(labeled_data, qurey_data, metric='euclidean')
# index = []
# mean = np.zeros(dist.shape[0])
# for i in range(len(dist)):
#     mean[i] = np.mean(dist[i])
# for x in range(len(mean)):
#     for y in range(dist.shape[1]):
#         if dist[x][y] > mean[x]:
#             index = np.append(index, y)
# sort = dict(Counter(index))
# qdata_set = np.concatenate((labeled_data, qurey_data), axis=0)
# qlabel_set = np.concatenate((labeled_label, qurey_label), axis=0)
# model = model_pretrain(qdata_set, qlabel_set)
# model_pretest('./data/ip_test_data.npy', './data/ip_test_label.npy')
# print('~')
