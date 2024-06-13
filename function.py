import numpy as np
import torch


# 数据文件保存
def save_file(data_file, ds_name, divide, dorl):
    np.save('./data/' + ds_name + '_' + divide + '_' + dorl + '.npy', data_file)


# 网络保存
def save_net(model, model_name):
    torch.save(model, 'model/model_' + model_name + '.pth')


# 化为邻接矩阵
def aff_to_adj(x):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj += -1.0 * np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0)  # rowise sum
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()
    return adj


# 提取特征
def get_features(model, data_loader):
    model.eval()
    features = torch.tensor([]).cuda()
    # pred = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda()
            output, features_batch = model(inputs)
            features = torch.cat((features, features_batch), 0)
            # outputs = torch.max(output, dim=1).indices
            # pred = torch.cat((pred, outputs))
        feat = features
        # l_pred = pred
    return feat


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss
