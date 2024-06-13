import scipy.io as scio
import random
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from config import *
from function import *


# 数据导入
def load_data(data_path, label_path):
    data_pre = scio.loadmat(data_path)
    label_pre = scio.loadmat(label_path)
    return data_pre, label_pre


# 数据变形
def data_rs(data_pre):
    data_rs = np.reshape(data_pre, (data_pre.shape[0] * data_pre.shape[1], data_pre.shape[2]))
    return data_rs


# 处理标签
def label_process(label_pre):
    label_rs = np.reshape(label_pre, -1)
    return label_rs


# 去除标签为0的数据
def del_0(data_rs, label_rs):
    del_list = np.array([]).astype(int)
    for i in range(len(data_rs)):
        if label_rs[i] == 0:
            del_list = np.append(del_list, i)
    data_del = np.delete(data_rs, del_list, axis=0)
    label_del = np.delete(label_rs, del_list, axis=0)
    return data_del, label_del


# 获取n个样本
def get_sample(data_prep, label_prep, n):
    index = [x for x in range(len(data_prep))]
    random.shuffle(index)
    data_sf = data_prep[index]
    label_sf = label_prep[index]
    del_list = np.array([]).astype(int)
    for j in range(17):
        count = 0
        for i in range(len(data_sf)):
            if (label_sf[i] == j) & (count < n):
                del_list = np.append(del_list, i)
                count = count + 1
    data_res = np.delete(data_sf, del_list, axis=0)
    label_res = np.delete(label_sf, del_list, axis=0)
    data_train = data_sf[del_list]
    label_train = label_sf[del_list]
    return data_train, label_train, data_res, label_res


# 乱序
def data_rd(data_pre, label_pre):
    index = [x for x in range(len(data_pre))]
    random.shuffle(index)
    data_rd = data_pre[index]
    label_rd = label_pre[index]
    return data_rd, label_rd


# 去除0标签后，将所有标签-1
def class_1(label_process):
    for i in range(len(label_process)):
        label_process[i] = label_process[i] - 1
    label_process = label_process
    return label_process


data_pre, label_pre = load_data(IP_data_path, IP_label_path)
data_pre = data_pre['indian_pines_corrected']
label_pre = label_pre['indian_pines_gt']
data_pre = data_pre.reshape(-1, data_pre.shape[2])
pca = PCA(n_components=5)
data_new = pca.fit_transform(data_pre).reshape((145, 145, 5))
data_rs = data_rs(data_pre)
label_rs = label_process(label_pre)
data_del0, label_del0 = del_0(data_rs, label_rs)
data_train, label_train, data_res, label_res = get_sample(data_del0, label_del0, 10)
data_train, label_train = data_rd(data_train, label_train)
data_train1, data_res, label_train1, label_res = train_test_split(data_res, label_res, train_size=0.004,
                                                                  random_state=666)
data_train = np.concatenate((data_train, data_train1), 0)
label_train = np.concatenate((label_train, label_train1), 0)
data_cdd, data_test, label_cdd, label_test = train_test_split(data_res, label_res, train_size=0.5, random_state=666)
label_train = class_1(label_train)
label_cdd = class_1(label_cdd)
label_test = class_1(label_test)
ds_name = 'ip'
save_file(data_train, ds_name, sn_train, sn_data)
save_file(label_train, ds_name, sn_train, sn_label)
save_file(data_cdd, ds_name, sn_cdc, sn_data)
save_file(label_cdd, ds_name, sn_cdc, sn_label)
save_file(data_test, ds_name, sn_test, sn_data)
save_file(label_test, ds_name, sn_test, sn_label)
