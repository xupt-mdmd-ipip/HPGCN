import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

device = torch.device("cuda:0")

cdd_data = np.load("data/ip_cdd_data.npy") / 1.0
model_f = torch.load("model/model_train_1k.pth").double().to(device).eval()
cdd_data = torch.tensor(cdd_data).to(device)
# print(model_f)

# 提取特征
_, cdd_data_after = model_f(cdd_data)
# print(cdd_data_after.shape)  # (5024,2048)
cdd_data_after = cdd_data_after.detach().cpu()
cdd_data_after = np.array(cdd_data_after)
# print(cdd_data_after.shape)
# K-means将5024分为16类
K_means = KMeans(n_clusters=16)
choose = []

for run in range(5):
    result = K_means.fit(cdd_data_after)
    for i in range(result.n_clusters):  # result.n_clusters
        # print(cdd_data_after[np.where(result.labels_ == i)].shape)
        classData = cdd_data_after[np.where(result.labels_ == i)]

        # print(result.cluster_centers_[i, :].shape)
        distance = euclidean_distances(cdd_data_after[np.where(result.labels_ == i)],
                                       result.cluster_centers_[i, :].reshape(1, -1))
        classIndex = np.where(result.labels_ == i)
        distance = distance.reshape(-1)
        maxIndex = np.argmax(distance)  # 簇中的id
        choose.append(classIndex[0][maxIndex])
        minIndex = np.argmin(distance)
        choose.append(classIndex[0][minIndex])

    # cdd_data_after = np.delete(cdd_data_after, np.where(cdd_data_after == choose))

print(len(choose))
print(choose)

cdd_data = np.array(cdd_data.cpu())
print(cdd_data.shape)

values = np.zeros((len(choose), 200))
for i, j in enumerate(choose):
    print(i, j)
    values[i] = cdd_data[j, :]
print(values.shape)

# print(cdd_data[choose[0]])
# remove_cdd = np.where(cdd_data[choose])
# for j in range(len(cdd_data_after)):
#     if classData[maxIndex]

# distanceMax = distance[0]
#     distanceMin = distance[0]
#     for j in range(len(distance) - 1):
#         if distance[j] < distance[j + 1]:
#             distanceMax = distance[j + 1]
#             idMax = j + 1
#     print(distanceMax, idMax)
#     print("....")
#     for x in range(len(distance) - 1):
#         if distance[x] > distance[x + 1]:
#             distanceMin = distance[x + 1]
#             idMin = x + 1
#     print(distanceMin, idMin)
#     print("MMMMMMMMMMMMMMMMMM")
# print(distance.shape)

# print(result.labels_[0])  # 聚类结果的所有点
# for i in range(result.n_clusters):  # 16类
#     point = np.where(result.labels_ == i)  # 找出属于这一类的所有点
#     pointClass = {}
#     pointClass = point[0]
#     # print(pointClass[0],pointClass[1])
#     ListMin = 1e8
#     ListMax = -1e8
#     min = 0
#     max = 0
#     for index in range(len(pointClass)):
#         # print(index)
#         a = cdd_data_after[pointClass[index]]
#         # print("a", a.shape)
#         # print("result", result.cluster_centers_[i, :].shape)
#         list = np.linalg.norm([a.any()], result.cluster_centers_[i, :].any())  # 类中每个点到中心的距离
#         if list > ListMax:
#             max = pointClass[index]
#             x = a
#             ListMax = list
#         elif list < ListMin:
#             min = pointClass[index]
#             ListMin = list
#             y = a
#     print(min, max)
#     print("******")
#     print(pointClass[min], pointClass[max])
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     K_id = {i: np.where(result.labels_ == i)[0] for i in range(result.n_clusters)}  # 得到每一类的点的id
#     print(K_id)
# print("k_means success")
