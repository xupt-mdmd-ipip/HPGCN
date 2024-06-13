import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [90.29, 90.71, 92.19, 93.39, 93.90, 95.90]  # Random
M2 = [90.29, 92.78, 95.59, 97.28, 97.59, 97.82]  # Entropy
M3 = [90.29, 91.47, 93.55, 93.64, 92.75, 94.12]  # CoreSet
M4 = [90.29, 91.14, 92.09, 94.23, 95.96, 96.76]  # BADGE
M5 = [90.29, 91.59, 92.09, 93.10, 95.67, 97.03]  # UncertainGCN
M6 = [90.29, 92.33, 94.44, 96.31, 96.76, 97.77]  # CoreGCN
M7 = [90.29, 91.81, 93.82, 95.04, 96.42, 97.91]  # method

AddNum = [80, 112, 144, 176, 208, 240]

fig = plt.figure(figsize=(6, 4), dpi=1000)
plt.grid()
plt.plot(AddNum, M1, label='Random Sampling', c='palevioletred', marker='o', alpha=0.6)
plt.plot(AddNum, M2, label='Entropy', c='b', marker='o', alpha=0.6)
plt.plot(AddNum, M3, label='CoreSet', c='g', marker='o', alpha=0.6)
plt.plot(AddNum, M4, label='BADGE', c='darkkhaki', marker='o', alpha=0.6)
plt.plot(AddNum, M5, label='UncertainGCN', c='teal', marker='o', alpha=0.6)
plt.plot(AddNum, M6, label='CoreGCN', c='mediumpurple', marker='o', alpha=0.6)
plt.plot(AddNum, M7, label='The proposed method', c='r', marker='o', alpha=0.6)
plt.xticks(AddNum, font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.legend(loc="lower right", prop='Times New Roman')
plt.xlabel('Number of labelled samples', font='Times New Roman')
plt.ylabel('Accuracy(mean of 5 trials)', font='Times New Roman')
plt.title('Testing accuracy of 3D-CNN on SA dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-3D-CNN-SA.png')
plt.show()
print('~')
