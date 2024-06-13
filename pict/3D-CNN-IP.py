import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [57.66, 62.91, 66.78, 72.86, 73.86, 77.56]  # Random
M2 = [57.66, 65.76, 70.30, 71.78, 77.50, 76.86]  # Entropy
M3 = [57.66, 63.46, 67.05, 73.75, 74.83, 78.03]  # CoreSet
M4 = [57.66, 65.59, 69.85, 70.87, 76.44, 77.72]  # BADGE
M5 = [57.66, 66.04, 72.60, 73.16, 74.38, 74.81]  # UncertainGCN
M6 = [57.66, 64.76, 66.96, 73.14, 74.00, 78.15]  # CoreGCN
M7 = [57.66, 63.40, 69.58, 74.57, 76.56, 78.25]  # method

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
plt.title('Testing accuracy of 3D-CNN on IP dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-3D-CNN-IP.png')
plt.show()
print('~')
