import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [56.20, 65.23, 73.14, 74.57, 77.47, 78.51]  # Random
M2 = [56.20, 66.37, 67.67, 70.38, 70.43, 71.86]  # Entropy
M3 = [56.20, 65.96, 69.05, 75.38, 76.83, 79.53]  # CoreSet
M4 = [56.20, 60.26, 67.65, 74.87, 74.18, 77.72]  # BADGE
M5 = [56.20, 60.85, 67.00, 69.54, 70.88, 78.58]  # UncertainGCN
M6 = [56.20, 62.99, 69.70, 75.22, 77.97, 79.45]  # CoreGCN
M7 = [56.20, 61.83, 67.51, 75.89, 78.23, 80.10]  # method

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
plt.title('Testing accuracy of HybridSN on IP dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-HybridSN-IP.png')
plt.show()
print('~')
