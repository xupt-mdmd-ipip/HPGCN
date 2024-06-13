import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [70.72, 70.09, 75.93, 76.87, 77.25, 78.62]  # Random
M2 = [70.72, 71.03, 73.27, 78.01, 79.27, 79.55]  # Entropy
M3 = [70.72, 71.89, 74.45, 76.58, 77.15, 79.86]  # CoreSet
M4 = [70.72, 70.70, 73.25, 74.61, 77.82, 76.26]  # BADGE
M5 = [70.72, 65.88, 75.12, 75.59, 76.36, 79.59]  # UncertainGCN
M6 = [70.72, 73.10, 74.40, 77.15, 78.62, 79.86]  # CoreGCN
M7 = [70.72, 73.00, 74.34, 77.62, 78.17, 80.22]  # method

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
plt.title('Testing accuracy of SSRN on IP dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-SSRN-IP.png')
plt.show()
print('~')
