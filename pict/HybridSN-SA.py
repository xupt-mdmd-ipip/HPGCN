import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [92.84, 91.93, 93.45, 95.27, 95.42, 96.98]  # Random
M2 = [92.84, 92.78, 95.59, 97.28, 97.59, 97.92]  # Entropy
M3 = [92.84, 91.47, 93.55, 93.64, 92.75, 94.12]  # CoreSet
M4 = [92.84, 93.24, 94.91, 96.60, 96.97, 97.52]  # BADGE
M5 = [92.84, 92.29, 93.43, 93.92, 93.97, 95.98]  # UncertainGCN
M6 = [92.84, 92.76, 94.35, 96.12, 96.73, 97.84]  # CoreGCN
M7 = [92.84, 92.09, 94.25, 95.58, 96.43, 97.96]  # method

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
plt.title('Testing accuracy of HybridSN on SA dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-HybridSN-SA.png')
plt.show()
print('~')
