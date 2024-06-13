import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [59.27, 65.31, 68.16, 71.04, 74.99, 75.50]  # Random
M2 = [59.27, 69.15, 73.36, 73.88, 75.48, 78.79]  # Entropy
M3 = [59.27, 60.23, 58.49, 57.91, 61.27, 74.35]  # CoreSet
M4 = [59.27, 56.86, 68.52, 73.74, 77.15, 78.37]  # BADGE
M5 = [59.27, 67.99, 72.64, 73.41, 74.63, 77.60]  # UncertainGCN
M6 = [59.27, 68.99, 73.65, 74.50, 75.77, 78.93]  # CoreGCN
M7 = [59.27, 69.14, 72.70, 74.50, 75.85, 79.02]  # method

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
plt.title('Testing accuracy of A2S2K-ResNet on PU dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-A2S2K-PU.png')
plt.show()
print('~')
