import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [94.20, 94.66, 95.14, 95.33, 96.03, 96.65]  # Random
M2 = [94.20, 93.89, 93.23, 95.56, 96.04, 95.72]  # Entropy
M3 = [94.20, 93.82, 94.34, 95.45, 95.55, 97.14]  # CoreSet
M4 = [94.20, 95.23, 96.13, 96.78, 97.03, 97.15]  # BADGE
M5 = [94.20, 94.81, 95.48, 95.55, 95.62, 95.95]  # UncertainGCN
M6 = [94.20, 94.88, 95.83, 96.51, 96.93, 98.00]  # CoreGCN
M7 = [94.20, 95.16, 95.77, 95.97, 97.22, 98.05]  # method

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
plt.title('Testing accuracy of HPCAN on SA dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-HPCAN-SA.png')
plt.show()
print('~')
