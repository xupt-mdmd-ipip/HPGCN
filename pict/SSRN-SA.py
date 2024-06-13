import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [93.15, 93.48, 94.91, 95.15, 96.43, 96.57]  # Random
M2 = [93.15, 93.82, 94.75, 95.19, 96.82, 97.29]  # Entropy
M3 = [93.15, 94.65, 94.78, 95.64, 96.87, 97.36]  # CoreSet
M4 = [93.15, 93.73, 94.54, 95.66, 96.72, 97.20]  # BADGE
M5 = [93.15, 93.23, 94.53, 95.57, 96.73, 97.23]  # UncertainGCN
M6 = [93.15, 93.71, 95.35, 95.64, 96.68, 97.59]  # CoreGCN
M7 = [93.15, 94.04, 94.44, 95.83, 96.52, 97.67]  # method

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
plt.title('Testing accuracy of SSRN on SA dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-SSRN-SA.png')
plt.show()
print('~')
