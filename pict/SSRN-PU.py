import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [74.18, 79.17, 81.52, 83.43, 85.21, 87.12]  # Random
M2 = [74.18, 80.12, 82.71, 83.09, 86.14, 87.63]  # Entropy
M3 = [74.18, 79.17, 82.03, 82.93, 85.94, 88.15]  # CoreSet
M4 = [74.18, 78.17, 81.76, 82.77, 86.91, 87.34]  # BADGE
M5 = [74.18, 79.76, 81.91, 83.75, 86.10, 86.78]  # UncertainGCN
M6 = [74.18, 81.23, 82.14, 84.20, 85.84, 88.49]  # CoreGCN
M7 = [74.18, 79.56, 81.45, 83.68, 85.98, 88.65]  # method

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
plt.title('Testing accuracy of SSRN on PU dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-SSRN-PU.png')
plt.show()
print('~')
