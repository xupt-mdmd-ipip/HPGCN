import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [46.00, 52.63, 53.29, 59.27, 60.08, 62.26]  # Random
M2 = [46.00, 52.17, 56.83, 58.03, 62.64, 62.91]  # Entropy
M3 = [46.00, 54.51, 56.66, 59.88, 61.26, 63.56]  # CoreSet
M4 = [46.00, 52.63, 53.29, 59.27, 60.08, 62.26]  # BADGE
M5 = [46.00, 54.34, 54.36, 55.20, 56.24, 58.82]  # UncertainGCN
M6 = [46.00, 51.66, 52.88, 57.07, 59.39, 63.05]  # CoreGCN
M7 = [46.00, 51.35, 56.09, 58.37, 60.49, 63.70]  # method

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
plt.title('Testing accuracy of A2S2K-ResNet on IP dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-A2S2K-IP.png')
plt.show()
print('~')
