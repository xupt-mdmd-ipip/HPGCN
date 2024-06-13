import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [75.54, 78.19, 82.06, 84.42, 87.75, 88.52]  # Random
M2 = [75.54, 76.62, 79.43, 81.48, 87.57, 87.35]  # Entropy
M3 = [75.54, 82.89, 85.35, 87.02, 86.10, 90.44]  # CoreSet
M4 = [75.54, 76.46, 78.53, 82.36, 85.29, 87.82]  # BADGE
M5 = [75.54, 80.88, 80.02, 80.43, 82.64, 87.30]  # UncertainGCN
M6 = [75.54, 81.30, 85.37, 86.31, 89.97, 90.05]  # CoreGCN
M7 = [75.54, 79.37, 84.09, 85.65, 87.85, 91.05]  # method

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
plt.title('Testing accuracy of HPCAN on IP dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-HPCAN-IP.png')
plt.show()
print('~')
