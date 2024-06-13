import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [65.45, 86.12, 88.84, 89.92, 90.63, 94.38]  # Random
M2 = [65.45, 78.89, 81.22, 83.06, 86.59, 86.69]  # Entropy
M3 = [65.45, 86.43, 86.63, 90.21, 91.82, 92.29]  # CoreSet
M4 = [65.45, 88.07, 90.43, 91.00, 91.43, 92.09]  # BADGE
M5 = [65.45, 83.79, 85.43, 86.71, 88.14, 88.98]  # UncertainGCN
M6 = [65.45, 86.55, 87.61, 88.40, 91.76, 92.72]  # CoreGCN
M7 = [65.45, 87.33, 88.59, 93.51, 94.23, 94.38]  # method

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
plt.title('Testing accuracy of HPCAN on PU dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-HPCAN-PU.png')
plt.show()
print('~')
