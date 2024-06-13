import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [86.18, 89.39, 91.68, 92.54, 93.09, 93.61]  # Random
M2 = [86.18, 89.45, 90.63, 92.04, 93.32, 93.55]  # Entropy
M3 = [86.18, 90.43, 91.09, 92.89, 93.26, 94.71]  # CoreSet
M4 = [86.18, 90.55, 90.91, 91.01, 92.58, 93.18]  # BADGE
M5 = [86.18, 88.73, 90.45, 91.73, 92.69, 93.72]  # UncertainGCN
M6 = [86.18, 90.88, 91.36, 92.92, 93.44, 94.62]  # CoreGCN
M7 = [86.18, 89.49, 91.75, 92.60, 93.45, 94.81]  # method

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
plt.title('Testing accuracy of A2S2K-ResNet on SA dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-A2S2K-SA.png')
plt.show()
print('~')
