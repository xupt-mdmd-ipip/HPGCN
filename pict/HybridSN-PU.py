import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1 = [66.46, 73.16, 74.87, 78.57, 81.81, 82.89]  # Random
M2 = [66.46, 66.33, 72.85, 75.66, 77.56, 82.79]  # Entropy
M3 = [66.46, 69.15, 75.69, 75.88, 77.69, 78.34]  # CoreSet
M4 = [66.46, 69.85, 74.95, 78.19, 81.04, 82.41]  # BADGE
M5 = [66.46, 69.54, 72.96, 79.91, 80.30, 82.36]  # UncertainGCN
M6 = [66.46, 77.89, 79.18, 80.35, 80.97, 83.07]  # CoreGCN
M7 = [66.46, 76.52, 77.52, 79.68, 81.11, 83.31]  # method

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
plt.title('Testing accuracy of HybridSN on PU dataset.', font='Times New Roman')

# 显示图像
plt.savefig('F:/Jetbrains/python/Projectzmm/picture/NEW-HybridSN-PU.png')
plt.show()
print('~')
