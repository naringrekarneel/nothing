import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
normal = np.random.randn(250, 2)
outliers = np.random.uniform(-6, 6, (50, 2))
data = np.vstack((normal, outliers))

data = StandardScaler().fit_transform(data)

dbscan = DBSCAN(eps=0.6, min_samples=6)
labels = dbscan.fit_predict(data)

outliers_mask = labels == -1
normal_mask = labels != -1

plt.scatter(data[normal_mask][:, 0], data[normal_mask][:, 1])
plt.scatter(data[outliers_mask][:, 0], data[outliers_mask][:, 1], marker='x')
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Total points:", len(data))
print("Outliers detected:", np.sum(outliers_mask))
print("Clusters found:", len(set(labels)) - (1 if -1 in labels else 0))