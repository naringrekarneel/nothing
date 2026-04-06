import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os

image_path = r"C:\Users\ASUS\Downloads\Screenshot 2026-03-06 181744.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"File not found: {image_path}")

image = Image.open(image_path)

image = image.convert("RGB")

image = np.array(image)

image = cv2.resize(image, (300, 300))

plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()

pixels = image.reshape(-1, 3)
pixels = np.float32(pixels)

k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(pixels)

centroids = np.uint8(kmeans.cluster_centers_)
labels = kmeans.labels_

segmented = centroids[labels].reshape(image.shape)

plt.imshow(segmented)
plt.title(f"Segmented Image (K={k})")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 2))
for i, color in enumerate(centroids):
    plt.subplot(1, k, i + 1)
    plt.imshow([[color]])
    plt.title(f"C{i+1}")
    plt.axis('off')

plt.suptitle("Cluster Colors")
plt.show()