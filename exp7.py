import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import tkinter as tk
from tkinter import filedialog

# Hide main tkinter window
root = tk.Tk()
root.withdraw()

# ✅ Open file dialog to select image
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

# Check if user selected a file
if not image_path:
    raise ValueError("No image selected!")

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"File does not exist: {image_path}")

# Read image
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Failed to load image. Try a different file.")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
image = cv2.resize(image, (300, 300))

# Show original image
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Reshape image into pixel array
pixels = image.reshape(-1, 3)
pixels = np.float32(pixels)

# Apply KMeans
k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(pixels)

centroids = np.uint8(kmeans.cluster_centers_)
labels = kmeans.labels_

# Reconstruct segmented image
segmented = centroids[labels].reshape(image.shape)

# Show segmented image
plt.imshow(segmented)
plt.title(f"Segmented Image (K={k})")
plt.axis('off')
plt.show()

# Show cluster colors
plt.figure(figsize=(8, 2))
for i, color in enumerate(centroids):
    plt.subplot(1, k, i + 1)
    plt.imshow([[color]])
    plt.title(f"C{i+1}")
    plt.axis('off')

plt.suptitle("Cluster Colors")
plt.show()