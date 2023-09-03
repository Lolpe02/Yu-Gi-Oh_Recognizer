import numpy as np
from sklearn.cluster import KMeans
import os
import cv2
import joblib

# extract images from the folder
def extract_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# turn images in array of RGB values
def rgb_array(images):
    rgb_values = []
    for image in images:
        x, y, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:x // 7,:y-y//10].flatten()
        # print(image.shape)
        for pixel in image:
            rgb_values.append(pixel)
    return np.asarray(rgb_values, dtype=np.uint8).reshape(-1,3)

def Kmeans(data, num_clusters=9 , initial_centroids='k-means++'):
    # Create a K-means clustering model
    kmeans = KMeans(n_clusters=num_clusters,init=initial_centroids, n_init=1, max_iter=500, tol=0.0001)
    print(data.shape)
    # Fit the model to your RGB data
    kmeans.fit(data)
    # Get the cluster centers (representative colors)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    # get the boundaries of the clusters
    joblib.dump(kmeans, 'kmeans_model.pkl')
    return labels, cluster_centers