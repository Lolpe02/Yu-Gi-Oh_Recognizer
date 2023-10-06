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
def col_array(images, mode):
    rgb_values = []
    for image in images:
        x, y, _ = image.shape
        if mode: # HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:x // 7,:y-y//10].flatten()
        if not mode: # RGB
            image = image[:x // 7,:y-y//10].flatten()
        # print(image.shape)
        for pixel in image:
            rgb_values.append(pixel)
    return np.asarray(rgb_values, dtype=np.uint8).reshape(-1,3)

def Kmeans(data, num_clusters=9 , initial_centroids='k-means++'):
    # Create a K-means clustering model
    kmeans = KMeans(n_clusters=num_clusters,init=initial_centroids, n_init=1, max_iter=500, tol=0.0001)
    # Fit the model to your RGB data
    kmeans.fit(data)
    # Get the cluster centers (representative colors)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    # get the boundaries of the clusters
    l = joblib.load('kmeans_model.pkl')
    l.append(kmeans)
    joblib.dump(l, 'kmeans_model.pkl')
    return labels, cluster_centers

# convert list of RGB values to HSV
a = np.asarray([(0,0,0),(58,60,74),(230,227,226),(27,156,139),(175,53,130),(138,71,188),(91,121,174),(186,149,84),(171,104,75)], dtype=np.uint8).reshape(-1,1,3)
init =  cv2.cvtColor(a, cv2.COLOR_BGR2HSV).reshape(-1,3)
a = a.reshape(-1,3)

if __name__ == '__main__':
    joblib.dump([], 'kmeans_model.pkl')
    # extract images from the folder
    images = extract_images('Kmeans_data')
    # turn images in array of RGB values
    rgb_values = col_array(images,0)
    hsv_values = col_array(images,1)
    # Apply Kmeans algorithm
    cluster_centers_bgr = Kmeans(rgb_values, num_clusters=9, initial_centroids=a)
    cluster_centers_hsv = Kmeans(hsv_values, num_clusters=9, initial_centroids=init)
    # print(cluster_centers)