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

# convert list of RGB values to HSV
a = np.asarray([(0,0,0),(58,60,74),(230,227,226),(27,156,139),(175,53,130),(138,71,188),(91,121,174),(186,149,84),(171,104,75)], dtype=np.uint8).reshape(-1,1,3)
print(a.shape)
init =  cv2.cvtColor(a, cv2.COLOR_BGR2HSV).reshape(-1,3)


if __name__ == '__main__':

    # extract images from the folder
    images = extract_images('Kmeans_data')
    # turn images in array of RGB values
    rgb_values = rgb_array(images)
    # Apply Kmeans algorithm
    cluster_centers = Kmeans(rgb_values, num_clusters=6, initial_centroids=init)
    # print(cluster_centers)



    # color = np.array((205,241,10), dtype=np.uint8)
    # low = np.array((100,100,9), dtype=np.uint8)
    # high = np.array((200,255,100), dtype=np.uint8)
    # print(all(color > low) and all(color < high))
    # gold = np.array([(146, 115, 75), (137, 108, 69), (243,219,173),(204,179,142), (127, 94, 48), (237, 203, 165), (143, 118, 78),
    #                  (217,190,116), (239,196,161), (225,200,172),(197,141,108),(187,129,91),(173,157,125)], dtype=np.uint8).reshape(-1, 3)
    #
    #
    # min_gold = (np.min(gold[:,0]), np.min(gold[:,1]), np.min(gold[:,2]))
    # max_gold = (np.max(gold[:,0]), np.max(gold[:,1]), np.max(gold[:,2]))
    # print(min_gold, max_gold)