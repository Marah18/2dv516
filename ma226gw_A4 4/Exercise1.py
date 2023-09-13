import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def bkmeans(X, k, number_it):
    samples_number = X.shape[0]
    k_clusters = np.zeros(samples_number, dtype=int)
    counter = 1
    while counter < k:
        cluster_counts = np.bincount(k_clusters)
        dom_cluster = np.argmax(cluster_counts)
        kmeans = KMeans(n_clusters=2, n_init=number_it)
        dominant_indices = np.where(k_clusters == dom_cluster)[0]
        cluster_label = kmeans.fit(X[dominant_indices]).labels_
        new_values = np.where(cluster_label == 0, dom_cluster, counter)
        k_clusters[dominant_indices] = new_values
        counter += 1
    return k_clusters

if __name__ == "__main__":
    
    X, y = make_blobs(centers=6,n_samples=700, random_state=4)
    k = 6
    i = 100
    c = bkmeans(X, k, i)

    plt.title("the clustering algorithm Bisecting k-Means")
    plt.scatter(X[:, 0],X[:, 1], c=c)
    plt.show()
    

