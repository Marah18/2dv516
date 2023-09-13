from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from Exercise1 import bkmeans 
from Exercise2 import sammon
from Exercise3_1 import loading_data


def plotting_clust(X, y, labels, titles):
    iters = 100
    for r in range(len(X)):
        bkm = bkmeans(X[r], len(labels[r]), iters)
        k_means = KMeans(n_clusters=len(labels[r]), n_init=iters).fit_predict(X[r])
        agglomerative = AgglomerativeClustering(n_clusters=len(labels[r]), linkage='ward').fit_predict(X[r])
        for c, Y in enumerate([bkm, k_means, agglomerative]):
            norm = Normalize(vmin=0, vmax=max(y[r]))
            plt.subplot(3, 3, [1, 4, 7][r] + c)
            plt.title(f'{titles[r]} | {["Bk-Means", "Classic k-Means", "Agglomerative"][c]}')
            plt.scatter(X[r][:, 0], X[r][:, 1], c=Y, marker=".")

def visualization(dataset_paths):
    X_list, y_list, labels_list, titles = [], [], [], []

    for file_path, x_lower, x_upper, y_pos in dataset_paths:
        X, y, labels = loading_data(file_path, x_lower, x_upper, y_pos)
        X_list.append(X)
        y_list.append(y)
        labels_list.append(labels)
        titles.append(file_path.split("/")[-1].split(".")[0])

    imputer = SimpleImputer(strategy="mean")
    X_list[2] = imputer.fit_transform(X_list[2])
    Y_list = [sammon(X) for X in X_list]

    plt.figure(figsize=(16, 8))
    plotting_clust(Y_list, y_list, labels_list, titles)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == "__main__":
    print("Running...")
    
    dataset_paths = [

        ("datasets/synthetic_control.arff", 0, -1, -1),
        ("datasets/dataset_16_mfeat-karhunen.arff", 0, -1, -1),
        ("datasets/analcatdata_authorship.arff", 0, -1, -1)
    ]
    
    visualization(dataset_paths)

