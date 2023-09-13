import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import numpy as np
from scipy.io import arff
from Exercise2 import sammon as sammon
warnings.filterwarnings("ignore") 


def fixed_features(features):
    # float
    features = features.astype(np.float64)
    # imput missed values
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(features)

def fixed_labels(labels):
    # string
    labels = labels.astype(str)
    # get number of different labels
    diff_labels, y = np.unique(labels, return_inverse=True)
    return y, diff_labels

def loading_data(file_path, x_lower, x_upper, y_pos):
    arff_data, _ = arff.loadarff(file_path)
    data = np.array(arff_data.tolist())

    X = fixed_features(data[:, x_lower:x_upper])
    y, labels = fixed_labels(data[:, y_pos])

    return X, y, labels


def plotting(X, y, labels, title, plt_i):
    norm = Normalize(vmin=min(y), vmax=max(y))
    cmap = get_cmap("Set3")  

    ax = plt.subplot(3, 3, plt_i)
    ax.set_title(title)

    for i, label in enumerate(labels):
        mask = y == i
        scatter = plt.scatter(X[mask, 0], X[mask, 1], c=cmap(norm(y[mask])), label=label, marker=".")

    ax.legend()

def visualization(dataset_paths):
    X_list, y_list, labels_list, titles = [], [], [], []

    for file_path, x_lower, x_upper, y_pos in dataset_paths:
        X, y, labels = loading_data(file_path, x_lower, x_upper, y_pos)
        X_list.append(X)
        y_list.append(y)
        labels_list.append(labels)
        titles.append(file_path.split("/")[-1].split(".")[0])

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    
    plt.figure(figsize=(16, 8))
    for i, (X, y, labels, title) in enumerate(zip(X_list, y_list, labels_list, titles), start=1):
        Y_tsne = tsne.fit_transform(X)
        Y_pca = pca.fit_transform(X)
        Y_sammon = sammon(X)
        
        plotting(Y_tsne, y, labels, f"{title} T-SNE", i )
        plotting(Y_pca, y, labels, f"{title} PCA", i +3)
        plotting(Y_sammon, y, labels, f"{title} Sammon", i + 6)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.3)
    plt.show()


if __name__ == "__main__":
    
    print ("Run..")
    dataset_paths = [
        ("datasets/synthetic_control.arff", 0, -1, -1),
        ("datasets/dataset_16_mfeat-karhunen.arff", 0, -1, -1),
        ("datasets/analcatdata_authorship.arff", 0, -1, -1)
    ]

    visualization(dataset_paths)

