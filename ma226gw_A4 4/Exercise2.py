import numpy as np
from sklearn.datasets import make_blobs, make_s_curve
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def compute_stress_E(X, Y):
    x_matrix = np.triu(X)
    y_matrix = np.triu(Y)
    mask = (x_matrix != 0)
    zero_checker = np.where(mask, x_matrix, 1)
    stress = np.sum(np.square(y_matrix - x_matrix) / zero_checker)
    stress_E = (1 / np.sum(x_matrix)) * stress
    return stress_E


def sammon(X, max_iterations=100, threshold=0.023, learning_rate=1.0, init="random"):
    if init == "random":
        layout = make_blobs(n_samples=X.shape[0], n_features=2, centers=1, random_state=1337)[0]
    else:
        pca = PCA(n_components=2)
        layout = pca.fit_transform(X)
    
    dis_X = pairwise_distances(X)
    dis_X[dis_X == 0] = 1e-100
    c = np.sum(np.triu(pairwise_distances(X)))
    # number of sampled of x
    num = dis_X.shape[0]
    for n in range(max_iterations+1):
        #distance between two y points
        dis_y = pairwise_distances(layout)
        dis_y[dis_y == 0] = 1e-100
        stress = compute_stress_E(dis_X, dis_y)
        print(f"Iteration: {n}, Stress = {stress}")

        if stress < threshold:
            print(f"Stopped with reached stress = {stress} and iteration: {n}")
            break
        
        if (n) == max_iterations:
            print(f" The maximum number of iterations {n} has been reached!.\n")
            break
        
        equ1 = np.divide((dis_X - dis_y), (dis_y * dis_X)).ravel()
        equ2 = np.reciprocal(dis_y * dis_X).ravel()
        equa3 = (dis_X - dis_y).ravel()
        equa4 = dis_y.ravel()
        equa5 = np.add(1, np.divide((dis_X - dis_y), dis_y)).ravel()


        for i in range(layout.shape[0]):
            start, end = i * num, (i * num) + num
            first = (-2 / c) * np.sum(np.c_[equ1[start:end], equ1[start:end]] * (layout[i] - layout), axis=0)
            second = (-2 / c) * np.sum(np.c_[equ2[start:end], equ2[start:end]] * (np.c_[equa3[start:end], equa3[start:end]] - ((np.square(layout[i] - layout) / equa4[start:end, None]) * equa5[start:end, None])), axis=0)
            layout[i] = layout[i] - (learning_rate * (first / np.abs(second)))

    return layout


if __name__ == "__main__":
    print ("Run..")

    X, y = make_s_curve(random_state=1, n_samples=300)
    fig = plt.figure(figsize=(16, 8))

    # Original X plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=10, azim=280)
    ax1.set_title("Original X")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2])

    # Sammon Mapping plot
    Y = sammon(X)
    ax2 = fig.add_subplot(122)
    ax2.set_title("Sammon Mapping")
    ax2.scatter(Y[:, 0], Y[:, 1])

    plt.tight_layout()
    plt.show()