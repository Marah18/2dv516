import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_dataset(path):
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    X1 = X[:, 0]
    X2 = X[:, 1]
    return X1, X2, y


def get_mesh_grid(X1, X2):
    xx, yy = np.meshgrid(np.linspace(X1.min() - 0.1, X1.max() + 0.1, 500),
                         np.linspace(X2.min() - 0.1, X2.max() + 0.1, 500))
    xx, yy = xx.ravel(), yy.ravel()
    return xx, yy


def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))


def get_decision_boundary(X1, X2, Y, iteration, costs, test_pred_prob, train_pred_label):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].set_title(f"Training errors = {np.sum(Y != train_pred_label)}")
    axs[0].set_xlabel("Feature 1")
    axs[0].set_ylabel("Feature 2")
    axs[0].imshow((test_pred_prob > 0.5).reshape(500, 500), origin="lower",
                  extent=(X1.min(), X1.max(), X2.min(), X2.max()),
                  cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
    axs[0].scatter(X1, X2, c=Y, cmap=ListedColormap(
        ["blue", "green"]), marker="*")
    axs[1].set_title("cost function over iterations")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Cost")
    axs[1].plot(range(iteration), costs)
    plt.show()



def gradient_descent(X, y, a=0.01, iter=1000):
    g = np.zeros(X.shape[1])
    cost_list = []
    for i in range(iter):
        g = g - (a * X.T.dot((sigmoid(X.dot(g)) - y))) / X.shape[0]
        cost_list.append(cost(X, y, g))
    return g, cost_list


def cost(X, y, beta):
    eps = 1e-5
    hat = sigmoid(np.dot(X, beta))
    cost = -(np.dot(y, np.log(hat + eps)) +
             np.dot((1 - y), np.log(1 - hat + eps))) / len(y)
    return cost


def plotX_y(X1, X2, Y):
    fig, ax = plt.subplots(figsize=(16, 8))
    scatter = ax.scatter(X1, X2, c=Y, cmap=ListedColormap(
        ["green", "purple"]), marker="*")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Microchips data")
    ax.legend(handles=scatter.legend_elements()[
              0], labels=["Class 0", "Class 1"])
    plt.show()


def map_feature(X1, X2, degree):
    X = np.column_stack((X1, X2))
    for i in range(2, (degree + 1)):
        for l in range(i + 1):
            po_feature = np.power(X1, (i - l)) * np.power(X2, l)
            X = np.column_stack((X, po_feature))
    return X


def find_b_and_plot_5(X1, X2, y, xx, yy):
    Xe = map_feature(X1, X2, 5)
    alpha = 5
    iteration = 10000
    beta, costs = gradient_descent(Xe, y, alpha, iteration)
    print(f"Beta:\n{beta}")
    test_grid = map_feature(xx, yy, 5)
    test_pred = sigmoid(np.dot(test_grid, beta))
    train_label = np.round(sigmoid(np.dot(Xe, beta)))
    print(f"Hyperparameter {alpha} and Number of iteration = {iteration}")
    get_decision_boundary(X1, X2, y, iteration, costs, test_pred, train_label)


def find_b_and_plot_7(X1, X2, y, xx, yy):
    Xe = map_feature(X1, X2, 5)
    alpha = 7
    iteration = 200000
    beta, costs = gradient_descent(Xe, y, alpha, iteration)
    test_grid = map_feature(xx, yy, 5)
    test_pred = sigmoid(np.dot(test_grid, beta))
    train_label = np.round(sigmoid(np.dot(Xe, beta)))
    print(f"Hyperparameter: alpha= {alpha} and Number of iteration = {iteration}")
    get_decision_boundary(X1, X2, y, iteration, costs, test_pred, train_label)

def mean():
    X1, X2, y = load_dataset("data/microchips.csv")
    plotX_y(X1, X2, y)
    xx, yy = get_mesh_grid(X1, X2)
    find_b_and_plot_5(X1, X2, y, xx, yy)
    find_b_and_plot_7(X1, X2, y, xx, yy)
    
if __name__ == "__main__":
    mean()
