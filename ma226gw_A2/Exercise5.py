import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


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


def map_feature(X1, X2, degree):
    X = np.column_stack((X1, X2))
    for i in range(2, (degree + 1)):
        for l in range(i + 1):
            po_feature = np.power(X1, (i - l)) * np.power(X2, l)
            X = np.column_stack((X, po_feature))
    return X


def plot_decision_bound(C, X1, X2, y, degree, test_grid):
    C.fit(map_feature(X1, X2, degree), y)
    predict_grid = C.predict(map_feature(test_grid[0], test_grid[1], degree))
    y_predict = C.predict(map_feature(X1, X2, degree))
    errors = np.sum(y != cross_val_predict(C, map_feature(X1, X2, degree), y))
    plt.subplot(3, 3, degree)
    plt.gca().set_title(
        f"Training errors = {np.sum(y != y_predict)} for Degree: {degree}")
    print(f"\tTraining errors = {np.sum(y != y_predict)} for Degree: {degree}")
    plt.imshow(predict_grid.reshape(500, 500), origin="lower", extent=(X1.min(
    ), X1.max(), X2.min(), X2.max()), cmap=ListedColormap(["#aaaaff", "#ffffaa"]))
    plt.scatter(X1, X2, c=y, cmap=ListedColormap(["green", "red"]), marker=".")


def plott_classif(C, X1, X2, y, t, degree_list):
    errors = []
    plt.figure(f"C = {C.C}", figsize=(14, 8))
    for degree in degree_list:
        print(f"C = {C.C}")
        plot_decision_bound(C, X1, X2, y, degree, t)
        errors.append(np.sum(y != cross_val_predict(
            C, map_feature(X1, X2, degree), y)))
    plt.tight_layout()
    return errors

def plott_error_val(X1, X2, y, t, degree_list):
    c_1 = LogisticRegression(C=10000., max_iter=1000, solver="lbfgs",)
    c_2 = LogisticRegression(C=1., max_iter=1000, solver="lbfgs")
    plot_c1 = plott_classif(c_1, X1, X2, y, t, degree_list)
    plot_c2 = plott_classif(c_2, X1, X2, y, t, degree_list)
    fig, ax = plt.subplots()
    ax.set_xlabel("Degree")
    ax.set_ylabel("Error number")
    ax.plot(degree_list, plot_c1, label=f"C = {c_1.C}")
    ax.plot(degree_list, plot_c2, label=f"C = {c_2.C}")
    ax.legend()
    plt.show()


def main():
    X1, X2, y = load_dataset("data/microchips.csv")
    xx, yy = get_mesh_grid(X1, X2)
    degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    t = xx, yy
    plott_error_val(X1, X2, y, t, degree_list)


if __name__ == "__main__":
    main()
