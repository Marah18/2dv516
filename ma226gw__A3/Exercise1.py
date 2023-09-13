import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

data = np.loadtxt("dist.csv", delimiter=";")
X = data[:, :-1]
y = data[:, -1]
n_s = 10000
np.random.seed(7)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]

X_train, y_train = X[r, :], y[r]
X_vali, y_vali = X[:n_s, :], y[:n_s]

kernal_val = {
    "linear": {"C": [0.01, 0.1, 1, 10, 15]},
    "rbf": {"C": [0.01, 0.1, 1, 10, 15], "gamma": [0.01, 0.05, 0.1, 0.5]},
    "poly": {"C": [0.01, 0.1, 1, 10, 15], "degree": [0.01, 0.1, 1, 2, 3, 4, 6], "gamma": [0.01, 0.05, 0.1]}
}


def gridSearch():
    list = {}
    for kernel, var in kernal_val.items():
        clf = SVC(kernel=kernel)
        grid_search = GridSearchCV(clf, var)
        grid_search.fit(X_train, y_train)
        list[kernel] = grid_search.best_estimator_
    return list


best_models = gridSearch()

margin = 0.3
x_min = np.min(X[:, 0]) - margin
x_max = np.max(X[:, 0]) + margin
y_min = np.min(X[:, 1]) - margin
y_max = np.max(X[:, 1]) + margin

grids = 300
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grids),
                     np.linspace(y_min, y_max, grids))
grid = np.c_[xx.ravel(), yy.ravel()]

fig = plt.figure(
    "The decision boundary for the best models with the data", figsize=(16, 8))

# Subplot 1
ax1 = fig.add_subplot(1, 3, 1)
lin_plot = best_models["linear"]
lin_score = lin_plot.score(X_vali, y_vali)
ax1.set_title(f"Linear \nScore: {lin_score}")
ax1.imshow(lin_plot.predict(grid).reshape(xx.shape),
           origin="lower", extent=(x_min, x_max, y_min, y_max))
ax1.scatter(X_train[:, 0], X_train[:, 1],
            cmap="Oranges", c=y_train, marker=".")
ax1.contour(xx, yy, lin_plot.predict(grid).reshape(xx.shape))

# Subplot 2
ax2 = fig.add_subplot(1, 3, 2)
RBF_plot = best_models["rbf"]
RBF_score = RBF_plot.score(X_vali, y_vali)
RBF_gamma = RBF_plot.gamma
ax2.set_title(f"RBF gamma={RBF_gamma}\nScore: {RBF_score}")
ax2.imshow(RBF_plot.predict(grid).reshape(xx.shape),
           origin="lower", extent=(x_min, x_max, y_min, y_max))
ax2.scatter(X_train[:, 0], X_train[:, 1],
            cmap="Oranges", c=y_train, marker=".")
ax2.contour(xx, yy, RBF_plot.predict(grid).reshape(xx.shape))

# Subplot 3
ax3 = fig.add_subplot(1, 3, 3)
poly_plot = best_models["poly"]
poly_score = poly_plot.score(X_vali, y_vali)
poly_degree = poly_plot.degree
ax3.set_title(
    f"Polynomial \n gamma={poly_plot.gamma}, d={poly_degree} \nScore: {poly_score}")
ax3.imshow(poly_plot.predict(grid).reshape(xx.shape),
           origin="lower", extent=(x_min, x_max, y_min, y_max))
ax3.scatter(X_train[:, 0], X_train[:, 1],
            cmap="Oranges", c=y_train, marker=".")
ax3.contour(xx, yy, poly_plot.predict(grid).reshape(xx.shape))

plt.show()
