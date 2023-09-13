# reference for the images in file A3_data is: "https://mattpetersen.github.io/load-mnist-with-numpy"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


data = np.loadtxt("bm.csv", delimiter=",")
X = data[:, :-1]
y = data[:, -1]
np.random.shuffle(data)
rng = np . random . default_rng()

train_indices = rng.choice(len(X), size=9000, replace=True)
X_train = X[train_indices]
y_train = y[train_indices]

test_indices = rng.choice(len(X), size=1000, replace=True)
X_test = X[test_indices]
y_test = y[test_indices]

margin = 0.3
grid_size = 300
x_min = np.min(X[:, 0]) - margin
x_max = np.max(X[:, 0]) + margin
y_min = np.min(X[:, 1]) - margin
y_max = np.max(X[:, 1]) + margin

xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                     np.linspace(y_min, y_max, grid_size))
grid = np.c_[xx.ravel(), yy.ravel()]

predictedS = np.zeros_like(y_test)
boundaryS = np.zeros(grid_size **2)
errors = 0

for i in range(100):
    clf = DecisionTreeClassifier()
    r = np.zeros(9000, dtype=int)
    XX = np.zeros((9000, X_train.shape[1]))

    for j in range(9000):
        r[j] = rng.choice(9000, size=1)
        XX[j] = X_train[r[j]]

    clf.fit(XX, y_train[r])
    y_predict = clf.predict(X_test)
    predictedS += y_predict
    errors += np.mean(y_test != y_predict)
    grid_pred = clf.predict(grid)
    boundaryS += grid_pred
    plt.figure("The decision boundaries of all the models", figsize=(12, 7))
    plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.contour(xx, yy, grid_pred.reshape(xx.shape), colors="black")

plt.figure("The ensemble model", figsize=(10, 5))
plt.contour(xx, yy, (boundaryS > 50).reshape(xx.shape), colors="black")

general_err = round((1.0 - np.mean(y_test == (predictedS > (50)))) * 100, 2)
general_err_ind = round(((errors / 100)) * 100, 3)

print(
    f"a)  The estimate of the generalization error using the test set of the ensemble of 100 decision trees= {general_err} procent\n")
print(
    f"b)  The average estimated generalization error of the individual decision trees= {general_err_ind} procent\n")
plt.show()
print(f"d)  Getting lower generalizations error using the test set of the ensemble of trees,",
      "because it combines multiple different models which can give better predictive, but on the other hand it takes a longer time. ")
