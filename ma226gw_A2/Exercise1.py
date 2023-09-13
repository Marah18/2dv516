import numpy as np
import matplotlib.pyplot as plt


def normalize(x_norm, X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_data = (x_norm - mean) / std
    return normalized_data


def cost(Xe, y, beta):
    y_p = Xe.dot(beta)
    m = Xe.shape[0]
    mse = np.sum(np.square(y_p - y)) / m
    return mse


def extend_matrix(X, degree=1):
    m, _ = X.shape
    Xe = np.ones((m, 1))
    for i in range(1, degree+1):
        X_power = np.power(X, i)
        Xe = np.concatenate((Xe, X_power), axis=1)
    return Xe


def gradient_descent(Xe, y, alpha):
    iterations = 1000
    beta = np.zeros(Xe.shape[1])
    grad_des = []
    for i in range(iterations):
        beta = beta - (np.dot((alpha * Xe.T), ((np.dot(Xe, beta)) - y)))
        cost_check = cost(Xe, y, beta)
        grad_des.append([i, cost_check])
    return grad_des, beta 

# Task1: normalizing X
def normalized_X():
    normalized_X = normalize(X, X)
    return normalized_X


def plot_X():
    X_normal = normalized_X()
    plt.figure("Task 1, plot X", figsize=(14, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for i in range(X_normal.shape[1]):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(X_normal[:, i], y, marker="o", c="green")
        plt.xlabel(f"Feature number{i+1}")
        plt.ylabel("y")
        plt.title(f"plot number {i+1}")
    plt.tight_layout()


def compute_beta():
    X_normal = normalized_X()
    Xe_normal = extend_matrix(X_normal)
    beta = np.linalg.solve(Xe_normal.T.dot(Xe_normal), Xe_normal.T.dot(y))
    return beta


def get_cost():
    X_normal = normalized_X()
    Xe_normal = extend_matrix(X_normal)
    beta = compute_beta()
    print("The benchmark result is: \n\t", np.dot(
        np.append([1], normalized_n), beta))
    print("The cost value using the beta computed by the normal equation: \n\t", cost(
        Xe_normal, y, beta))


def predict_result(beta, test_set, X):
    feature_val = np.zeros(len(test_set))
    for i in range(len(test_set)):
        feature_val[i] = normalize(test_set[i], X[:, i])
    feature_val = np.insert(feature_val, 0, 1)
    predicted = np.dot(beta, feature_val)
    return predicted


def find_hyper_bench():
    n_X = np.zeros(X.shape)
    for i in range(X.shape[1]):
        n_X[:, i] = normalize(X[:, i], X[:, i])
    Xe = np.c_[np.ones((len(X), 1)), n_X]
    gd, beta = gradient_descent(Xe, y, 0.01)
    feature_val = [2432, 1607, 1683, 8, 8, 256]
    predicted = predict_result(beta, feature_val, X)
    print(f"Hyperparameter: \n\t "
          + "alpha = 0.01    Iterations = 1000")
    print("Predicted Benchmark Result:\n\t", predicted)
    plt.show()


data = np.loadtxt("data/GPUBenchmark.csv", delimiter=",")
X = data[:, :-1]
y = data[:, -1]
normalized_n = normalize(np.array([2432, 1607, 1683, 8, 8, 256]), X)

def main():
    plot_X()
    compute_beta()
    get_cost()
    find_hyper_bench()


if __name__ == "__main__":
    main()
