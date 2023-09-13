import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/banknote_authentication.csv", header=None).values
np.random.shuffle(data)
X_train, y_train = data[:1000, :4], data[:1000, 4]
X_val, y_val = data[1000:1200, :4], data[1000:1200, 4]
X_test, y_test = data[1200:, :4], data[1200:, 4]


def normalize(norm_X, X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_data = (norm_X - mean) / std
    return normalized_data


def sigmoid(z):
    return np.divide(1, 1 + np.exp(-z))

def predict(X, beta):
    X_norm = normalize(X, X)
    X_norm = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
    return sigmoid(np.dot(X_norm, beta)) >= 0.5


def cost_fun(Xe, y, b):
    z = np.dot(Xe, b)
    prob = sigmoid(z)
    cost_fun = -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    return cost_fun


def logic_regration(X, y, alpha=0.01, iterations=1000):
    X_n = normalize(X, X)
    # add coloumn with 1
    X_n = np.hstack((np.ones((X_n.shape[0], 1)), X_n))
    beta = np.zeros(X_n.shape[1])
    cost_list = []
    for i in range(iterations):
        z = np.dot(X_n, beta)
        prob = sigmoid(z)
        cost_value = cost_fun(X_n, y, beta)
        cost_list.append(cost_value)
        gradient = np.mean((prob - y)[:, np.newaxis] * X_n, axis=0)
        beta = np.subtract(beta, np.multiply(alpha, gradient))
    return (beta, cost_list)


def plot_cost(cost_fun):
    plt.plot(cost_fun)
    plt.title("cost function over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("cost")
    plt.show()


def comute_train_acc(beta):
    y_train_predict = predict(X_train, beta)
    train_accuracy = np.mean(y_train_predict == y_train)
    train_error = np.sum(y_train_predict != y_train)
    print(f"Training error: {train_error}")
    print(f"Training accuracy: {train_accuracy*100} %")


def comute_test_acc(beta):
    y_test_predict = predict(X_test, beta)
    test_accuracy = np.mean(y_test_predict == y_test)
    test_error = np.sum(y_test_predict != y_test)
    print(f"Test error: {test_error}")
    print(f"Test accuracy: {round(test_accuracy*100, 2)} % ")


def main():
    # look after best hyperparameter
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    num_iterations = [500, 1000, 2000, 3000]

    best_accuracy = 0.0
    best_hyper_p = None

    for i in learning_rates:
        for iterations in num_iterations:
            beta, cost_fun = logic_regration(
                X_train, y_train, alpha=i, iterations=iterations)

            y_val_predict = predict(X_val, beta)
            accuracy = np.mean(y_val_predict == y_val)

            print("Hyperparameters: alpha={}, iteration number={}".format(
                i, iterations))
            print(f"Validation accuracy: {accuracy*100} %")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyper_p = (i, iterations)

    print("Best hyperparameters: alpha={}, iteration number={}".format(*best_hyper_p))

    # model with the best hyperparameters using training data
    best_beta, _ = logic_regration(
        X_train, y_train, alpha=best_hyper_p[0], iterations=best_hyper_p[1])

    plot_cost(cost_fun)
    comute_train_acc(best_beta)
    comute_test_acc(best_beta)
    print("The model learns from the training dataset trying to make the training loss (error) less. \n"
          + "The benchmark helps to make the model performance better, avoiding to overfiiting(big data size) or underfitting(small data size) between training and test accuracy.\n"
          + "The best model is the model that gives the highest training and test accuracy, "
          + "After running mutable times the differance is small with changing the number of data"
          + "However the test error can change depending on the data size used, but the accurancy procent remains approximately the same")


if __name__ == '__main__':
    main()
