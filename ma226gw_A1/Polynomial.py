from matplotlib import pyplot as plt
import numpy as np


data        = np.loadtxt("Polynomial200.csv", delimiter=",", dtype=None)
trainingS   = data[0:100]
testS       = data[100:]

k_values    = [1, 3, 5, 7, 9, 11]
x_values    = np.linspace(1, 25, 250)

# function for the first plot
def plt1():
    fig = plt.figure("Training set and Test set", figsize=(14, 7))
    # the first subplot in a 1 row, 2 column grid
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel("X label")
    ax1.set_ylabel("Y label")
    ax1.title.set_text("Training set points")
    ax1.scatter(trainingS[:, 0], trainingS[:, 1], marker="*")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel("X label")
    ax2.set_ylabel("Y label")
    ax2.title.set_text("Test set points")
    ax2.scatter(testS[:, 0], testS[:, 1], marker="*")
    plt.show()


def get_nermast(x1, x2, axis=1):
    return np.linalg.norm(x1 - x2, axis=axis)


def regression(k, data_set):
    neighpors = []
    for i in data_set:
        near = get_nermast(
            np.array([trainingS[:, 0]]).transpose(), np.array([i]))
        sorted_indices = near.argsort()
        k_neighbors = trainingS[:, 1][sorted_indices][:k]
        mean_of_k_neighbors = np.mean(k_neighbors, dtype=np.float64)
        neighpors.append(mean_of_k_neighbors)
    return np.array(neighpors)


def calc_mse(value_true, value_pred, decimals=-1):
    # calculate Mean Squared Error
    MSE = np.square(np.subtract(value_true, value_pred)).mean()
    return round(MSE, decimals)


def plot2():
    fig = plt.figure(
        "k-NN Regression  with different values of k, ", figsize=(14, 7))
    for i, k in enumerate(k_values):
        ax = fig.add_subplot(231 + i)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        regressY = regression(k, x_values)
        ax.scatter(trainingS[:, 0], trainingS[:, 1], marker=".", c="blue")
        ax.plot(x_values, regressY, color="green")
        estimated_Ytrain = regression(k, trainingS[:, 0])
        ax.title.set_text(
            f"k = {k}, MSE = {calc_mse(trainingS[:, 1], estimated_Ytrain, 2)}")

        estimated_Ytest = regression(k, testS[:, 0])
        print(f"MSE test error for k = {k}:", calc_mse(
            testS[:, 1], estimated_Ytest, 2), sep="\n ")
    plt.show()


def explain():
    print("Lowest MSE value for the test data is the best")
    print("\tBecause the minimum MSE test result is best to fit and handle ")
    print("Higher degree of MSE train set gives higher order polynomials, which makes it easier to fit the training data better. ")


def mainMenu():
    while True:
        print("1. To plot the training and test set side-by-side in a 1 × 2 pattern")
        print("2. To plot the k-NN regression result and the MSE training error for different k_values")
        print("3. To exit")
        inp = int(input("Enter the number of option : "))
        if (inp == 1):
            plt1()
        elif (inp == 2):
            explain()
            plot2()
        elif (inp == 3):
            False
            break
        else:
            mainMenu()


mainMenu()
