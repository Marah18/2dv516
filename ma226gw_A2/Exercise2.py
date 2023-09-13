import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/secret_polynomial.csv', header=None, names=['x', 'y'])
X = data['x']
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
degree_list = list(range(1, 7))


def perform_regression(X_train, X_test, y_train, y_test, degree):
    polynomial = PolynomialFeatures(degree=degree)
    X_train_pol = polynomial.fit_transform(X_train.values.reshape(-1, 1))
    X_test_pol = polynomial.transform(X_test.values.reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_pol, y_train)
    y_predact = lin_reg.predict(X_test_pol)
    mse = mean_squared_error(y_test, y_predact)
    indices_s = np.argsort(X_test)
    X_test_sorted = X_test.iloc[indices_s]
    y_pred_sorted = y_predact[indices_s]

    return mse, X_test_sorted, y_pred_sorted


def poly_regression():
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    mse_list = []
    best_degree = None
    best_mse = float('inf')
    for i, degree in enumerate(degree_list):
        mse_list = []
        for run in range(5):
            mse, X_test_sorted, y_pred_sorted = perform_regression(
                X_train, X_test, y_train, y_test, degree)
            mse_list.append(mse)
            ax = axs.flatten()[i]
            ax.set_title(f"Degree = {degree}  MSE = {mse}")
            ax.scatter(X_test, y_test, color='red')
            ax.plot(X_test_sorted, y_pred_sorted, color='green')
            if mse < best_mse:
                best_degree = degree
                best_mse = mse

    plt.tight_layout()
    plt.show()
    return (best_degree, mse)


def main():
    best_degree, mse = poly_regression()
    print(f"The degree gives the best fit is: \n\t{best_degree} with MSE = {mse.round(2)},"
          + " because of the lowest MSE it gives, and more accurate estimate."
          + " In other word the average squared difference between actual and predicted data is lowest")


if __name__ == '__main__':
    main()
