import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_val_data_splitted(data):
    np.random.seed(1)
    mask = np.random.rand(len(data)) < 0.8
    train_data = data[mask]
    vali_data = data[~mask]
    return train_data, vali_data


def get_mse_list(train_data, best_features, train_features):
    mse_list = []
    for feature in train_features:
        model = LinearRegression()
        X_train = train_data[best_features + [feature]].values
        y_train = train_data.iloc[:, 0].values
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_train, model.predict(X_train))
        mse_list.append(mse)
    return mse_list


def forward_selection(train_data):
    train_features = list(train_data.columns[1:])
    best_features_list = []
    smallest_mse = np.inf

    while train_features:
        mse_list = get_mse_list(
            train_data, best_features_list, train_features)
        best_feature_index = np.argmin(mse_list)
        best_feature = train_features[best_feature_index]
        best_mse = mse_list[best_feature_index]

        if best_mse < smallest_mse:
            best_features_list.append(best_feature)
            smallest_mse = best_mse

        train_features.remove(best_feature)

    return best_features_list


def find_best_model(vali_data, best_models_list):
    best_mse = np.inf
    best_model_index = None
    for i, (model, features) in enumerate(best_models_list):
        X_vali = vali_data[features].values
        y_vali = vali_data.iloc[:, 0].values
        model_mse = mean_squared_error(y_vali, model.predict(X_vali))
        if model_mse < best_mse:
            best_mse = model_mse
            best_model_index = i
    best_model, best_features = best_models_list[best_model_index]
    return best_model, best_features, best_mse


def estimate_best_model(train_data, validation_data, model_list):
    best_models_list = []
    for i in model_list:
        model_features = forward_selection(train_data.iloc[:, :i+1])
        model = LinearRegression()
        X_train = train_data[model_features].values
        y_train = train_data.iloc[:, 0].values
        model.fit(X_train, y_train)
        best_models_list.append((model, model_features))
        print(f"Model {i}: {', '.join(model_features)}")
    best_model, best_features, best_mse = find_best_model(
        validation_data, best_models_list)

    print(f"The best model: {', '.join(best_features)}")
    print(f"The most important feature: {best_features[0]}")
    print("The best MSE: ", best_mse)


def main():
    data = pd.read_csv('data/cars-mpg.csv')
    model_list = [1, 2, 3, 4, 5, 6]
    train_data, val_data = train_val_data_splitted(data)
    estimate_best_model(train_data, val_data, model_list)


if __name__ == '__main__':
    main()
