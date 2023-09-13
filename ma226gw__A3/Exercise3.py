
from time import time
import numpy as np
import os
import gzip
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

data_directory = "A3_data"
def vectorization(X_e, y, more=False):
    beta = np.zeros(X_e.shape[1])
    data = []
    for i in range(1000):
        g_Xb = 1 / (1 + np.exp(-X_e.dot(beta)))
        gradient = X_e.T.dot(g_Xb - y)
        beta -= (0.3 / X_e.shape[0]) * gradient
        if more:
            cost = -np.mean(np.log(g_Xb) * y + np.log(1 - g_Xb) * (1 - y))
            data.append(cost)
    if more:
        return beta, np.array(data)
    else:
        return beta
    
def read_file(file):
    filepath = os.path.join(data_directory, file)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File: '{file}' doesn't exist.")
    with gzip.open(filepath, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).copy()
    
def gen_parameters():
    kernel_P = ["rbf"]
    gamma_P = np.logspace(-3, -1, num=5)
    C_P = np.logspace(0, 1, num=5)
    parameters = [{"kernel": kernel_P, "gamma": gamma_P, "C": C_P}]
    return parameters


X = read_file("train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
X_test = read_file("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
y = read_file("train-labels-idx1-ubyte.gz")[8:]
y_test = read_file("t10k-labels-idx1-ubyte.gz")[8:]

np.random.seed(7)
shuffle = np.random.permutation(len(y))
X = X[shuffle]
y = y[shuffle]
X, y = X[shuffle, :], y[shuffle]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25)
X_train = X_train / 255.0
X_validation = X_validation / 255.0
X_test = X_test / 255.0

# 1vs1
print("\tOne vs One data \n")
parameters = gen_parameters()
print("Searching for best hyperparemeters using grid search: ")
clf = GridSearchCV(SVC(), parameters)
clf.fit(X_validation, y_validation)
print("Best hyperparemeters:")
best_parameters = clf.best_params_
for key, value in best_parameters.items():
    print(f"{key}: {value}")
start_time = time()
clf = clf.best_estimator_
clf.fit(X_train, y_train)
predict1 = clf.predict(X_test)
print("\tOne vs One data \n")
print("Time:", time() - start_time, "seconds")
accuracy1 = accuracy_score(predict1, y_test) * 100.0
print("Train accuracy for One vs One:", clf.score(X_train, y_train) * 100)
print("Test accuracy for One vs One:", clf.score(X_test, y_test) * 100)
print(classification_report(y_test, predict1))

# 1vsall.
best_pred = np.full(y_test.shape, np.finfo(dtype=float).min)
X_test_n      = np.c_[np.ones((X_test.shape[0], 1)), X_test]
X_train_n     = np.c_[np.ones((X_train.shape[0], 1)), X_train]
predicts   = []

start_time2 = time()
for label in range(10):
    y_train_ = y_train == label
    beta = vectorization(X_train_n, y_train_)
    prediction = X_test_n.dot(beta)
    predicts.append(prediction)
predicted_labels = np.argmax(predicts, axis=0)
print("Time:", time() - start_time2, "seconds")
accuracy2 = accuracy_score(predicted_labels, y_test) * 100.0

print("\tOne vs all data")
print("Accuracy for One vs all:", accuracy2)


# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
confusion_matrix1 =confusion_matrix(y_test, predict1)
display1 = ConfusionMatrixDisplay(confusion_matrix1)
ax1.set_title("One vs One")
ax1.text(0.5, -0.15, f"Accuracy: {round(accuracy1, 3)}", transform=ax1.transAxes, ha='center')
display1.plot(ax=ax1)

confusion_matrix2  =confusion_matrix(y_test, predicted_labels)
display2 = ConfusionMatrixDisplay(confusion_matrix2)
ax2.set_title("One vs All")
display2.plot(ax=ax2)
ax2.text(0.5, -0.15, f"Accuracy: {round(accuracy2, 3)}", transform=ax2.transAxes, ha='center')
plt.tight_layout()
plt.show()

# results:
print("Conclusion: The classifier that is built by one vs one pattern gives better results in predictions and accuracy."
        , "On the other hand, one vs all pattern is faster in the implementation process. "
        , "However, both strategies gave good results with accuracy higher than 90 percent")
