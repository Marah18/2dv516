import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data_train = np.array(pd.read_csv("fashion-mnist_train.csv"))
data_test = np.array(pd.read_csv("fashion-mnist_test.csv"))
np.random.shuffle(data_train)
np.random.shuffle(data_test)
X_train, y_train = (data_train[:, 1:] /
                    255.0).reshape(-1, 28, 28), data_train[:, 0]
X_test, y_test = (data_test[:, 1:] /
                  255.0).reshape(-1, 28, 28), data_test[:, 0]
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
          'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def task1():
    random_samples = np.random.choice(len(X_train), size=16, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(10, 7))
    for i, ax in enumerate(axes.flatten()):
        if i < 16:
            img = X_train[random_samples[i]]
            label = labels[int(y_train[random_samples[i]])]
            ax.imshow(img, cmap='gray')
            ax.set_title(label)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def task2(X_train, y_train, X_test, y_test, labels):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    classes = 10
    y_train = tf.one_hot(y_train, depth=classes)
    y_val = tf.one_hot(y_val, depth=classes)
    y_test = tf.one_hot(y_test, depth=classes)

    # Make an MLP model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=64, epochs=20, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Loss for test set:', loss)
    print('Accuracy for test set:', accuracy)

    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    display = confusion_matrix(y_true_classes, y_pred_class)

    # plot
    plt.imshow(display)
    plt.colorbar()
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, display[i, j], ha='center',
                     va='center', color='gray')
    marks = np.arange(len(labels))
    plt.xticks(marks, labels, rotation=45)
    plt.yticks(marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for ML in Fashion')
    plt.show()


def task3():
    print("Categories with high accuracy and high values along the diagonal are easier to classify ex T-shirt/top\n", "while categories with low accuracy and  low values along the diagonal are hard to classify for example sneaker and Ankle boot.\n"
          "Some Categories could be mixed together for example coat and pullover because they have high values outside the diagonal.\n")


task1()
task2(X_train, y_train, X_test, y_test, labels)
