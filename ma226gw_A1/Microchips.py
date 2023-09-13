import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

un_chips    = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
# Load the data from the file using numpy
data        = np.loadtxt("microchips.csv", delimiter=",", dtype=None)
# data from first two couloumns x and y saves in variable X and last coloumn that represent color saves in color
X           = data[:, :-1]
z_value     = data[:, -1]

def plot():
    # Create a plot for the data using two colors and two marks red and green
    fig, ax = plt.subplots()
    # Make background for the figure black
    fig.set_facecolor("black")
    # Create a scatter plot of the data, with different markers for the two classes
    plt.scatter(X[z_value == 1, 0], X[z_value == 1, 1], c='green', marker='.', label='OK')
    plt.scatter(X[z_value == 0, 0], X[z_value == 0, 1], c='r', marker='x', label='Fail')

    ax.legend()
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.set_title('Microchip x and y points')
    plt.show()

# function to calculate the distance between two points
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# function to give the nearest points  
def knn(X_train, y_train, X_test, k):
    distances = []
    for i in range(len(X_train)):
        # save calculated distances with the status 0 or 1 in dist
        dist = distance(X_train[i], X_test)
        distances.append((dist, y_train[i]))
    distances.sort()
    nearestPoints = distances[:k]
    counts = np.bincount([neighbor[1] for neighbor in nearestPoints])
    # return 0 or 1 depending on the status of the given point
    return np.argmax(counts)

def calculate_errors(k, train_as_test):
    errorCounter = 0
    for i, test_data in enumerate(train_as_test):
        errors =  int(test_data[2]) == int(knn(X, z_value, (test_data[0], test_data[1]), k))
        if(errors):
            errorCounter +=1
            
    return len(train_as_test) - errorCounter



def status():
    for k in [1, 3, 5, 7]:
        predicts = []
        print(f"k = {k}")
        for i, chip in enumerate(un_chips):
            label = knn(X, z_value, chip, k)
            predicts.append(label)
            print(f"chip{i+1}: {chip} => {'Ok' if label == 1 else 'Fail'}")
        print("\n")


  
def mainMenu():
    while True:
        print("1. To plot the original microchip data using different markers for the two classes OK and Fail ")
        print("2. To predict whether three unknown microchips are likely to be OK or Fail ")
        print("3. Tp print number of training errors")
        print("4. To exit")
        inp = int(input("Enter the number of option : "))
        if (inp == 1):
            plot()
        elif (inp == 2):
            status()
        elif (inp==3):
            for i, k in enumerate([1, 3, 5, 7]):
                print(f"k = {k}, training errors = {calculate_errors(k, data)}")  
        elif (inp == 4):
            False;
            break        
        else:
            mainMenu()

mainMenu()