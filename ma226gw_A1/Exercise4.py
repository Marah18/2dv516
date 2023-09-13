import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

un_chips       = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
KNN            = KNeighborsClassifier()
data           = np.loadtxt("microchips.csv", delimiter=",", dtype=None)
X              = data[:, :-1]
z_value        = data[:, -1]

KNN.fit(X, z_value)


def plot2():
    # Create a plot for the data using two colors and two marks red and green
    fig, ax = plt.subplots()
    # Make background for the figure black
    fig.set_facecolor("black")

    plt.scatter(X[z_value == 1, 0], X[z_value == 1, 1], c='green', marker='.', label='OK')
    plt.scatter(X[z_value == 0, 0], X[z_value == 0, 1], c='r', marker='x', label='Fail')

    ax.legend()
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.set_title('Microchip x and y points')
    plt.show()


def decision_boundary():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
    for i, k in enumerate([1, 3, 5, 7]):
        ax = fig.add_subplot(221 + i)
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-1.5, 1.5, 100)
        xx, yy = np.meshgrid(x,y)
        grid = np.c_[xx.ravel(), yy.ravel()]
        i_grid= np.array(KNN.predict( grid))
        
        i_grid = i_grid.reshape(xx.shape)
        ax.contourf(xx, yy, i_grid, alpha = 0.4,  extend = "both")
        ax.scatter(X[z_value == 1, 0], X[z_value == 1, 1], c='green', marker='.', label='OK')
        ax.scatter(X[z_value == 0, 0], X[z_value == 0, 1], c='r', marker='x', label='Fail')
        
        for i, k in enumerate([1, 3, 5, 7]):
            KNN.n_neighbors = k
            ax = axes[i//2, i%2]  # get the current axis for the subplot
            ax.set_title(f"k = {k}, training errors = {calculate_errors()}")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            
            ax.contourf(xx, yy, i_grid, alpha = 0.4,  extend = "both")
            ax.scatter(X[z_value == 1, 0], X[z_value == 1, 1], c='green', marker='.', label='OK')
            ax.scatter(X[z_value == 0, 0], X[z_value == 0, 1], c='r', marker='x', label='Fail')
        
            
        fig.tight_layout()
        plt.show()

 
def calculate_errors():
    errors = KNN.predict(X) == z_value
    return np.size(errors) - np.count_nonzero(errors)
 
 
def status():
    for i, k in enumerate([1, 3, 5, 7, 9, 11]):
        KNN.n_neighbors = k
        print(f"k = {k}")
        for j, chip in enumerate(KNN.predict(un_chips)):
            print(f"  chip{j+1}: [{un_chips[j][0]}, {un_chips[j][1]}] => {'OK' if chip == 1 else 'Fail'}")


  
def mainMenu():
    while True:
        print("1. To plot the original microchip data using different markers for the two classes OK and Fail ")
        print("2. To predict whether three unknown microchips are likely to be OK or Fail ")
        print("3. To show decision boundary and training error")
        print("4. To exit")
        inp = int(input("Enter the number of option : "))
        if (inp == 1):
            plot2()
        elif (inp == 2):
            status()
        elif (inp==3):
            decision_boundary()
        elif (inp == 4):
            False;
            break        
        else:
            mainMenu()
            

mainMenu()