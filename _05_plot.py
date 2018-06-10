import numpy as np
import matplotlib.pyplot as plt

#Plot all the trained modells here -> ADD ADDITIONAL MODELLS HERE ASWELL
def plot(svr_rbf, svr_lin, svr_poly, SGD_reg, Gaus_reg, MLP_reg, dates, prices):
    
    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.plot(dates,SGD_reg.predict(dates), color= 'purple', label= 'Stochastic Gradient Descent') # plotting the line made by Stochastic Gradient Descent algorithm
    plt.plot(dates,Gaus_reg.predict(dates), color= 'orange', label= 'Gaussian Process') # plotting the line made by Gaussian Process algorithm
    plt.plot(dates,MLP_reg.predict(dates), color= 'yellow', label= 'MLP Neural Network') # plotting the line made by MLP NN algorithm
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('DAX Prediction')
    plt.legend()
    plt.show()

    return 
