import numpy as np
import matplotlib.pyplot as plt

#Plot all the trained modells here -> ADD ADDITIONAL MODELLS HERE ASWELL
def plot(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, dates, prices, name):
    
    #dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, SVR.predict(dates), color= 'blue', label= 'Support Vektor Regression') #plotting SVR
    plt.plot(dates,SGD_reg.predict(dates), color= 'purple', label= 'Stochastic Gradient Descent') # plotting the line made by Stochastic Gradient Descent algorithm
    plt.plot(dates,Gaus_reg.predict(dates), color= 'orange', label= 'Gaussian Process') # plotting the line made by Gaussian Process algorithm
    plt.plot(dates,NN_reg.predict(dates), color= 'brown', label= 'Nearest Neighbor') # plotting the line made by Nearest Neighbors algorithm
    plt.plot(dates,MLP_reg.predict(dates), color= 'yellow', label= 'MLP Neural Network') # plotting the line made by MLP NN algorithm
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title(name)
    plt.legend()
    plt.show()

    return 

    #Plot single model here. good for testing.
def single_plot(model, dates, prices, name, modelname):
    
    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates,model.predict(dates), color= 'orange', label= modelname) # plotting the line made by algorithm
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title(name)
    plt.legend()
    plt.show()

    return 
