import numpy as np
import matplotlib.pyplot as plt

#Plot all the trained modells here -> ADD ADDITIONAL MODELLS HERE ASWELL
def plot(svr_rbf, svr_lin, svr_poly, dates, prices): 

    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Trading Bot')
    plt.legend()
    plt.show()

    return 
