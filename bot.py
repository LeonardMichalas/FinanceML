#using Python 2.7.14

import csv #allows to reat data from excel sheet
import numpy as np #allows to perform calculation on our data
from sklearn.svm import SVR  #allows us to build a predictive model
import matplotlib.pyplot as plt #allows to plot our data

plt.switch_backend('MacOSX')

dates = []
prices = []

svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models

def read_data(file, i):
    with open(file, 'r') as csvfile:
        csvfileReader = csv.reader(csvfile)
        next(csvfileReader)
        for row in csvfileReader:
            dates.append(int(row[0].split('-')[0])) #appends all dates to the array
            prices.append(float(row[1])) #appends all start prices to the array

            print dates[i] #for testing
            i += 1 #for testing  
    return

def train(dates, prices):
    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1
    svr_rbf.fit(dates, prices) #fitting the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    print('models trained')
    return

def predict(x, modell):
    lin_prediction = svr_lin.predict(x)[0]
    poly_prediction = svr_poly.predict(x)[0]
    rbf_prediction = svr_rbf.predict(x)[0]

    if modell == 'lin':
        return lin_prediction

    if modell == 'poly':    
        return poly_prediction

    if modell == 'rbf':
        return rbf_prediction

    return   

def plot(dates, prices):    
    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1
    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return    

def buy_or_sell(last_price, predicted_price):  

    return  

read_data('fb4g.csv', 0) #file has to be in same dir

train(dates, prices)

prediction_with_lin = predict(29, 'lin')
prediction_with_poly = predict(29, 'poly')
prediction_with_rbf = predict(29, 'rbf')

print('Lin Prediction:', prediction_with_lin)
print('Poly Prediction:', prediction_with_poly)
print('Rbf Prediction:', prediction_with_rbf)

plot(dates, prices)

