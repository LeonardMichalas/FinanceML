#using Python 2.7.14

import csv #allows to reat data from excel sheet
import numpy as np #allows to perform calculation on our data
from sklearn.svm import SVR  #allows us to build a predictive model
import matplotlib.pyplot as plt #allows to plot our data

#plt.switch_backend('MacOSX')

dates = []
prices = []
score = 0

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

def get_data_length(file):
    with open(file, 'r') as csvfile:
        csvfileReader = csv.reader(csvfile)
        next(csvfileReader)
        data_length = sum(1 for row in csvfileReader)     
    return data_length    

def get_last_date(dates):
    try:
        last_date = dates[0]
    except IndexError:
        last_date = 0
    return last_date

def get_last_price(prices):
    try:
        last_price = prices[0]
    except IndexError:
        last_price = 0
    return last_price    

def train(dates, prices):
    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1
    svr_rbf.fit(dates, prices) #fitting the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
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

def define_score(prediction_with_rbf, last_price, score):
    if prediction_with_rbf <= last_price:
        score -= 1

    if prediction_with_rbf >= last_price:
        score += 1
    return  score

#Fill arrays with data from the data set  
read_data('fb4.csv', 0) #file has to be in same dir
print('Data successfully saved in Arrays')

#Get length of the data set   
data_length = get_data_length('fb4.csv')
print('The dat set has', data_length, 'entrys')

#Get last entry of the data set
last_date = get_last_date(dates)
print(last_date)
last_price = get_last_price(prices)
print(last_price)

#Split the dataset

#Train the model with the filled arrays
train(dates, prices)

#Make predictions with different models
prediction_with_lin = predict(29, 'lin')
print('Lin Prediction:', prediction_with_lin)

prediction_with_poly = predict(29, 'poly')
print('Poly Prediction:', prediction_with_poly)

prediction_with_rbf = predict(29, 'rbf')
print('Rbf Prediction:', prediction_with_rbf)

#Define a score based on your prediction (very negative score => Sell, very positive score => Buy)
score = define_score(prediction_with_rbf, last_price, score)
print(score)

#Plot the data
plot(dates, prices)



