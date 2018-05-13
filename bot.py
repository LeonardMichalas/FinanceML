#using Python 2.7.14

import csv #allows to reat data from excel sheet
import numpy as np #allows to perform calculation on our data
from sklearn.svm import SVR  #allows us to build a predictive model
import matplotlib.pyplot as plt #allows to plot our data
import threading

dates = []
prices = []
score = 0

#defining the support vector regression models
svr_lin = SVR(kernel = 'linear', C = 1e3)
svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1) 
svr_sig = SVR(kernel = 'sigmoid')

#function that allows you to save data from a csv file to an array
def read_data(file, i):
    with open(file, 'r') as csvfile:
        csvfileReader = csv.reader(csvfile)
        next(csvfileReader)          
        for row in csvfileReader:
            dates.append(int(i)) #appends all dates to the array
            prices.append(float(row[1])) #appends all start prices to the array

            #print dates[i] #for testing
            i += 1 #for testing  
    return

#function that returns the amount of rows in a given csv file
def get_data_length(file):
    with open(file, 'r') as csvfile:
        csvfileReader = csv.reader(csvfile)
        next(csvfileReader)
        data_length = sum(1 for row in csvfileReader)     
    return data_length 

#function that returns the date of the last entry in your data set
def get_last_date(dates):
    try:
        last_date = dates[0]
    except IndexError:
        last_date = 0
    return last_date

#function that returns the price of the last entry in your data set
def get_last_price(prices):
    try:
        last_price = prices[0]
    except IndexError:
        last_price = 0
    return last_price  

#function that returns the price for a given date
def get_price_at_date(date):
    my_index = dates.index(date)
    return prices[my_index]

#function that splits your data set in to training (80%) and testing (20%) data
def split_data(dates,prices):
    counter1 = 0
    counter2 = 0
    train_size=int(0.80*len(dates))
    print('Volume of train data', train_size)
    TrainDates,TrainPrices=[],[]
    TestDates,TestPrices=[],[]

    for date in dates:
        if counter1<train_size:
            TrainDates.append(date)
            counter1 += 1
        else:
            TestDates.append(date)

    for price in prices:
        if counter2<train_size:
            TrainPrices.append(price)
            counter2 += 1
        else:
            TestPrices.append(price)        
      
    return TrainDates,TrainPrices,TestDates,TestPrices 

#function that trains your modell with the training data
def train(TrainDates, TrainPrices):
    prices = TrainPrices # name swap from trainprices to prices
    dates = np.reshape(TrainDates,(len(TrainDates), 1)) #converting to matrix of n X 1 / name swap from traindates to dates
    #svr_rbf.fit(dates, prices) #fitting the data points in the models without multi threadin
    #svr_lin.fit(dates, prices)
    #svr_poly.fit(dates, prices)
    #svr_sig.fit(dates, prices)

    t1 = threading.Thread(target=svr_rbf.fit, args=(dates, prices)) #fitting the data points in the models with multi threading
    t2 = threading.Thread(target=svr_lin.fit, args=(dates, prices))
    t3 = threading.Thread(target=svr_poly.fit, args=(dates, prices))
    t4 = threading.Thread(target=svr_sig.fit, args=(dates, prices))
    
    t1.start()
    print('t1 started')
    t2.start()
    print('t2 started')
    t3.start()
    print('t3 started')
    t4.start()
    print('t4 started')
    
    t1.join()
    print('t1 done...')
    t2.join()
    print('t2 done...')
    t3.join()
    print('t3 done...')
    t4.join()
    print('t4 done...')
    return

#function that tests your train modell with the test data
def test(TestDates, TestPrices):   
    dates = TestDates #name swap from trainprices to prices

    lin_test_predictions = []
    poly_test_predictions = []
    rbf_test_predicitions = []
    sig_test_predicitions = []

    lin_test_prediction = 0
    poly_test_prediction = 0
    rbf_test_predicition = 0
    sig_test_predicition = 0

    for date in dates:

        lin_test_prediction = svr_lin.predict(date)[0]
        poly_test_prediction = svr_poly.predict(date)[0]
        rbf_test_predicition =  svr_rbf.predict(date)[0]
        sig_test_predicition = svr_sig.predict(date)[0]

        lin_test_predictions.append([date, lin_test_prediction])
        poly_test_predictions.append([date, poly_test_prediction])
        rbf_test_predicitions.append([date, rbf_test_predicition])
        sig_test_predicitions.append([date, sig_test_predicition])

    return lin_test_predictions, poly_test_predictions, rbf_test_predicitions, sig_test_predicitions 

#function that allows you to predict the data of the future, based on a day and the modell you want to use
def predict(x, modell):
    
    if modell == 'lin':
        lin_prediction = svr_lin.predict(x)[0]
        return lin_prediction

    if modell == 'poly':  
        poly_prediction = svr_poly.predict(x)[0]  
        return poly_prediction

    if modell == 'rbf':
        rbf_prediction = svr_rbf.predict(x)[0]
        return rbf_prediction

    if modell == 'sig':
        sig_prediction = svr_sig.predict(x)[0]
        return sig_prediction    

    return   

#function that plots your data to a nice graph
def plot(dates, prices):    
    dates = np.reshape(dates,(len(dates), 1)) #converting to matrix of n X 1
    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.plot(dates,svr_sig.predict(dates), color= 'purple', label= 'Sigmoid model') # plotting the line made by polynomial kernel
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return 

#function to define a score, which tell you wether you should buy or sell
def define_score(prediction_with_rbf, last_price, score):
    if prediction_with_rbf <= last_price:
        score -= 1

    if prediction_with_rbf >= last_price:
        score += 1
    return  score

#function to return the difference of modell prediction and actual for one date
def prediction_difference(date, modell):
    a = predict(date, modell) 
    b = get_price_at_date(date)
    print (a)
    print (b)
    difference = a - b
    return difference

#function to return the average difference between prediction and actual price for all test data
def prediction_difference_avg (test_predictions):
    counter = 0
    difference = 0
    for prediction in test_predictions: 
        actual = get_price_at_date(prediction[0])
        #print (actual, ' counter: ', counter) #for testing purposes
        my_prediction = prediction[1]
        #print(my_prediction, " counter: ", counter) #for testing purposes
        difference = difference + abs(my_prediction - actual) #absolute value used
        #print ("difference: ", difference) #for testing purposes
        counter += 1 #iterate counter
        
    return difference/counter #avg difference


#function Calls    

#fill arrays with data from the data set  
read_data('dax20.csv', 1) #file has to be in same dir
print('Data successfully saved in Arrays')

#get length of the data set  - Not needed at the moment 
#data_length = get_data_length('fb4.csv')
#print('The data set has', data_length, 'entrys')

#get last entry of the data set - Not needed at the moment
#last_date = get_last_date(dates)
#print(last_date)
#last_price = get_last_price(prices)
#print(last_price)

#split the dataset
TrainDates,TrainPrices,TestDates,TestPrices = split_data(dates, prices)
print('TrainDates: ',TrainDates)
print('TestDates: ',TestDates)

#train the model with the train data set
train(TrainDates, TrainPrices)

#test the model with the test data set and print results
lin_test_predictions, poly_test_predictions, rbf_test_predicitions, sig_test_predicitions = test(TestDates, TestPrices)

print('Lin: ',lin_test_predictions)    
print('Poly: ',poly_test_predictions) 
print('Rbf: ',rbf_test_predicitions)
print('Sig: ',sig_test_predicitions)

#make predictions for the future (30th day) with different models
prediction_with_lin = predict(21, 'lin')
print('Lin prediction:', prediction_with_lin)

prediction_with_poly = predict(21, 'poly')
print('Poly prediction:', prediction_with_poly)

prediction_with_rbf = predict(21, 'rbf')
print('Rbf prediction:', prediction_with_rbf)

prediction_with_sig = predict(21, 'sig')
print('Sig prediction:', prediction_with_sig)

'''
print ("price at 0", prices[0])
print ("price at 27", prices[27])
'''
#calcutate the difference between the predicted and the actual data for single data point 
difference_with_lin = prediction_difference(20, 'lin') #input date and modell
print('Lin difference (prediction - acutal): ', difference_with_lin) #the function prints the predicted price and actual

difference_with_poly = prediction_difference(20, 'poly')
print('Poly difference (prediction - acutal):', difference_with_poly)

difference_with_rbf = prediction_difference(20, 'rbf')
print('Rbf difference (prediction - acutal):', difference_with_rbf)

difference_with_sig = prediction_difference(20, 'sig')
print('Sig difference (prediction - acutal):', difference_with_sig)

#calcutate the average difference between the predicted and the actual data for test set for diff modells
print ("average difference with lin", prediction_difference_avg(lin_test_predictions))

print ("average difference with poly", prediction_difference_avg(poly_test_predictions))

print ("average difference with rbf", prediction_difference_avg(rbf_test_predicitions))

print ("average difference with sig", prediction_difference_avg(sig_test_predicitions))

#define a score based on your prediction (very negative score => Sell, very positive score => Buy) - Not needed at the moment but working
#score = define_score(prediction_with_rbf, last_price, score)
#print(score)

#Plot the data
plot(dates, prices)



