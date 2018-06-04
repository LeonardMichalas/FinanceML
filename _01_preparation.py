#Import libraries
import csv #allows to reat data from excel sheet
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

#Initialize variables
dates = []
prices = []

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
    return dates, prices

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

#Normalizes the data so we can train the modells with high performance
def normalize(dates):
    series = Series(dates)
    values = series.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    
    dates = normalized    

    return dates       


