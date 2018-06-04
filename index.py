#Import the other modules
import _01_preparation as prep
import _02_training as train
import _03_testing as test
import _04_deviation as dev
import _05_plot as plot
import _06_predict as pred

#Import the libraries we need
from sklearn.svm import SVR  #import SVM/Modells
import numpy as np
import matplotlib.pyplot as plt

###INITIALIZE THE VARIABLES HERE###
dates = []
prices = []
TrainDates,TrainPrices=[],[]
TestDates,TestPrices=[],[]

#SVR - VARIABLES
lin_test_predictions = []
poly_test_predictions = []
rbf_test_predicitions = []

#NEURAL NETWORK - VARIABLES


#STOCASTIC GRADIENT DESCENT - VARIABLES


#NEAREST NEIGHBOUR - VARIABLES


#KERNEL RIDGE REGRESSION - VARIABLES


#GAUSIAN PROZESS - VARIABLES


#DECISSION TREE - VARIABLES


#GRADIENT TREE BOOSTING - VARIABLES


###INITIALIZE THE MODELLS HERE#####
svr_lin = SVR(kernel = 'linear', C = 1e3)
svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1) 

###READ DATA###
file = 'data/dax.csv'
dates, prices = prep.read_data(file, 1)

###NORMALIZE THE DATA###
dates = prep.normalize(dates) 

###SPLIT THE DATA (80/20)###
TrainDates,TrainPrices,TestDates,TestPrices = prep.split_data(dates,prices)

###TRAIN THE MODELLS###

#SVR
svr_rbf, svr_lin, svr_poly = train.training_SVR(svr_rbf, svr_lin, svr_poly, TrainDates, TrainPrices)

#NEURAL NETWORK

#STOCASTIC GRADIENT DESCENT 

#NEAREST NEIGHBOUR

#KERNEL RIDGE REGRESSION

#GAUSIAN PROZESS 

#DECISSION TREE

#GRADIENT TREE BOOSTING

###TEST THE TRAINED MODELLS###

#SVR
lin_test_predictions, poly_test_predictions, rbf_test_predicitions = test.testing_SVR(svr_rbf, svr_lin, svr_poly, TestDates, TestPrices)

#NEURAL NETWORK

#STOCASTIC GRADIENT DESCENT 

#NEAREST NEIGHBOUR

#KERNEL RIDGE REGRESSION

#GAUSIAN PROZESS 

#DECISSION TREE

#GRADIENT TREE BOOSTING

###CALCULATE AVERAGE DEVIATION####
print('Average Deviation:')
dev.deviation_avg_SVR(dates, prices, lin_test_predictions, poly_test_predictions, rbf_test_predicitions)

###PLOT THE DATA###
plot.plot(svr_rbf, svr_lin, svr_poly, dates, prices)

###MAKE FUTURE PREDICTIONS###
print('Future Predictions:')

pred.predict(svr_lin, svr_poly, svr_rbf, 1.1, 'lin')

pred.predict(svr_lin, svr_poly, svr_rbf, 1.1, 'poly')

pred.predict(svr_lin, svr_poly, svr_rbf, 1.1, 'rbf')


























