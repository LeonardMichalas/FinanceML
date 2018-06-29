#Import the other modules
import _01_preparation as prep
import _02_training as train
import _03_testing as test
import _04_deviation as dev
import _05_plot as plot
import _06_predict as pred

#Import the libraries we need
from sklearn.svm import SVR  #import SVM/Modells
from sklearn.linear_model import SGDRegressor #import Stochastic Gradient Decent regression model
from sklearn import neighbors #import nearest neighbor models
from sklearn.gaussian_process import GaussianProcessRegressor #import Gaussian Process regression model
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

###INITIALIZE THE VARIABLES HERE###
dates = []
prices = []
TrainDates,TrainPrices=[],[]
TestDates,TestPrices=[],[]

#SVR - VARIABLES
SVR_test_predictions = []

#NEURAL NETWORK - VARIABLES
MLP_test_predictions = []

#STOCASTIC GRADIENT DESCENT - VARIABLES
SGD_test_predictions = []

#NEAREST NEIGHBOUR - VARIABLES
NN_test_predictions = []

#GAUSIAN PROZESS - VARIABLES
Gaus_test_predictions = []

#DECISSION TREE - VARIABLES
DT_test_predictions = []

#GRADIENT TREE BOOSTING - VARIABLES
GBRT_test_predictions = []

###INITIALIZE THE MODELLS HERE#####
#SVR - INITIALIZATION

#svr_lin = SVR(kernel = 'linear', C = 1e3)
#svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
#svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1) 
#SVR = SVR(kernel = 'rbf', C = 1.0 , gamma = 0.1, cache_size=200, max_iter=-1) #default
SVR = SVR(kernel = 'linear', C = 1e3, gamma = 0.1, cache_size=200, max_iter=-1)

#NEURAL NETWORK - INITIALIZATION
MLP_reg = MLPRegressor(hidden_layer_sizes=(300,300), activation='relu', solver='lbfgs', alpha=1e-10, learning_rate='constant', max_iter=150, random_state=1) #optimized
#MLP_reg = MLPRegressor(hidden_layer_sizes=(200,200), solver='lbfgs', max_iter=300, learning_rate='adaptive', random_state=1) #default 


#STOCASTIC GRADIENT DESCENT - INITIALIZATION
SGD_reg = SGDRegressor(loss='squared_epsilon_insensitive', learning_rate='constant', alpha=.00001) #please check paramters for optimization: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor


#NEAREST NEIGHBOUR - INITIALIZATION
NN_reg = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')

#GAUSSIAN PROCESS REGRESSION - INITIALIZATION
#create our own kernel based on Matern
#white kernel not added because our data is assumed noiseless
#nu values of 3/2 or 5/2 most common. Specifies smoothness.
#normalize_y=True says the mean is not at 0
#constant kernel shifts mean
#myKernel = ConstantKernel(constant_value=2.5) + Matern(length_scale=7, nu=3/2)
#myRBF = RBF(length_scale=1, length_scale_bounds=(1e-5, 1e5)) #attempt with RBF kernel results in same as Matern
#myRBF2 = 1.0 * RBF(1.0) #should be the same as default kernel used if no parameter is given but results in very different graph
#MaternK = Matern(length_scale=.05, nu=1.5) #usage results in sudden drop and then flat horizontal line
Gaus_reg = GaussianProcessRegressor(kernel=None, alpha=0.1, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=False, random_state=None) #plese check paramters for optimization: http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor

#DECISSION TREE - INITIALIZATION
DT_reg = DecisionTreeRegressor(criterion='friedman_mse', max_depth=14, presort=True)

#GRADIENT TREE BOOSTING - INITIALIZATION
GBRT_reg = GradientBoostingRegressor(loss='huber', n_estimators=400, max_depth=5, criterion='friedman_mse', alpha=0.99)  # verbose, warm_start and presort do not optmize the outcome but time and usability


###READ DATA### 

#If you change the file, also change the name!!
name = 'DAX - Prediction' #der BTC - Prediction
file = 'data/dax.csv' #oder data/BTCEUR.csv' oder data/dax-old.csv
#name = 'BTC - Prediction' #der BTC - Prediction 
#file = 'data/BTCEUR.csv' #oder data/BTCEUR.csv' oder data/dax-old.csv
#name = 'S & P 500 - Prediction'
#file = 'data/sp500.csv'
#name = 'Dow Jones - Prediction'
#file = 'data/dow.csv'

dates, prices = prep.read_data(file, 1)

###NORMALIZE THE DATA###
dates = prep.normalize(dates) 

###SPLIT THE DATA (80/20)###
TrainDates,TrainPrices,TestDates,TestPrices = prep.split_data(dates,prices)

###TRAIN THE MODELLS###

#SVR
SVR = train.training_SVR(SVR, TrainDates, TrainPrices)

#NEURAL NETWORK
MLP_reg = train.training_MLP(MLP_reg, TrainDates, TrainPrices)

#STOCHASTIC GRADIENT DESCENT 
SGD_reg = train.training_SGD(SGD_reg, TrainDates, TrainPrices)

#NEAREST NEIGHBOUR
NN_reg = train.training_NN(NN_reg, TrainDates, TrainPrices)
#KERNEL RIDGE REGRESSION

#GAUSIAN PROZESS 
Gaus_reg = train.training_Gaus(Gaus_reg, TrainDates, TrainPrices)

#DECISSION TREE
DT_reg = train.training_DT(DT_reg, TrainDates, TrainPrices)

#GRADIENT TREE BOOSTING
GBRT_reg = train.training_GBRT(GBRT_reg, TrainDates, TrainPrices)

###TEST THE TRAINED MODELLS###

#SVR
SVR_test_predictions = test.testing_SVR(SVR, TestDates, TestPrices)

#NEURAL NETWORK
MLP_test_predictions = test.testing_MLP(MLP_reg, TestDates, TestPrices)

#STOCASTIC GRADIENT DESCENT 
SGD_test_predictions = test.testing_SGD(SGD_reg, TestDates, TestPrices)

#NEAREST NEIGHBOUR
NN_test_predictions = test.testing_NN(NN_reg, TestDates, TestPrices)

#GAUSIAN PROZESS 
Gaus_test_predictions = test.testing_Gaus(Gaus_reg, TestDates, TestPrices)

#DECISSION TREE
DT_test_predictions = test.testing_DT(DT_reg, TestDates, TestPrices)

#GRADIENT TREE BOOSTING
GBRT_test_predictions = test.testing_GBRT(GBRT_reg, TestDates, TestPrices)


###CALCULATE AVERAGE DEVIATION####
print('Average Deviation:')

#SVR
print('SVR avg:', dev.deviation_avg_single(dates, prices, SVR_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, SVR_test_predictions))

#NEURAL NETWORK
print('MLP avg:', dev.deviation_avg_single(dates, prices, MLP_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, MLP_test_predictions))

#STOCASTIC GRADIENT DESCENT 
print('SGD avg:', dev.deviation_avg_single(dates, prices, SGD_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, SGD_test_predictions))

#NEAREST NEIGHBOUR
print('NN avg:', dev.deviation_avg_single(dates, prices, NN_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, NN_test_predictions))

#GAUSIAN PROZESS 
print('Gaussian Process avg:', dev.deviation_avg_single(dates, prices, Gaus_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, Gaus_test_predictions))

#DECISSION TREE
print('DT avg:', dev.deviation_avg_single(dates, prices, DT_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, DT_test_predictions))

#GRADIENT TREE BOOSTING
print('GBRT avg:', dev.deviation_avg_single(dates, prices, GBRT_test_predictions))
print('Smape and Smdape:', dev.smape_and_smdape(dates, prices, GBRT_test_predictions))

###PLOT THE DATA###
#all Algorithms on one graph. Just pass the model as argument here.

plot.plot(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, DT_reg, GBRT_reg, dates, prices, name)
#plot.single_plot(MLP_reg, dates, prices, name, 'Neural Network') #Plot MLP
#plot.single_plot(SVR, dates, prices, name, 'Support Vectore Regression') #Plot SVR
plot.single_plot(GBRT_reg, dates, prices, name, 'GBRT Regression') #Plot SVR

###MAKE FUTURE PREDICTIONS###
print('Future Predictions:')

pred.predict(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, DT_reg, GBRT_reg, 1.1, 'SVR')

pred.predict(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, DT_reg, GBRT_reg, 1.1, 'SGD')

pred.predict(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, DT_reg, GBRT_reg, 1.1, 'Gaus')

pred.predict(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, DT_reg, GBRT_reg, 1.1, 'MLP')

pred.predict(SVR, SGD_reg, NN_reg, NN_reg, Gaus_reg, DT_reg, GBRT_reg, 1.1, 'NN')

pred.predict(SVR, SGD_reg, NN_reg, NN_reg, Gaus_reg, DT_reg, GBRT_reg, 1.1, 'DT')

pred.predict(SVR, SGD_reg, NN_reg, NN_reg, Gaus_reg, DT_reg, GBRT_reg, 1.1, 'GBRT')
























