import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
import threading

#SVR - TESTING

def testing_SVR(SVR, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel()

    SVR_test_predicitions = []

    SVR_test_prediction = 0

    for date in TestDates:
        SVR_test_prediction = SVR.predict(date)[0]

        SVR_test_predicitions.append([date, SVR_test_prediction])

    return SVR_test_predicitions 

'''
DEPRICATED...
def testing_SVR(svr_rbf, svr_lin, svr_poly, TestDates, TestPrices):  

    TestDates = np.array(TestDates).ravel() #Brings testing data into the right format for the predicition functions

    lin_test_predictions = []
    poly_test_predictions = []
    rbf_test_predicitions = []

    lin_test_prediction = 0
    poly_test_prediction = 0
    rbf_test_predicition = 0

    for date in TestDates:

        lin_test_prediction = svr_lin.predict(date)[0]
        poly_test_prediction = svr_poly.predict(date)[0]
        rbf_test_predicition =  svr_rbf.predict(date)[0]

        lin_test_predictions.append([date, lin_test_prediction])
        poly_test_predictions.append([date, poly_test_prediction])
        rbf_test_predicitions.append([date, rbf_test_predicition])

    return lin_test_predictions, poly_test_predictions, rbf_test_predicitions
'''

#NEURAL NETWORK - TESTING
def testing_MLP(MLP_reg, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel() #Brings testing data into the right format for the predicition functions

    MLP_test_predicitions = []

    MLP_test_prediction = 0

    for date in TestDates:
        MLP_test_prediction = MLP_reg.predict(date)[0]

        MLP_test_predicitions.append([date, MLP_test_prediction])

    return MLP_test_predicitions    

#STOCASTIC GRADIENT DESCENT - TESTING
def testing_SGD(SGD_reg, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel() #Brings testing data into the right format for the predicition functions
    
    SGD_test_predicitions = []

    SGD_test_prediction = 0

    for date in TestDates:

        SGD_test_prediction = SGD_reg.predict(date)[0]

        SGD_test_predicitions.append([date, SGD_test_prediction])

    return SGD_test_predicitions

#NEAREST NEIGHBOUR - TESTING
def testing_NN(NN_reg, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel() #Brings testing data into the right format for the predicition functions
    
    NN_test_predicitions = []

    NN_test_prediction = 0

    for date in TestDates:

        NN_test_prediction = NN_reg.predict(date)[0]

        NN_test_predicitions.append([date, NN_test_prediction])

    return NN_test_predicitions

#GAUSIAN PROZESS - TESTING
def testing_Gaus(Gaus_reg, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel() #Brings testing data into the right format for the predicition functions
    
    Gaus_test_predicitions = []

    Gaus_test_prediction = 0

    for date in TestDates:

        Gaus_test_prediction = Gaus_reg.predict(date)[0]

        Gaus_test_predicitions.append([date, Gaus_test_prediction])

    return Gaus_test_predicitions

#DECISSION TREE - TESTING
def testing_DT(DT_reg, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel()  # Brings testing data into the right format for the predicition functions

    DT_test_predictions = []

    for date in TestDates:

        DT_test_prediction = DT_reg.predict(date)[0]

        DT_test_predictions.append([date, DT_test_prediction])

    return DT_test_predictions


#GRADIENT TREE BOOSTING - TESTING
def testing_GBRT(GBRT_reg, TestDates, TestPrices):
    TestDates = np.array(TestDates).ravel()  # Brings testing data into the right format for the predicition functions

    GBRT_test_predictions = []

    for date in TestDates:
        GBRT_test_prediction = GBRT_reg.predict(date)[0]

        GBRT_test_predictions.append([date, GBRT_test_prediction])

    return GBRT_test_predictions
