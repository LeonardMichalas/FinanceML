import numpy as np
from sklearn.svm import SVR
import threading

#SVR - TESTING
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

#NEURAL NETWORK - TESTING

#STOCASTIC GRADIENT DESCENT - TESTING

#NEAREST NEIGHBOUR - TESTING

#KERNEL RIDGE REGRESSION - TESTING

#GAUSIAN PROZESS - TESTING

#DECISSION TREE - TESTING

#GRADIENT TREE BOOSTING - TESTING
    