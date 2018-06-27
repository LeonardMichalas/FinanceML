import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn import neighbors 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
import threading

#SVR - Training
def training_SVR(SVR, TrainDates, TrainPrices):

    TrainDates = np.reshape(TrainDates,(len(TrainDates), 1))

    SVR = SVR.fit(TrainDates, TrainPrices)

    return SVR

'''
DEPRICATE....
def training_SVR(svr_rbf, svr_lin, svr_poly, TrainDates, TrainPrices):
   
    TrainDates = np.reshape(TrainDates,(len(TrainDates), 1)) #converting to matrix of n X 1 / name swap from traindates to dates

    t1 = threading.Thread(target=svr_rbf.fit, args=(TrainDates, TrainPrices)) #fitting the data points in the models with multi threading
    t2 = threading.Thread(target=svr_lin.fit, args=(TrainDates, TrainPrices))
    t3 = threading.Thread(target=svr_poly.fit, args=(TrainDates, TrainPrices))
    
    #Starts a thread for each modell, so that they get computed simuntainously
    t1.start()
    print('t1 started')
    t2.start()
    print('t2 started')
    t3.start()
    print('t3 started')
 
    #Synchronize the threads
    t1.join()
    print('t1 done...')
    t2.join()
    print('t2 done...')
    t3.join()
    print('t3 done...')

    return svr_rbf, svr_lin, svr_poly
'''

#NEURAL NETWORK - Training
def training_MLP(MLP_reg, TrainDates, TrainPrices):

    TrainDates = np.reshape(TrainDates,(len(TrainDates), 1)) #converting to matrix of n X 1 / name swap from traindates to dates

    MLP_reg = MLP_reg.fit(TrainDates, TrainPrices)

    return MLP_reg

#STOCASTIC GRADIENT DESCENT - Training
def training_SGD(SGD_reg, TrainDates, TrainPrices):
    TrainDates = np.reshape(TrainDates,(len(TrainDates), 1)) #converting to matrix of n X 1 / name swap from traindates to dates

    SGD_reg = SGD_reg.fit(TrainDates, TrainPrices) #fitting the data points in the SGD model

    return SGD_reg

#NEAREST NEIGHBOUR - Training
def training_NN(NN_reg, TrainDates, TrainPrices):
    TrainDates = np.reshape(TrainDates,(len(TrainDates), 1)) #converting to matrix of n X 1 / name swap from traindates to dates

    NN_reg = NN_reg.fit(TrainDates, TrainPrices) #fitting the data points in the SGD model

    return NN_reg

#KERNEL RIDGE REGRESSION - Training

#GAUSIAN PROZESS - Training
def training_Gaus(Gaus_reg, TrainDates, TrainPrices):
    TrainDates = np.reshape(TrainDates,(len(TrainDates), 1)) #converting to matrix of n X 1 / name swap from traindates to dates

    Gaus_reg = Gaus_reg.fit(TrainDates, TrainPrices) #fitting the data points in the SGD model

    return Gaus_reg

# DECISION TREE - Training
def training_DT (DT_reg, TrainDates, TrainPrices):
    TrainDates = np.reshape(TrainDates,
                            (len(TrainDates), 1))  # converting to matrix of n X 1 / name swap from traindates to dates

    DT_reg = DT_reg.fit(TrainDates, TrainPrices)

    return DT_reg

# GRADIENT TREE BOOSTING - Training
def training_GBRT (GBRT_reg, TrainDates, TrainPrices):
    TrainDates = np.reshape(TrainDates,
                            (len(TrainDates), 1))  # converting to matrix of n X 1 / name swap from traindates to dates

    GBRT_reg = GBRT_reg.fit(TrainDates, TrainPrices)

    return GBRT_reg
