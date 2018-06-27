#FUTURE PREDICTION -> ADD YOUR MODELL HERE
def predict(SVR, SGD_reg, NN_reg, Gaus_reg, MLP_reg, DT_reg, GBRT_reg, x, modell):
    
    if modell == 'SVR':
        SVR_prediction = SVR.predict(x)[0]
        print('SVR:', SVR_prediction)
    
    if modell == 'SGD':
        SGD_prediction = SGD_reg.predict(x)[0]
        print('SGD Future Prediction:', SGD_prediction)

    if modell == 'Gaus':
        Gaus_prediction = Gaus_reg.predict(x)[0]
        print('Gaussian Process Future Prediction:', Gaus_prediction)

    if modell == 'NN':
        NN_prediction = NN_reg.predict(x)[0]
        print('Nearest Neighbor Future Prediction:', NN_prediction)
        
    if modell == 'MLP':
        MLP_prediction = MLP_reg.predict(x)[0]
        print('MLP NN Future Prediction:', MLP_prediction)

    if modell == 'DT':
        DT_prediction = DT_reg.predict(x)[0]
        print('DT Future Prediction:', DT_prediction)

    if modell == 'GBRT':
        GBRT_prediction = GBRT_reg.predict(x)[0]
        print('GBRT Future Prediction:', GBRT_prediction)

    return 

