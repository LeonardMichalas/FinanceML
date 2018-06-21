#FUTURE PREDICTION -> ADD YOUR MODELL HERE
def predict(svr_lin, svr_poly, svr_rbf, SGD_reg, NN_reg, Gaus_reg, x, modell):
    
    if modell == 'lin':
        lin_prediction = svr_lin.predict(x)[0]
        print('Lin Future Prediction:', lin_prediction)

    if modell == 'poly':  
        poly_prediction = svr_poly.predict(x)[0]  
        print('Poly Future Prediction:', poly_prediction)

    if modell == 'rbf':
        rbf_prediction = svr_rbf.predict(x)[0]
        print('Rbf Future Prediction:', rbf_prediction)
    
    if modell == 'SGD':
        SGD_prediction = SGD_reg.predict(x)[0]
        print('SGD Future Prediction:', SGD_prediction)

    if modell == 'Gaus':
        Gaus_prediction = Gaus_reg.predict(x)[0]
        print('Gaussian Process Future Prediction:', Gaus_prediction)

    if modell == 'NN':
        NN_prediction = NN_reg.predict(x)[0]
        print('Nearest Neighbor Future Prediction:', NN_prediction)

    return 

