#FUTURE PREDICTION -> ADD YOUR MODELL HERE
def predict(svr_lin, svr_poly, svr_rbf, x, modell):
    
    if modell == 'lin':
        lin_prediction = svr_lin.predict(x)[0]
        print('Lin Future Prediction:', lin_prediction)

    if modell == 'poly':  
        poly_prediction = svr_poly.predict(x)[0]  
        print('Poly Future Prediction:', poly_prediction)

    if modell == 'rbf':
        rbf_prediction = svr_rbf.predict(x)[0]
        print('Rbf Future Prediction:', rbf_prediction)

    return 

