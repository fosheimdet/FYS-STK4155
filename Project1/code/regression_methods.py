import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from functions import FrankeFunction,desMat,getMSE,getR2,StandardPandascaler




def OLS(xr,yr,Z,order,sigma,scaled,sklearn):

    X = desMat(xr,yr,order) #Construct design matrix

    n = int(np.sqrt(X.shape[0])) #Number of data points
    #Matrix of same dimension as Z, containing draws from N(0,sigma2)
    #where sigma2 is the variance of epsilon
    epsilon = np.random.normal(0,sigma,(n,n))
    f = np.ravel(Z).reshape(-1,1)
    Z = Z + epsilon
    Z_orig = Z
    if(scaled):
        X = StandardPandascaler(X) #Scales all columns but the first
    z = np.ravel(Z).reshape(-1,1)

    X_train,X_test,z_train,z_test,f_train,f_test = train_test_split(X,z,f,test_size=0.2)
    if(sklearn==False):
        #print('Using ''home made'' OLS algorithm')
        beta_hat = np.linalg.pinv(X_train.T@X_train)@X_train.T@z_train
        z_tilde = X_train@beta_hat
        z_predict = X_test@beta_hat
        #For plotting the fitted function
        z_fitted = X@beta_hat
        Z_tilde = z_fitted.reshape(n,n)
    elif(sklearn==True):
        # print('Using sklearn to perform OLS')
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(X_train,z_train)
        beta_hat = regressor.coef_
        z_tilde = regressor.predict(X_train)
        z_predict =regressor.predict(X_test)
        #For plotting the fitted function
        z_fitted = regressor.predict(X) #All data
        Z_tilde = z_fitted.reshape(n,n)
        # print('z_predict.shape: ', z_predict.shape)
        # print('f_test.shape: ', f_test.shape)

    #Training
    MSE_train= getMSE(z_train,z_tilde)
    R2_train= getR2(z_train,z_tilde)
    #Test
    #MSE_test = getMSE(z_test,z_predict)
    R2_test = getR2(z_test,z_predict)

    var_beta = np.diagonal(np.linalg.pinv(X.T@X)*sigma**2) #Should you use X or X_train/X_test?

    # z_test=z_test.reshape(1,-1)
    # z_predict = z_predict.reshape(1,-1)
    # f_test = f_test.reshape(1,-1)


    # #===========================================================================
    # MSE_test2 = np.mean( np.mean((z_test - z_predict)**2, axis=1, keepdims=True) )
    # bias2 = np.mean( (f_test - np.mean(z_predict, axis=1, keepdims=True))**2 )
    # variance2 = np.mean( np.var(z_predict, axis=1, keepdims=True) )
    # #===========================================================================

    #===========================================================================
    a = 2

    #Testing
    if(a==0):
        #Expected value on all data
        EZ_tilde = np.mean(z_fitted)
        MSE_test = getMSE(z,z_fitted)
        bias =getMSE(f,EZ_tilde)
        variance =getMSE(z_fitted,EZ_tilde)
    #Calculate MSPE on training data
    elif(a==1):
        #Expected value on training data
        EZ_tilde = np.mean(z_tilde)
        MSE_test = getMSE(z_train,z_tilde)
        bias =getMSE(f_train,EZ_tilde)
        variance =getMSE(z_tilde,EZ_tilde)
    #Calculate MSPE on test data
    elif(a==2):
        #Expected value on test data
        EZ_tilde = np.mean(z_predict)
        MSE_test = getMSE(z_test,z_predict)
        bias =getMSE(f_test,EZ_tilde)
        #bias =getMSE(z_test,EZ_tilde)
        variance =getMSE(z_predict,EZ_tilde)
    cov_f_ytilde= np.cov(f_test.reshape(1,-1),z_predict.reshape(1,-1))[0,1]
    #cov_f_ytilde= np.mean(f_test*z_predict) - np.mean(f_test)*np.mean(z_predict)
    # ###########################################################################
    # print('MSE:\t', MSE_test, '\t','\nbias:\t', bias, '\t', \
    # '\nvariance:', variance, '\t' '\nnoise:\t', sigma2,\
    # '\ntotal:\t', bias+variance+sigma2, '\t',  '\nnoise_p: ',\
    # MSE_test-(bias+variance) )
    # ###########################################################################

    return cov_f_ytilde,bias,variance,X,Z_orig,Z_tilde,beta_hat, MSE_test, R2_test, MSE_train, R2_train, var_beta

#
# def OLS(xr,yr,Z,order,scaled,sigma2):
#
#     X = desMat(xr,yr,order) #Construct design matrix
#
#     n = int(np.sqrt(X.shape[0])) #Number of data points
#     #Matrix of same dimension as Z, containing draws from N(0,sigma2)
#     #where sigma2 is the variance of epsilon
#     epsilon = np.random.normal(0,sigma2,(n,n))
#     f = np.ravel(Z).reshape(-1,1)
#     Z = Z + epsilon
#     if(scaled):
#         X = StandardPandascaler(X) #Scales all columns but the first
#     z = np.ravel(Z).reshape(-1,1)
#
#     X_train,X_test,z_train,z_test,f_train,f_test = train_test_split(X,z,f,test_size=0.2)
#     beta_hat = np.linalg.pinv(X_train.T@X_train)@X_train.T@z_train
#     z_tilde = X_train@beta_hat
#     z_predict = X_test@beta_hat
#     #For plotting the fitted function
#     z_fitted = X@beta_hat
#     Z_tilde = z_fitted.reshape(n,n)
#     # Z_orig = z.reshape(n,n)
#     Z_orig = Z
#
#     #Training
#     MSE_train= getMSE(z_train,z_tilde)
#     R2_train= getR2(z_train,z_tilde)
#     #Test
#     #MSE_test = getMSE(z_test,z_predict)
#     R2_test = getR2(z_test,z_predict)
#
#     var_beta = np.diagonal(np.linalg.pinv(X.T@X)*sigma2) #Should you use X or X_train/X_test?
#
#     z_test=z_test.reshape(1,-1)
#     z_predict = z_predict.reshape(1,-1)
#     f_test = f_test.reshape(1,-1)
#
#     # print('z: ', z)
#     # print('z shape: ', z.shape)
#     # print('z_train:', z_train.reshape(1,-1))
#     # print('z_train shape: ', z_train.shape)
#     # print('z_test: ', z_test)
#     # print('z_test shape: ', z_test.shape)
#     # print('z_predict: ', z_predict)
#     # # print('z_predict shape: ', z_predict.shape)
#     # print(np.var(z_predict))
#     # print('hey: ',np.var(z_predict, axis=1, keepdims=True))
#
#     #===========================================================================
#     MSE_test2 = np.mean( np.mean((z_test - z_predict)**2, axis=1, keepdims=True) )
#     bias2 = np.mean( (f_test - np.mean(z_predict, axis=1, keepdims=True))**2 )
#     variance2 = np.mean( np.var(z_predict, axis=1, keepdims=True) )
#     #===========================================================================
#
#     #===========================================================================
#     a = 2
#     if(a==0):
#         #Expected value on all data
#         EZ_tilde = np.mean(z_fitted)
#         MSE_test = getMSE(z,z_fitted)
#         bias =getMSE(f,EZ_tilde)
#         variance =getMSE(z_fitted,EZ_tilde)
#     elif(a==1):
#         #Expected value on training data
#         EZ_tilde = np.mean(z_tilde)
#         MSE_test = getMSE(z_train,z_tilde)
#         bias =getMSE(f_train,EZ_tilde)
#         variance =getMSE(z_tilde,EZ_tilde)
#     elif(a==2):
#         #Expected value on test data
#         EZ_tilde = np.mean(z_predict)
#         MSE_test = getMSE(z_test,z_predict)
#         bias =getMSE(f_test,EZ_tilde)
#         variance =getMSE(z_predict,EZ_tilde)
#
#     #variance2 =np.var(z_predict)
#     #===========================================================================
#     # print('z_test:',z_test)
#     # print(z_predict)
#     # print((z_test-z_predict)**2)
#     # print(np.mean((z_test-z_predict)**2))
#     # print(np.mean((z_test - z_predict)**2, axis=1, keepdims=True))
#     # print(np.sum((z_test-z_predict)**2)/len(z_test))
#     #
#     # print(z_predict)
#     # print(np.var(z_predict))
#     # print(np.var(z_predict,axis=1,keepdims=True))
#     # print(EZ_tilde)
#     # # print(z_train)
#     # # print((z_train-EZ_tilde)**2)
#     # print(np.sum((z_train-EZ_tilde)**2)/(np.maximum(z_train.shape[0],z_train.shape[1])))
#     ############################################################################
#     # print('MSE:\t', MSE_test, '\t', MSE_test2, '\nbias:\t', bias, '\t', bias2, \
#     # '\nvariance:', variance, '\t', variance2, '\nnoise:\t', sigma2,'\t\t\t\t', sigma2,\
#     # '\ntotal:\t', bias+variance+sigma2, '\t', bias2+variance2+sigma2, '\nnoise_p: ',\
#     # MSE_test-(bias+variance), '\t', MSE_test2-(bias2+variance2) )
#     ############################################################################
#     # MSE_test = np.mean((z_test-z_predict)**2)
#     #variance = np.var(z_predict)
#     #
#     # print('Z_predict_transpose: ', z_predict.reshape(1,-1))
#     # #print('var(z_predict): ',np.var(z_predict,axis=1,keepdims=True))
#     # print('var(z_predict): ',np.var(z_predict))
#     # print('var in OLS: ',variance)
#     # EZ_predict = np.mean(z_predict)
#
#     # bias =getMSE(z_test,EZ_predict)
#     # variance = getMSE(z_predict,EZ_predict)
#
#     # bias2 = np.mean((z_test-EZ_tilde)**2)
#     # variance2 = np.mean((z_predict-EZ_tilde)**2)
#     # print('MSE_test_new: ', MSE_test)
#     # print('MSE_test_old: ', getMSE(z_test,z_predict))
#     # print('bias_new: ', bias)
#     # print('bias_old: ', getMSE(f_test,EZ_predict))
#     # print('z_predict: ', z_predict)
#     # print('mean_z_predict: ', np.mean(z_predict,axis=1,keepdims=True))
#
#     return bias,variance,X,Z_orig,Z_tilde,beta_hat, MSE_test, R2_test, MSE_train, R2_train, var_beta
