import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from functions import OLS,ridge,lasso,getScores,addNoise,FrankeFunction,\
desMat,getMSE,getR2,StandardPandascaler



def linReg(regMeth,scoreNames,X,z,sigma,scaling,skOLS,lmd):

    #X = desMat(xr,yr,order)

    n = int(np.sqrt(len(z)))    #Number ticks of x and y axes

    z = z
    f = np.zeros(n*n)
    f=z                    #Target function
      #Save original values for plotting
    #z = addNoise(z,sigma)    #Add normally distributed noise w. std sigma
    # z_temp = z
    # z_temp = addNoise(z,sigma)
    # z=z_temp

    # z_noise = np.zeros(n*n)
    # for i in range(0,n*n):
    #     epsilon = np.random.normal(0,sigma)
    #     z_noise[i] = z[i] + epsilon
    #     #print(epsilon)
    # z=z_noise

    z_temp = addNoise(z,sigma)
    z = z_temp

    Z_orig = z.reshape(n,n)
    # print(z)

    z = z.reshape(-1,1)
    f = f.reshape(-1,1)


    #z_noise = 0


    if(scaling):
        X = StandardPandascaler(X) #Scales all columns but the first

    #beta_hat = np.zeros((X.shape[1],1))#Initialize the optimal reg. param. vector

    X_train,X_test,z_train,z_test,f_train,f_test = train_test_split(X,z,f,test_size=0.2)
    #X_train,X_test,z_train,z_test  = train_test_split(X,z,test_size=0.2)

    #f_train,f_test = train_test_split(f,test_size=0.2)
    beta_hat = np.ones(X.shape[1])

    if(regMeth=='OLS'):
        beta_hat = OLS(X_train,z_train,skOLS) #Use ordinary least squares
    if(regMeth=='ridge'):
        beta_hat = ridge(X_train,z_train,lmd)
    if(regMeth=='lasso'):
        beta_hat = lasso(X_train,z_train,lmd)

    #beta_hat = np.linalg.pinv(X_train.T@X_train)@X_train.T@z_train
    z_tilde = X_train@beta_hat
    z_predict = X_test@beta_hat
    z_fitted = X@beta_hat  #Fit all data

    #For plotting the fitted function
    Z_tilde = z_fitted.reshape(n,n)


    scoreValues = getScores(scoreNames,z_test,f_test,z_train,z_predict,z_tilde)

    var_beta = np.diagonal(np.linalg.pinv(X.T@X)*sigma**2) #Should you use X or X_train/X_test?

    z = 0
    return [scoreValues,Z_orig,Z_tilde,beta_hat,var_beta]

    # return [bias,variance,cov,MSEtest,MSEtrain,R2test,R2train,Z_orig,
    # Z_tilde,beta_hat,var_beta]
