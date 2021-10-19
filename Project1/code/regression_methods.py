import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from functions import OLS,ridge,lasso,getScores,addNoise,FrankeFunction,\
desMat,getMSE,getR2,StandardPandascaler


def linReg(regMeth,emptyScoreScalars,X,sigma,lmd,z,scaling,skOLS):

    n = int(np.sqrt(len(z)))    #Number ticks of x and y axes
    f=z   #Target function. In the case of terrain data, these are the same.
    #Add noise:
    z_temp = addNoise(z,sigma)
    z = z_temp
    z_noisy = z

    # z = z.reshape(-1,1)
    # f = f.reshape(-1,1)

    if(scaling):
        X = StandardPandascaler(X) #Scales all columns but the first

    #beta_hat = np.zeros((X.shape[1],1))#Initialize the optimal reg. param. vector

    #X_train,X_test,z_train,z_test,f_train,f_test = train_test_split(X,z,f,test_size=0.2)
    X_train,X_test,z_train,z_test  = train_test_split(X,z,test_size=0.2)

    f_train,f_test = train_test_split(f,test_size=0.2)
    beta_hat = np.ones(X.shape[1])

    if(regMeth=='OLS'):
        beta_hat = OLS(X_train,z_train,skOLS) #Use ordinary least squares
    if(regMeth=='ridge'):
        beta_hat = ridge(X_train,z_train,lmd)
    if(regMeth=='lasso'):
        beta_hat = lasso(X_train,z_train,lmd)

    #beta_hat = np.linalg.pinv(X_train.T@X_train)@X_train.T@z_train
    z_tilde = X_train@beta_hat #Model fitted on training data
    z_predict = X_test@beta_hat #Generates predicted values for the unseen test data set
    z_fitted = X@beta_hat  #Fit all data

    #For plotting the fitted function
    #Z_tilde = z_fitted.reshape(n,n)


    scoreScalars = getScores(emptyScoreScalars,z_test,f_test,z_train,z_predict,z_tilde)

    var_beta = np.diagonal(np.linalg.pinv(X.T@X)*sigma**2) #Should you use X or X_train/X_test?
    #print("scoreValues from linreg: ", scoreValues)
    return [scoreScalars,z_noisy,z_fitted,beta_hat,var_beta]

    # return [bias,variance,cov,MSEtest,MSEtrain,R2test,R2train,Z_orig,
    # Z_tilde,beta_hat,var_beta]
