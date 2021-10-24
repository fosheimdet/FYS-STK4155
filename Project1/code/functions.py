import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


def OLS(X,z,sklearn):
    if(sklearn):
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(X,z)
        beta_hat = regressor.coef_.T
    else:
        beta_hat = np.linalg.pinv(X.T@X)@X.T@z
    return beta_hat

def ridge(X,z,lmd):
    lamb = lmd
    lamb = pow(10,lmd) #The input are log values in range e.g. -6 to 0
    lmd = lamb
    n = X.shape[1]
    I_n = np.identity(n)
    beta_hat = np.linalg.pinv(X.T@X+lmd*I_n)@X.T@z
    return beta_hat

def lasso(X,z,lmd):
    lamb = pow(10,lmd)
    regLasso = linear_model.Lasso\
    (lamb,fit_intercept=True,tol = 0.01,max_iter = 1000)
    regLasso.fit(X,z)
    beta_hat = regLasso.coef_
    return beta_hat

def getTheScore(scoreName,z_test,f_test,z_train,z_predict,z_tilde):
    scoreVal = 0
    Ez_predict = np.mean(z_predict)
    if(scoreName == "bias"):
        #scoreVal = getMSE(z_test, Ez_predict)
        scoreVal = np.mean((z_tilde-z_train)**2)
    if(scoreName == "variance"):
        scoreVal = getMSE(z_predict, Ez_predict)
    if(scoreName == "MSEtest"):
        scoreVal = getMSE(z_test, z_predict)
    if(scoreName == "MSEtrain"):
        scoreVal = getMSE(z_train, z_tilde)
    if(scoreName == "R2test"):
        scoreVal = getR2(z_test, z_predict)
    if(scoreName == "R2train"):
        scoreVal = getR2(z_train, z_tilde)
    return scoreVal

def getScores(emptyScoreScalars,z_test,f_test,z_train,z_predict,z_tilde):
    for scoreName in emptyScoreScalars:
        emptyScoreScalars[scoreName] = getTheScore(scoreName,z_test,f_test,z_train,z_predict,z_tilde)
    return emptyScoreScalars  #Not empty anymore, but this way we don't have to create a new list, reducing computational cost?

def getR2(z,z_tilde):
    n = len(z)
    num = np.sum((z-z_tilde)**2)
    denom = np.sum((z-np.mean(z_tilde))**2)
    return 1-num/denom

def getMSE(z,z_tilde):
    return np.sum((z-z_tilde)**2)/len(z)

#Adds noise to the data
def addNoise(z,sigma):
    n = len(z)
    z_noise = np.zeros(n)
    for i in range(0,n):
        epsilon = np.random.normal(0,sigma)
        z_noise[i]=z[i] + epsilon
    return z_noise

#Scales our design matrix by subtracting the mean of each column except the first
def StandardPandascaler(X):
    Xpd = pd.DataFrame(X)
    Xpd_reduced = Xpd.loc[:,Xpd.columns != 0] #Scale all columns except the first
    Xmean = Xpd_reduced.mean()
    X_scaled_red= (Xpd_reduced-Xmean).to_numpy()
    oneCol = X[:,0]
    X_scaled = np.concatenate((oneCol.reshape((len(oneCol),1)),X_scaled_red),axis=1)
    return X_scaled

#Create the design matrix.Inputs are np.ravel(xx) and np.ravel(yy)
def desMat(xr,yr,p):
    N = len(xr)
    numEl = int((p+2)*(p+1)/2)#Number of elements in beta
    X = np.ones((N,numEl))
    colInd=0#Column index
    for l in range(1,p+1):
        for k in range(0,l+1):
            X[:,colInd+1] = (xr**(l-k))*yr**k
            colInd = colInd+1
    return X

#Franke's bivariate test function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
