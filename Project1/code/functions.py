import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


def StandardPandascaler(X):
    Xpd = pd.DataFrame(X)
    Xpd_reduced = Xpd.loc[:,Xpd.columns != 0] #Scale all columns except the first
    Xmean = Xpd_reduced.mean()
    X_scaled_red= (Xpd_reduced-Xmean).to_numpy()
    oneCol = X[:,0]
    X_scaled = np.concatenate((oneCol.reshape((len(oneCol),1)),X_scaled_red),axis=1)
    return X_scaled


def OLS(X,z,sklearn):
    if(sklearn):
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(X,z)
        beta_hat = regressor.coef_.T
    else:
        beta_hat = np.linalg.pinv(X.T@X)@X.T@z

    return beta_hat

def ridge(X,z,lmd):
    lamb = pow(10,lmd) #The input values are log values in range e.g. -6 to 0
    n = X.shape[1]
    I_n = np.identity(n)
    print(lamb)
    beta_hat = np.linalg.pinv(X.T@X+lamb*I_n)@X.T@z
    # print('Ridge: \n')
    # print('beta_hat\n', beta_hat)
    return beta_hat

def lasso(X,z,lmd):
    #print(lmd)
    # lmd2 = lmd
    # lmd2 = pow(10,lmd)
    # lmd = lmd2
    print('before',lmd)
    lamb = pow(10,lmd)
    print(lamb)
    #max_iter=1e4 #default max_iter is 1e3
    #tol=0.0001 #default os 0.0001?
    # regLasso = linear_model.Lasso\
    # (lmd,fit_intercept=False,tol = 0.0001,max_iter = 10000)
    regLasso = linear_model.Lasso\
    (lamb,fit_intercept=True,tol = 0.01,max_iter = 1000)
    #
    # regLasso=linear_model.Lasso(alpha=0.015, fit_intercept=False, tol=0.0001,
    #       max_iter=10000, positive=True)
    regLasso.fit(X,z)
    beta_hat = regLasso.coef_
    #print("running lasso")

    #regLasso = linear_model.Lasso(lmd)
    # regLasso=linear_model.Lasso(tol, max_iter,alpha=0.015, fit_intercept=False,
    #      positive=True)
    # _ = regLasso.fit(X,z)
    # z_model = RegLasso.predict(X_train)
    # z_predict = regLasso.predict(X)
    #
    # beta_hat = regLasso.coef_
    #beta_hat = np.linalg.pinv(X)@z_predict

    # print('Lasso: \n')
    # print('beta_hat\n', beta_hat)
    return beta_hat



#Adds noise to the data
def addNoise(z,sigma):
    #print("adding noise, sigma=", sigma)
    n = len(z)
    z_noise = np.zeros(n)
    for i in range(0,n):
        epsilon = np.random.normal(0,sigma)
        z_noise[i]=z[i] + epsilon

    return z_noise

#Calculate mean squared error
def getMSE(z,z_tilde):
    # n = z.shape[0]
    # if(z.shape[1]>z.shape[0]):
    #     n=z.shape[1]
    #n = len(z)
    return np.sum((z-z_tilde)**2)/len(z)

def getR2(z,z_tilde):
    n = np.max(z.shape)
    num = np.sum((z-z_tilde)**2)
    denom = np.sum((z-np.mean(z_tilde))**2)
    return 1-num/denom



#Create the design matrix. Note that the inputs are np.ravel(xx) and np.ravel(yy)
def desMat(xr,yr,p):
    #This if-statement allows for
    #xx and yy to be used as input
    if len(xr.shape)>1:
        xr = np.ravel(xr)
        yr = np.ravel(yr)
    #Number of datapoints:
    N = len(xr)
    #Number of elements in beta:
    numEl = int((p+2)*(p+1)/2)
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



# def getScores2(z_test,f_test,z_train,z_pred,z_tilde):
#     Ez = np.mean(z_pred)
#     bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
#     variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
#
#     cov= np.cov(f_test.reshape(1,-1),z_pred.reshape(1,-1))[0,1]
#     MSEtest = MSE(z_test,z_pred)
#     MSEtrain = MSE(z_train,z_tilde)
#     R2test = R2(z_test,z_pred)
#     R2train = R2(z_train,z_tilde)
#     # R2test = getR2(z_test,z_predict)
#     # R2train = getR2(z_train,z_tilde)
#
#     return bias,variance,cov,MSEtest,MSEtrain,R2test,R2train
#
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def bias(z_test,Ez): return getMSE(z_test, Ez)
def variance(z_predict, Ez): return getMSE(z_predict, Ez)
def cov(f_test, z_predict): return np.cov(f_test.reshape(1,-1),z_predict.reshape(1,-1))[0,1]

def MSEtest(z_test, z_predict): return getMSE(z_test, z_predict)
def MSEtrain(z_train, z_tilde): return getMSE(z_train, z_tilde)
def R2test(z_test, z_predict): return getR2(z_test, z_predict)
def R2train(z_train, z_tilde): return getR2(z_train, z_tilde)

def getScores(scoreNames,z_test,f_test,z_train,z_predict,z_tilde):
    scoreValues = []
    if("bias" in scoreNames):
        Ez = np.mean(z_tilde)
        scoreValues.append(bias(f_test,Ez))

    if("variance" in scoreNames):
        Ez = np.mean(z_tilde)
        scoreValues.append(variance(z_predict,Ez))

    if("cov" in scoreNames):
        #scoreValues.append(cov(f_test, z_predict))
        scoreValues.append(1)

    if("MSEtest" in scoreNames):
        scoreValues.append(MSEtest(z_test, z_predict))

    if("MSEtrain" in scoreNames):
        scoreValues.append(MSEtrain(z_train, z_tilde))

    if("R2test" in scoreNames):
        scoreValues.append(R2test(z_test, z_predict))

    if("R2train" in scoreNames):
        scoreValues.append(R2train(z_train, z_tilde))

    return scoreValues
