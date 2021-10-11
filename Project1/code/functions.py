import numpy as np
import pandas as pd
from scipy.stats import norm


#Adds noise to the data
def addNoise(z,sigma):
    n = len(z)
    for i in range(0,n):
        z[i]+=np.random.normal(0,sigma)
    return z
#Franke's bivariate test function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

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

#Calculate mean squared error
def getMSE(z,z_tilde):
    # n = z.shape[0]
    # if(z.shape[1]>z.shape[0]):
    #     n=z.shape[1]
    n = len(z)
    return np.sum((z-z_tilde)**2)/n

def getR2(z,z_tilde):
    n = np.max(z.shape)
    num = np.sum((z-z_tilde)**2)
    denom = np.sum((z-np.mean(z_tilde))**2)
    return 1-num/denom

def StandardPandascaler(X):
    Xpd = pd.DataFrame(X)
    Xpd_reduced = Xpd.loc[:,Xpd.columns != 0] #Scale all columns except the first
    Xmean = Xpd_reduced.mean()
    X_scaled_red= (Xpd_reduced-Xmean).to_numpy()
    oneCol = X[:,0]
    X_scaled = np.concatenate((oneCol.reshape((len(oneCol),1)),X_scaled_red),axis=1)
    return X_scaled
