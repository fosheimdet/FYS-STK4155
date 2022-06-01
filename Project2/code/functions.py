import numpy as np


def shuffle_in_unison(X,z):
    assert len(X) == len(z)
    p = np.random.permutation(len(z))
    return X[p], z[p]


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def addNoise(z,sigma):
    n = len(z)
    z_noise = np.zeros(n)
    for i in range(0,n):
        epsilon = np.random.normal(0,sigma)
        z_noise[i]=z[i] + epsilon
    return z_noise

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


def getR2(z,z_tilde):
    n = len(z)
    num = np.sum((z-z_tilde)**2)
    denom = np.sum((z-np.mean(z))**2)
    return 1-num/denom

def getMSE(z,z_tilde):
    return np.sum((z-z_tilde)**2)/len(z)
