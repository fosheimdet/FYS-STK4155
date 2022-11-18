import numpy as np
from sklearn.preprocessing import StandardScaler


#Subtracts the mean of a given column from all elements of that column.
#Doesn't scale column 0, thereby allowing an intercept to be calculated
def scaler(X,X_mean):
    colInd = 1
    #X_train_mean = np.mean(X_train,0) #Mean of columns
    # X[:,colInd:] = X[:,colInd:] - np.mean(X[:,colInd:],0)
    X[:,colInd:] = X[:,colInd:] - X_mean[colInd:]
    # print("After scaling: ", X)
    return X

def scale_data(X_train,X_test,sklearn=True):


    if(sklearn):
        #Scale columns by subtracting their mean and dividing by their std.
        #Done through sklearn
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        scaler.fit(X_test)
        X_test_scaled = scaler.transform(X_test)


    else:
        #Scale columns by subtracting the mean
        n_features = X_train.shape[1]
        X_train_scaled, X_test_scaled = X_train, X_test
        for i in range(n_features):
            mu_train = np.mean(X_train[:,i])
            mu_test = np.mean(X_test[:,i])
            X_train_scaled[:,i] = (X_train[:,i]-mu_train)
            X_test_scaled[:,i] = (X_test[:,i]-mu_test)

    return X_train_scaled, X_test_scaled


    #Use mean and std of the columns of X_train to scale X_test
    # n_features = X_train.shape[1]
    # X_test_scaled = X_test
    # for i in range(n_features):
    #     mu = np.mean(X_train[:,i])
    #     sigma = np.std(X_train[:,i])
    #     X_test_scaled[:,i] = (X_test[:,i]-mu)/sigma

    return X_train_scaled, X_test_scaled


def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

#Calculate accuracy score given one-hot encoded y
def accuracy(y, pred):
    y = np.argmax(y, axis = 1)
    pred = np.argmax(pred, axis = 1)

    return np.sum(y==pred)/len(y)

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


def OLS(X,z,skOLS):
    if(skOLS):
        regressor = LinearRegression(fit_intercept=True)
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

def desMat1D(x,p):
    N = len(x)
    X = np.ones((N,p))
    for i in range(1,p):
        X[:,i] = x**i
    return X




def getR2(z,z_tilde):
    n = len(z)
    num = np.sum((z-z_tilde)**2)
    denom = np.sum((z-np.mean(z))**2)
    return 1-num/denom

def getMSE(z,z_tilde):
    return np.sum((z-z_tilde)**2)/len(z)
