'''
PROJECT 2
Linus Hoetzel
Erasmus Exchange Student
'''
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

def r2(y_data, y_model):
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def mse(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y, noise):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise*np.random.normal(0,1,x.shape)
    #noise added here with function call FrankeFunction()

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X

t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

def sgdOLS(X):
    b_ols = np.random.randn(21,1)

    for epoch in range(1, n_epochs+1):
        for i in range(m):
            k = np.random.randint(m) #Pick the k-th minibatch at random

            xi = X[k:k+1]
            zi = z[k:k+1]

            #Compute the gradient using the data in minibatch Bk
            gradients = 2.0/n * xi.T @ ((xi @ b_ols) - zi)

            #Compute new suggestion for eta and b
            eta = learning_schedule(epoch*m+i)
            b_ols -= eta*gradients

    return b_ols

def ridgeReg(X):
    b_ridge = np.random.randn(21,1)

    for epoch in range(1, n_epochs+1):
        for i in range(m):
            k = np.random.randint(m) #Pick the k-th minibatch at random

            xi = X[k:k+1]
            zi = z[k:k+1]

            #Compute the gradient using the data in minibatch Bk
            gradients = 2.0/n*xi.T @ (xi @ (b_ridge) - zi) + 2*lmbda*b_ridge

            #Compute new suggestion for eta and b
            eta = learning_schedule(epoch*m+i)
            b_ridge -= eta*gradients

    return b_ridge

##### Set up of data #####
np.random.seed(1214) 

#data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x,y)
Z = FrankeFunction(X, Y, 0.15) # increased to 0.5 for higher bias
x = X.ravel()
y = Y.ravel()
z = Z.ravel()

#degree of polynomial
p = 5
#  The design matrix now as function of a given polynomial
X = create_X(x, y, p)

# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)

##### Exercise a) - Gradient descent #####

n = len(X_train)    #320 datapoints
M = 50              #size of each mini-batche
m = int(n/M)        #number of minibatches
n_epochs = 20       #number of epochs
lmbda  = 0.1        #Ridge parameter lambda

##### plain SGD for OLS #####
'''
Hyperparameters to tune:
    - learning rate (eta)
    - number and size of mini-batches
    - number of epochs
'''
# With optimal b from SDG for OLS --> sgdOLS
# Make the predictions for OLS regression and SGD
print("Gradient descent with OLS:")
# train set
ytildeOLS = X_train @ sgdOLS(X_train)
print("Training MSE for OLS:", str(mse(y_train, ytildeOLS)))
# test set
ypredictOLS = X_test @ sgdOLS(X_train)
print("Test MSE for OLS:", str(mse(y_test, ypredictOLS)))

##### Ridge regression #####
'''
Hyperparameters to tune:
    - learning rate (eta)
    - lambda
'''
# With optimal b from SDG for Ridge --> ridgeReg
# Make the predictions for Ridge regression and SGD:
print()
print("Gradient descent with Ridge Regression:")
# train set
ytilderidge2 = X_train @ ridgeReg(X_train)
print("Training MSE for Ridge:", str(mse(y_train, ytilderidge2)))
# test set
ypredictridge2 = X_test @ ridgeReg(X_train)
print("Test MSE for Ridge:", str(mse(y_test, ypredictridge2)))
print()

############# algorithm from scikit learn ####################
sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
sgdreg.fit(X,z)
print("SGD from SDG from scikit-learn: ")
# training set
ytildeOLS = X_train @ sgdreg.coef_
print("Training MSE for OLS:", str(mse(y_train, ytildeOLS)))
# test set
ypredictOLS = X_test @ sgdreg.coef_
print("Test MSE for OLS:", str(mse(y_test, ypredictOLS)))
##############################################################