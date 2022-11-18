import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from imageio import imread
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

from gridsearch import gridsearch
from functions import FrankeFunction, addNoise, getMSE, getR2, desMat, scale_data,scaler, desMat, desMat1D, OLS, ridge
from activation_functions import sigmoidL,noActL,tanhL, reluL,softmaxL, derCrossEntropy, derMSE
from neural_network import FFNN

def plotsurface(xx,yy,Z, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,Z, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    ax.view_init(20, 65)

np.random.seed(10)  #Ensure we get the same random numbers each time

n = 20   #ticks on x- and y-axis
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
xx,yy = np.meshgrid(x,y)
Z = FrankeFunction(xx,yy)
z = Z.ravel()

sigma = 0.1
z_noisy = addNoise(z,sigma).reshape(-1,1)   #Add gaussian noise of standard deviation sigma to our franke data

xr, yr = np.ravel(xx), np.ravel(yy)

#Construct designmatrix:
X = np.ones((len(xr),2))
X[:,0] = xr
X[:,1] = yr

X_train,X_test, z_train, z_test = train_test_split(X,z_noisy,test_size = 0.2)

##==================== Set hyperparameters =========================================
#List showing hyperparameters which we can iterate over to generate heatmaps
iterableStr = ['epochs','batch size','eta','lambda', 'hidden layers', 'activation']
#                 0          1         2       3            4               5
itIndices=[0,1] #Choose which two hyperparameters to iterate over

#Set hyperparameters
epochs = 100
M = 20
eta = 0.1
lmbd = 1e-4
lengths_h = [50]
activation = "sigmoid"
hyperparams = [epochs,M,eta,lmbd, lengths_h, activation]


regressionBool = True #Regression or classification
sklearnBool = False   #Use sklearn's MLP?
useOptParams = True  #Use optimal parameters from gridsearches in model comparison?

#Set to false to perform gridsearch. If true, perform model assessment only.
modelSelection = False

##===================================================================================
#                              Model Selection
##===================================================================================
if(modelSelection):
    data = [X_train, X_test, z_train, z_test]
    MSE_test,R2_test, it1_opt, it2_opt = gridsearch(data,itIndices,hyperparams,regressionBool,sklearnBool)

#Choose the found optimal parameters from the gridsearches
if(useOptParams):
    if(activation=="sigmoid"):
        epochs, M, eta, lmbd, lengths_h = 300, 5, 0.1, 1e-4,[10]
    elif(activation=="relu"):
        epochs, M, eta, lmbd, lengths_h = 250,30,0.1,1e-5,[10,10]
##===================================================================================
#                              Model Assessment
##===================================================================================
##===================================================================================
#                               Using our NN
##===================================================================================

MLP = FFNN(X_train, z_train,noActL,derMSE,hyperparams,softmaxBool =False)
MLP.initializeNetwork()
MLP.train()

z_pred = MLP.feedForward(X_test)

MSE = getMSE(z_test,z_pred)
print("MSE: ",MSE)
R2 = getR2(z_pred,z_test)
print("R2: ",R2)

ztilde = MLP.predict(X)
Ztilde = ztilde.reshape(n,n)

title = f"Franke data fitted with our NN using {activation} activation \n" + f"MSE_test={MSE:.3f}, R2_test={R2:.3f}"
plotsurface(xx,yy,Ztilde,title)
##===================================================================================
#                                sklearn NN
##===================================================================================
MLP_sk = MLPRegressor(hidden_layer_sizes=lengths_h, activation='logistic', solver='sgd',
alpha=0,batch_size=M, learning_rate_init=eta, max_iter=epochs, momentum=0,
tol = 0, nesterovs_momentum=False).fit(X_train,np.ravel(z_train))

# MLP_sk = MLPRegressor([100,100,100]).fit(X_train,np.ravel(z_train))
z_pred_sk = MLP_sk.predict(X_test)
MSE = getMSE(z_pred_sk, z_test)
R2 = getR2(z_pred_sk, z_test)
print("MSE_sk: ",MSE)
print("R2_sk: ",R2)


ztilde_sk = MLP_sk.predict(X)
Ztilde_sk = ztilde_sk.reshape(n,n)


title = f"Franke data fitted sklearn's NN \n" + f"MSE_test={MSE:.3f}, R2_test={R2:.3f}"
plotsurface(xx,yy,Ztilde_sk,title)

##===================================================================================
#                               Linear regression
##===================================================================================
p = 8
X = desMat(xr,yr,p)
X_train,X_test,z_train,z_test = train_test_split(X,z_noisy,test_size = 0.2)

beta_hat = OLS(X_train,z_train,False)

z_tilde = X_train@beta_hat
z_predict = X_test@beta_hat      #Generates predicted values for the unseen test data set

MSE = getMSE(z_predict, z_test)
R2 = getR2(z_predict, z_test)
print("MSE_linreg: ",MSE)
print("R2_linreg: ",R2)

z_fitted = X@beta_hat            #Model fitted on training data
Z_fitted =z_fitted.reshape(n,n)  #Model fitted on all data (for plotting)


title = f"Franke data fitted using OLS. Pol.Deg.={p} \n" + f"MSE_test={MSE:.3f}, R2_test={R2:.3f}"
plotsurface(xx,yy,Z_fitted,title)

# ##===================================================================================
# #                           Plot Franke function with noise
# ##===================================================================================
title = f"Franke function with $\sigma={sigma}$"
plotsurface(xx,yy, z_noisy.reshape(n,n), title)

plt.show()
