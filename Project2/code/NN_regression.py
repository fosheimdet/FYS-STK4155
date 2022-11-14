import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

from functions import FrankeFunction, addNoise, getMSE, getR2, desMat
from activation_functions import sigmoidL,noActL,tanhL, reluL,softmaxL, derCrossEntropy, derMSE
from neural_network import FFNN,FFNN1,FFNN2


n = 100
p = 5

x = np.linspace(0,2,n)
#x = 2*np.random.rand(n)
#y = 2*x + 10*np.random.random_sample(len(x)) - 10/2
a0 = 4
a1 = 3
a2 = 6

y = a0 + a1*x + a2*x**2 + np.random.randn(n)

x,y = x.reshape(-1,1),y.reshape(-1,1)


# X_train,X_test, y_train,y_test = train_test_split(x,y,train_size=0.8)

n0 = 1 #Number of input nodes
nhidden = int(input("Please enter the number of hidden layers \n"))
lengths = [n0]
for i in range(nhidden):
    lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
nL = 1         #Number of output nodes
lengths.append(nL)

epochs = 50
M = 15
eta = 0.01
lmbd = 0
hyperparams =[epochs,M,eta,lmbd]

# Using our own NN as model
##===================================================================================
MLP = FFNN(x, y, nhidden, lengths,sigmoidL,noActL,derMSE,False,hyperparams)
MLP.initializeNetwork()
MLP.train()
y_pred = MLP.feedForward(x)
##===================================================================================


# Using sklearn
##===================================================================================
MLP_sklearn = MLPRegressor(hidden_layer_sizes=lengths[1:-1], activation='logistic', solver='sgd',
batch_size=M, learning_rate_init=eta, max_iter=epochs, momentum=0, nesterovs_momentum=False).fit(x.reshape(1,-1),y.reshape(1,-1))
print("hidden_layer_sizes: ", lengths[1:-1])
y_pred_sklearn = MLP_sklearn.predict(x.reshape(1,-1))
##===================================================================================


# Using Keras
##===================================================================================

model = Sequential()
for i in range(len(lengths)-2):
    if(i==0):
        model.add(Dense(units=lengths[i+1], input_dim=1, activation = 'sigmoid'))
    else:
        model.add(Dense(units=lengths[i+1], activation = 'sigmoid'))

model.add(Dense(units=1))

model.compile(loss='mean_squared_error',
              optimizer='sgd')

model.fit(x, y, epochs=100, batch_size=10, verbose=0)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)

y_pred_keras = model.predict(x, batch_size=1)

##===================================================================================



plt.scatter(x,y,c='r')
plt.plot(x,y_pred,'b', label="NN from scratch")
plt.plot(x,np.ravel(y_pred_sklearn),'g', label="sklearn's NN")
plt.plot(x,y_pred_keras, 'm', label = "Keras")
plt.legend()

plt.show()
