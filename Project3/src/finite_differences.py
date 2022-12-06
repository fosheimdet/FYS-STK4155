import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns
import time
from scipy import signal
from sklearn import datasets
from sklearn.model_selection import train_test_split

from neural_network import FFNN
from layers import Dense, Conv2D, Flatten
from functions import scale_data, accuracy, to_categorical_numpy, cross_corr
from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL



#Prepare data
def preprocess(classification_dataset,images=True, biases = False):
    X = classification_dataset.data
    if(images):
        X = classification_dataset.images
    y = classification_dataset.target
    print("X.shape:", X.shape, ", y.shape:", y.shape)
    if(biases):
        n_inputs = X.shape[0]
        n_features = X.shape[1]
        #Add a one column to the design matrix
        one_col = np.ones((n_inputs,1))
        X = np.concatenate((one_col, X), axis = 1 )

    y = to_categorical_numpy(y) #One-hot encode the labels

    return X,y

np.random.seed(0)




dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
X,y = preprocess(dataset,images=False,biases=False)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train, X_test = scale_data(X_train,X_test)

##============Hyperparameters=====================
epochs = 100
M = 1
eta = 0.01
lmbd = 1e-4
gridsearchBool = False

hyperparams = [epochs,M,eta,lmbd]

##===================================================================================
#                               Model assessment
##===================================================================================

dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
X,y = preprocess(dataset,images=True,biases=False)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#X_train, X_test = scale_data(X_train,X_test)
print(X[0].shape)

# dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
# input = dataset.images
# labels = dataset.target
# test = input[0:2]
# print(labels[0])
# print(dataset.images.shape)


# full: full cross-corr
# same: pad such that output has the same dimensions
# valid: no padding
padding = "same"
cnn = FFNN(X.shape)
cnn.addLayer(Conv2D((3,3),sigmoidL,padding))
cnn.addLayer(Conv2D((5,5),sigmoidL,padding))
cnn.addLayer(Flatten())
cnn.addLayer(Dense(10,softmaxL))
#cnn.addLayer(Flatten())

cnn.initialize_network()
#print(cnn.predict(X_train[0:2]).shape)
#
# cnn.train(X_train,y_train,hyperparams)
# cnn.predict(X[0])
# cnn.train(hyperparams)

K0 = cnn.layers[0].K

def cross_entropy(AL,y):
    return -np.sum(y*np.log(AL))


def finite_diff(model,layer_ind,X, y, cost_func):
    K = model.layers[layer_ind].K
    #output = model.predict(test)
    # cost_before = np.sum(output[0])
    print(X[0:2,:,:].shape)

    output = model.predict(X[0:2,:,:])
    # cost_before = -np.sum(y[sample_ind,:]*np.log(output[sample_ind,:]))
    cost_before = -np.sum(y*np.log(output))

    model.backpropagate(y)

    Atilde = np.pad(X,((0,0),(1,1),(1,1)))
    delCdelK = signal.correlate2d(Atilde[0],model.layers[layer_ind].delta[0], mode="valid")
    for n in range(1,X.shape[0]):
        delCdelK+=signal.correlate2d(Atilde[n],model.layers[layer_ind].delta[n], mode="valid")

    print("delCdelK: \n ", delCdelK)
    partial_analytical = delCdelK[0,2]
    print(partial_analytical)

    dw = 1e-10
    delCdelK_num = np.zeros(delCdelK.shape)
    for u in range(delCdelK.shape[0]):
        for v in range(delCdelK.shape[1]):
            K0[u,v]+=dw
            output2 = model.predict(X)
            # cost_after = np.sum(output2[0])
            # cost_after =-np.sum(y[sample_ind,:]*np.log(output2[sample_ind,:]))
            cost_after = -np.sum(y*np.log(output2))
            delCdelK_num[u,v] = (cost_after-cost_before)/dw
            K0[u,v]-=dw

    print(delCdelK_num)

layer_ind = 0
finite_diff(cnn,layer_ind,X_train[0:2],y_train[0:2], cross_entropy)


    # plt.figure()
    # plt.imshow(output[0], cmap='gray', interpolation = 'nearest')
    # plt.show()
# K0[0,2] += dw
# output2 = cnn.predict(test)
# cost_after = np.sum(output2[0])
#
# partial_numerical = (cost_after-cost_before)/dw
# print("analytical: ", partial_analytical)
# print("numerical: ", partial_numerical)

# for l in range(len(cnn.layers)):
#     print("delta ", l)
#     print(cnn.layers[l].delta)

# print("Cost: ", cost)


# print("output shape: ", output.shape)
