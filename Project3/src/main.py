import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

import seaborn as sns
import time
from scipy import signal
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import keras
import tensorflow as tf
from tensorflow.keras import layers, models

from neural_network import CNN
from dense import Dense
from convolutional import Conv2D
from flatten import Flatten
from max_pool import MaxPool
from functions import scale_data, accuracy, to_categorical_numpy, cross_corr
from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL
from finite_diff import fd_kernel,fd_biases



# #Prepare data
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



#Prepare data
def preprocess_images(X,y):
    #Assumes X.shape = (n_inputs,Height,Width,n_channels) or
    #        datasetsX.shape = (n_inputs,n_channels,Height,Width) <---used by our CNN
    X = X/255.0
    y = to_categorical_numpy(y)
    assert(len(X.shape)>=3)
    if(len(X.shape)==3):
        X = X[:,:,:,np.newaxis]

    return X,y




# dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
# X,y = preprocess(dataset,images=False,biases=False)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# X_train, X_test = scale_data(X_train,X_test)

##============Hyperparameters=====================
epochs = 1
M = 20
eta = 0.01
lmbd = 1e-4

gridsearchBool = False

hyperparams = [epochs,M,eta,lmbd]

##===================================================================================
#                               Model assessment
##===================================================================================

np.random.seed(0)

from tensorflow.keras import layers, models
from tensorflow.keras import datasets as tf_datasets

# # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# # print(train_images.shape, test_images.shape)
# # print(train_labels.shape, test_labels.shape)
# (X_train, y_train), (X_test,y_test) = tf_datasets.cifar10.load_data()
#
# # X_train,y_train = preprocess_images(X_train,y_train)
# targets = to_categorical_numpy(y_train[0:10])

dataset = sk_datasets.load_digits() #Dowload MNIST dataset (8x8 pixels)
X,y = dataset.images,dataset.target
print(X.shape)
X,y = preprocess_images(X,y) #Adds a dimension to X
print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


input = X_train[0:2]
print("input: ")
print(input.shape)
targets = y_train[0:2]
print("X_train shape: ", X_train.shape)
# model = CNN(X.shape)
act = reluL
model = CNN(input.shape)
model.addLayer( Conv2D(32,(3,3),act,"same") )
model.addLayer( Conv2D(32,(3,3),act,"same") )
model.addLayer( MaxPool((2,2),2,"same") )
# model.addLayer( Conv2D(64,(3,3),act,"valid") )
# model.addLayer( Conv2D(64,(3,3),act,"valid") )
# model.addLayer( MaxPool((2,2),2,"same") )
# model.addLayer( Conv2D((3,3),3,sigmoidL,"same") )
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( MaxPool((2,2),2,"same") )
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( MaxPool((2,2),2,"same") )
model.addLayer( Flatten() )
model.addLayer( Dense(10,sigmoidL) )
model.addLayer( Dense(10,softmaxL) )

#model.addLayer(Flatten())
model.initialize_network()
# output = model.predict(X[0:20])
t_start = time.time()
output = model.predict(X_train[0:10])
t_end = time.time()
print("Forward prop. time: ", t_end-t_start)
print("output shape: ", output.shape)

n_samples=10
# model.backpropagate(y[0:n_samples])
# model.update(X[0:10],0.01,0)


fd_kernel(model,0,X[0:n_samples],y[0:n_samples])
fd_biases(model,0,X[0:n_samples],y[0:n_samples])
#
# # finite_diff(model,0,X,y)
#
# #model.train(X_train[0:50],y_train[0:50],hyperparams)
#
#
# y_pred = model.predict(X_test[0:5])
# print(y_pred.shape)
#
# n_categories = y_pred.shape[1]
# print("n_categories:", n_categories)
# conf_matrix = np.zeros((n_categories,n_categories))
#
# for n in range(y_pred.shape[1]):
#     conf_matrix[ np.argmax(y_test[n,:]) , np.argmax(y_pred[n,:]) ]+= 1/y_pred.shape[1]
#
# print(conf_matrix.shape)
# print(conf_matrix)
#
# # sample = 1
# # layer=0
# # model.plot_FMs(layer,sample,X_train)
# # layer=1
# # model.plot_FMs(layer,sample,X_train)
# # layer=2
# # model.plot_FMs(layer,sample)

# channel = 1
# layer = 0
# model.plot_kernels(layer,channel)
# layer=1
# model.plot_kernels(layer,channel)
# layer=3
# model.plot_kernels(layer,channel)
# layer=4
# model.plot_kernels(layer,channel)
# plt.show()

# model = CNN(X.shape)
# model.addLayer( Flatten() )
#
# model.initialize_network()
# X_pred = model.predict(X[0:2,:])
# print("X_pred.shape:",X_pred.shape)
#
# print(model.layers[0].backpropagate(X_pred).shape)

# t_start = time.time()
# model.train(X_train,y_train,hyperparams)
# t_end = time.time()
# print("Backprop. time: ", t_end-t_start)
# y_tilde = model.predict(X_train)
# y_pred = model.predict(X_test)
# acc_train = accuracy(y_train,y_tilde)
# acc_test = accuracy(y_test,y_pred)
#
# print("acc_train: ", acc_train)
# print("acc_test: ", acc_test)

##================================== Keras =========================================


##===================================================================================
#                               Model selection
##===================================================================================
#
# eta_vals = np.logspace(-5, 1, 7)
# lmbd_vals = np.logspace(-6, 0, 7)
# # eta_vals = np.logspace(-2, -1, 2)
# # lmbd_vals = np.logspace(-2, -1, 2)
# acc_train = np.zeros((len(lmbd_vals), len(eta_vals)))
# acc_test = np.zeros((len(lmbd_vals), len(eta_vals)))
# gridsearchBool = False
#
# model = CNN(X_train.shape)
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( Flatten() )
# model.addLayer( Dense(y.shape[1],softmaxL) )
# if(gridsearchBool):
#     for i,eta in enumerate(eta_vals):
#         for j,lmbd in enumerate(lmbd_vals):
#             hyperparams = [epochs,M,eta,lmbd]
#
#             model.initialize_network() #Needed to reset the weights
#             model.train(X_train,y_train,hyperparams)
#
#             y_pred = model.predict(X_test)
#             y_tilde = model.predict(X_train)
#
#             # theta_opt = logistic(X_train,y_train,epochs,M,eta,lmbd,gamma,False)
#             # y_pred = np.floor(softmax(X_test@theta_opt)+0.5)
#             # y_tilde = np.floor(softmax(X_train@theta_opt)+0.5)
#             acc_train[i,j] = accuracy(y_tilde,y_train)
#             acc_test[i,j] = accuracy(y_pred,y_test)
#
#     fig, ax = plt.subplots(figsize = (10, 10))
#     sns.heatmap(100*acc_test,xticklabels = lmbd_vals, yticklabels =eta_vals,
#              annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
#     ax.set_title("Test accuracy(%) on MNIST data using CNN with SGD\n" +
#     f"epochs={epochs}, batch size={M}")
#     ax.set_xlabel("$\lambda$")
#     ax.set_ylabel("$\eta$")
#
#     fig, ax = plt.subplots(figsize = (10, 10))
#     sns.heatmap(100*acc_train,xticklabels = lmbd_vals, yticklabels =eta_vals,
#             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
#     ax.set_title("Training accuracy(%) on MNIST data using CNN with SGD \n" +
#     f"epochs={epochs}, batch size={M}")
#     ax.set_xlabel("$\lambda$")
#     ax.set_ylabel("$\eta$")
#     plt.show()
