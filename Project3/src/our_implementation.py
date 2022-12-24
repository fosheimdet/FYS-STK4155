import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import time
from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import datasets as tf_datasets
from sklearn import datasets as sk_datasets

from neural_network import CNN
from dense import Dense
from convolutional import Conv
from flatten import Flatten
from max_pool import MaxPool
from functions import scale_data, accuracy, to_categorical_numpy, cross_corr
from functions import confusion_matrix, reshape_imgs, partition_data, pick_sample
from activation_functions import noActL, sigmoidL, reluL, leakyReluL, tanhL, softmaxL
from finite_diff import fd_kernel,fd_biases
from reformat_data import format_mnist8x8,format_mnist,format_svhn
from benchmarking import evaluate_fit, gridsearch


def get_data(data_name):
    if(data_name=="MNIST"):
        return format_mnist() #Hand-written digits, 28x28
    elif(data_name=="MNIST8x8"):
        return format_mnist8x8() #Hand-written digits, 8x8
    elif(data_name=="SVHN"):
        return format_svhn() #Housing numbers, 32x32

np.random.seed(0)
##===================================================================================
#                              Data preprocessing
##===================================================================================
data_name="MNIST8x8"
data=(X_train,X_test,y_train,y_test)= get_data(data_name)
#Scale data:
X_train,X_test = X_train/255.0, X_test/255.0

##======================================================================
##========================== Example ===================================
##Example of how to construct a model and perform training/prediction
# print("============= Using example model ================")
# ex_mod = CNN(name="example")
# ex_mod.addLayer( Conv(2,(3,3),sigmoidL,padding="same") )
# ex_mod.addLayer( Flatten() )
# ex_mod.addLayer( Dense(20,sigmoidL) )
# ex_mod.addLayer( Dense(10,softmaxL) )
#
# ex_mod.scheme="Xavier" #Defaults to std.normal unless provided "He" or "Xavier"
# ex_mod.initialize_weights(X_train.shape)
# hyperparams = [100,15,0.001,1e-4] #Epochs, batch_size, eta, lmbd
# ex_mod.set_hyperparams(hyperparams)
#
# ex_mod.train(X_train,y_train)
# y_pred = ex_mod.predict(X_test)
#
# print("test accuracy of example model:", accuracy(y_pred,y_test))
##======================================================================
##======================================================================


##==============================================================================
#                           Train and evaluate
##==============================================================================

#----------------------Import models-----------------------------
#----------------------------------------------------------------
# from model_templates import shallow_dense
# from model_templates import denseNet1_10
# from model_templates import denseNet1_20
from model_templates import denseNet2_20     #Tuned to MNIST8x8

#from model_templates import denseNet3_128
from model_templates import convNet1_3       #Tuned to MNIST8x8
# from model_templates import denseNet3_128
from model_templates import denseNet4_300    #Tuned to MNIST
#-----------------------------------------------------------------

##======Pick model for training/testing================
# model = denseNet4_300
# model = denseNet2_20
model = convNet1_3

model.initialize_weights(X_train.shape)


tune=False
#The hyperparameters of the model are already set to their tuned values
#in model_templates, but can be reset here for further tuning:
hyperparams=[epochs,M,eta,lmbd]=[100,10,0.001,1e-4]
if(tune):
    model.set_hyperparams(hyperparams)
model.scheme=""

model_assessment=True #Accuracy plot from training and confusion matrix
model_selection=False  #Gridsearch over two hyperparameters

# pick (n_test,n_train) random samples for a quick test of the algo
data=pick_sample(data,1000,1000,0)

val_size = 0.1 #Size of validation set

##========================= Perform calculations ===============================
print("Using",model.name,":")
print(model.summary())

# data = (X_train,X_test,y_train,y_test)
t_est = model.estimate_train_time(X_train,y_train,val_size)
print("======================================================================")
print(f"      Estimated training time: {t_est:.3f} s = {t_est//60:.0f}min,{t_est%60:.0f}sec ")
print("======================================================================")


if(model_assessment):
    acc_test, conf_mat = evaluate_fit(model,data,data_name,val_size=0.1)

if(model_selection):
    gridsearch(model,data,data_name)

plt.show()
# #Test backprop. implementation w. finite differences method
# fd_kernel(model,0,X_train[0:10],y_train[0:10])

# plt.show()
# sample = 1
# layer=0
# model.plot_FMs(layer,sample,X_train)
# layer=1
# model.plot_FMs(layer,sample,X_train)
# layer=2
# model.plot_FMs(layer,sample)

# channel = 0
# layer = 0
# model.plot_kernels(layer,channel)
# plt.show()
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
# model.initialize_weights()
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
