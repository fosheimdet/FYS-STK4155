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
data_name="MNIST"
X_train,X_test,y_train,y_test = get_data(data_name)
#Scale data:
X_train,X_test = X_train/255.0, X_test/255.0


##========================Models=============================

##================Shallow DNN ====================
denseNet1_10 = CNN(X_train.shape,name="denseNet1_10")
denseNet1_10.addLayer( Flatten() )
denseNet1_10.addLayer( Dense(10,reluL) )
denseNet1_10.addLayer( Dense(10,softmaxL) )
denseNet1_10.initialize_network("He")
##=============Dense Neural Network==============
act = sigmoidL
denseNet = CNN(X_train.shape,name="denseNet")
denseNet.addLayer( Flatten() )
# denseNet.addLayer( Dense(128,act) )
# denseNet.addLayer( Dense(128,act) )
# denseNet.addLayer( Dense(64,act) )
denseNet.addLayer( Dense(50,act) )
denseNet.addLayer( Dense(10,softmaxL) )
denseNet.initialize_network("")
##===============================================
##=================conv1=========================
conv = CNN(X_train.shape,name="conv")
conv.addLayer (Conv(3,(3,3),sigmoidL,"same"))
# conv1.addLayer( MaxPool((2,2),stride=2, padding="valid") )
# conv1.addLayer (Conv(6,(3,3),sigmoidL,"same"))
conv.addLayer( Flatten() )
conv.addLayer( Dense(30,sigmoidL) )
conv.addLayer( Dense(128,sigmoidL) )
conv.addLayer( Dense(10,softmaxL) )
conv.initialize_network()
##===============================================
##=================conv3=========================
act=sigmoidL
convNet1_3 = CNN(X_train.shape,name="convNet1_3")
convNet1_3.addLayer (Conv(3,(3,3),act,"valid"))
# convNet1_3.addLayer( MaxPool((2,2),stride=2, padding="valid") )

# convNet1_3.addLayer( Conv(5,(5,5),act,"valid"))
# convNet1_3.addLayer( Conv(5,(5,5),act,"valid"))
# conv1.addLayer (Conv(6,(3,3),sigmoidL,"same"))
convNet1_3.addLayer( Flatten() )
convNet1_3.addLayer( Dense(100,sigmoidL) )
#convNet1_3.addLayer( Dense(128,sigmoidL) )
convNet1_3.addLayer( Dense(10,softmaxL) )
convNet1_3.initialize_network("")
##===============================================
##================ LeNet-5 ======================
act = reluL
leNet = CNN(X_train.shape,name="LeNet-5")
leNet.addLayer( Conv(6,(5,5),act,"custom",p=2) ) #Use a padding of 2 to go from 28x28->32x32
leNet.addLayer( MaxPool((2,2),stride=2, padding="valid") )
leNet.addLayer( Conv(16,(5,5),act,padding="valid") )
leNet.addLayer( MaxPool((2,2),stride=2,padding="valid") )
leNet.addLayer( Conv(120,(5,5),act, "valid") )
leNet.addLayer( Flatten() )
leNet.addLayer( Dense(120,act) )
leNet.addLayer( Dense(86,act) )
leNet.addLayer( Dense(10,softmaxL) )
leNet.initialize_network()

# define LeNet-5
# lenet = nn.Sequential()
# lenet.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2))
# lenet.add_module("tanh1", nn.Tanh())
# lenet.add_module("avg_pool1", nn.AvgPool2d(kernel_size=2, stride=2))
# lenet.add_module("conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1))
# lenet.add_module("avg_pool2", nn.AvgPool2d(kernel_size=2, stride=2))
# lenet.add_module("conv3", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5,stride=1))
# lenet.add_module("tanh2", nn.Tanh())
# lenet.add_module("flatten", nn.Flatten(start_dim=1))
# lenet.add_module("fc1", nn.Linear(in_features=120 , out_features=84))
# lenet.add_module("tanh3", nn.Tanh())
# lenet.add_module("fc2", nn.Linear(in_features=84, out_features=10))
# print(lenet)
##=====================================================
##================Hyperparameters======================

epochs = 10
M = 128

# epochs = 100
# M=10
eta = 0.01
lmbd = 1e-4
# lmbd = 0


hyperparams = [epochs,M,eta,lmbd]
n_train = X_train.shape[0]
n_test = X_test.shape[0]
# n_train,n_test = 100,100
#Partition/split datasets to circumvent memory limitations
n_part_train = 1
n_part_test = 1
write_to_file = True #Write model and its test accuracy to file

model_assessment = False
model_selection = True
val_size = 0.1

##=====================================================
##======Pick model for training/testing================
model = leNet
model = conv
# model = conv3
model = denseNet
epochs,M,eta,lmbd = 10,128,0.01,1e-4 #denseNet
model = convNet1_3
epochs,M,eta,lmbd = 10,128,0.001,1e-4 #convNet1_3

# epochs,M,eta,lmbd = 150,20,0.01,1e-4
# epochs,M,eta,lmbd = 10,128,1e-3,1e-3
# model=denseNet1_10
# epochs,M,eta,lmbd = 150,20,0.01,1e-3

hyperparams = [epochs,M,eta,lmbd]

t_est = model.estimate_train_time(X_train,y_train,hyperparams,val_size)
print("======================================================================")
print(f"      Estimated training time: {t_est:.3f} s = {t_est//60:.0f}min,{t_est%60:.0f}sec ")
print("======================================================================")
print("Using",model.name,":")
# model.initialize_network()
print(model.summary())


model_assessment=True
model_selection=False

data = (X_train,X_test,y_train,y_test)
#data = pick_sample(data,500,100)

if(model_assessment):
    acc_test, conf_mat = evaluate_fit(model,data,hyperparams,data_name,val_size=0.1)

if(model_selection):
    gridsearch(model,data,hyperparams,data_name)

plt.show()
##=====================================================
##=====================================================
# eta_vals = [0.001,0.005,0.01,0.05]
# lmbd_vals = [0]
# def model_assessment(model,data,hyperparams,n_part):
# conf_mat, acc_test = fit_model(model,data,hyperparms,n_part,val_split)
# def fit_model(model,data,hyperparams,n_part,val_split=0.1)
#
# if(model_assessment):
#     ##=====================================================
#     ##==================== Training =======================
#     def partition_dataset(X,y,n_part):
#         return partition_data(X,n_part),partition_data(y,n_part)
#     print("===========================Training======================================")
#
#
#     X_train,y_train = X_train[0:n_train],y_train[0:n_train]
#     X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size)
#
#
#     n_part = 1
#
#     Xp_train,yp_train = partition_dataset(X_train,y_train,n_part_train)
#     acc_vals=[]
#     acc_trains=[]
#     epoch_list=[]
#     for p in range(n_part):
#         print(f"---------------------Training on partition {p}-----------------------------")
#         #These are lists
#         val,train,iters = model.train(Xp_train[p],yp_train[p],hyperparams,X_val,y_val)
#         # val,train,iters = model.train(Xp_train[p],yp_train[p],hyperparams)
#
#         acc_vals+=val
#         acc_trains+=train
#         iters =(np.array(iters)+p*epochs)/n_part #Convert to actual epoch number
#         epoch_list+=iters.tolist()
#
#     acc_vals,acc_trains,epoch_list = np.array(acc_vals),np.array(acc_trains),np.array(epoch_list)
#
#
#     plt.figure()
#     plt.title(f"Running accuracy of our '{model.name}' during training on {dataset}.\n" +
#     f"$\eta={eta}$, $\lambda={lmbd}$, epochs={epochs}, batch size={M}")
#     plt.plot(epoch_list,acc_vals, color = 'darkgreen', label = "Validation accuracy")
#     plt.plot(epoch_list,acc_trains, color = 'mediumblue', label = "Training accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
#     plt.tight_layout()
#     plt.legend()
#
#     print("===========================Testing======================================")
#
#     ##====================================================
#     ##==================== Testing =======================
#     X_test,y_test = X_test[0:n_test],y_test[0:n_test]
#
#
#
#     # n_part_train = 10
#     Xp_test,yp_test = partition_dataset(X_test,y_test,n_part)
#     yp_pred= np.zeros(yp_test.shape)
#     for p in range(n_part):
#         yp_pred[p] = model.predict(Xp_test[p])
#
#     #Reverse partitions
#     y_pred = partition_data(yp_pred,n_part,reverse=True)
#
#
#     # output_test = model.predict(X_test[0:p_size])
#     # output_train = model.predict(X_train[0:p_size])
#
#     # conf_train = confusion_matrix(y_tilde,y_train)
#     conf_test = confusion_matrix(y_pred,y_test)
#
#
#     acc_test = accuracy(y_pred,y_test[0:y_pred.shape[0]])
#
#     if(write_to_file==True):
#         #Print accuracy and model used to file
#         model_summary = model.summary() #String
#         filename = "results/"+model.name+".txt"
#         with open(filename,'a') as f:
#             f.write("********************\n")
#             f.write(f"{dataset}\n")
#             f.write(f"Test accuracy={acc_test}\n")
#             f.write("********************\n")
#             f.write(model_summary+"\n")
#
#
#     n_categories = y_test.shape[1]
#     fig, ax = plt.subplots(figsize = (10, 10))
#     sns.heatmap(100*conf_test,xticklabels = np.arange(0,n_categories), yticklabels =np.arange(0,n_categories),
#              annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
#     ax.set_title(f"Confusion Matrix(%) on {dataset}_test with '{model.name}' using Sigmoid act.\n" +
#     f"$\eta={eta}$, $\lambda={lmbd}$, epochs={epochs}, batch size={M}\n Total accuracy: {100*acc_test:.2f}%")
#     ax.set_xlabel("Prediction",size=13)
#     ax.set_ylabel("Label",size=13)
#
#     # fig, ax = plt.subplots(figsize = (10, 10))
#     # sns.heatmap(100*conf_train,xticklabels = np.arange(0,n_categories), yticklabels =np.arange(0,n_categories),
#     #         annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
#     # ax.set_title("Confusion Matrix(%) on MNIST training data using DNN with SGD \n" +
#     # f"epochs={epochs}, batch size={M}\n Total accuracy: {100*acc_train:.2f}%")
#     # ax.set_xlabel("Prediction")
#     # ax.set_ylabel("label")
#
#     #plt.show()
#
#
#
# if(model_selection):
#     ##===================================================================================
#     #                               Model selection
#     #===================================================================================
#     print("===========================Gridsearch======================================")
#
#     epoch_vals= [50,80,100,150,200,250]      #Epochs
#     M_vals = [10,20,50,80,100]             #Batch_sizes
#     epoch_vals= [1,3,5,10]
#     M_vals = [1,3,5,10]
#     # eta_vals dataset="MNIST"
#
#     # lmbd_vals = np.logspace(-6, 0, 7)
#     # eta_vals = np.logspace(-2, -1, 2)
#     # lmbd_vals = np.logspace(-2, -1, 2)
#     # eta_vals = [0.0005,0.001,0.005,0.01,0.02]
#     # eta_vals = [0.005,0.01,0.02,0.1,0.2]
#     # lmbd_vals = [0.0001,0.0005,0.001,0.005]
#     eta_vals = [0.001,0.01,0.05]
#     lmbd_vals = [0.0001,0.001]
#
#     valList = [epoch_vals,  M_vals, eta_vals,  lmbd_vals] #List containing values we want to loop over
#     iterableStr = ['epochs','batch size','eta','lambda']
#                 #       0        1         2       3
#     itIndices=[2,3]
#     iterable1 = valList[itIndices[0]]
#     iterable2 = valList[itIndices[1]]
#
#     acc_test = np.zeros((len(iterable1), len(iterable2)))
#     acc_train= np.zeros((len(iterable1), len(iterable2)))
#
#     # X_train, y_train = X_train[0:50], y_train[0:50]
#
#     for i, it1 in enumerate(iterable1):
#         for j, it2 in enumerate(iterable2):
#             #Set the pertinent elements of hyperparams
#             hyperparams[itIndices[0]] = it1
#             hyperparams[itIndices[1]] = it2
#             epochs,M,eta,lmbd =hyperparams
#             hyperparams = [epochs,M,eta,lmbd]
#
#             model.initialize_network() #Needed to reset the weights
#             model.train(X_train,y_train,hyperparams,verbose=False)
#
#             y_pred = model.predict(X_test)
#             y_tilde = model.predict(X_train)
#
#             acc_train[i,j] = accuracy(y_tilde,y_train)
#             acc_test[i,j] = accuracy(y_pred,y_test)
#
#     #Create list of indices of the hyperparameters not looped over (to be used in title)
#     indices_not = [x for x in range(len(iterableStr))]
#     indices_not.remove(itIndices[0])
#     indices_not.remove(itIndices[1])
#     titleStr=''
#     for i in range(len(indices_not)):
#         if(i>0):
#             titleStr+=", "
#         titleStr+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"
#
#
#     fig, ax = plt.subplots(figsize = (10, 10))
#     sns.heatmap(100*acc_test,xticklabels = iterable2, yticklabels =iterable1,
#              annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
#     ax.set_title(f"Test accuracy(%) on {dataset} data using '{model.name}' with Sigmoid\n" +
#     titleStr)
#
#     ax.set_xlabel(iterableStr[itIndices[1]])
#     ax.set_ylabel(iterableStr[itIndices[0]])
#
#
#     fig, ax = plt.subplots(figsize = (10, 10))
#     sns.heatmap(100*acc_train,xticklabels = iterable2, yticklabels =iterable1,
#             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
#     ax.set_title(f"Training accuracy(%) on {dataset} data using '{model.name}' with Sigmoid\n" +
#     titleStr)
#     ax.set_xlabel(iterableStr[itIndices[1]])
#     ax.set_ylabel(iterableStr[itIndices[0]])
#     plt.show()
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
