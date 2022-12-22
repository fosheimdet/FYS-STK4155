import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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
from tensorflow.keras import datasets, layers, models
from keras.metrics import categorical_crossentropy
from keras.optimizers import SGD

from functions import scale_data, accuracy, to_categorical_numpy, cross_corr, pick_sample
from reformat_data import format_mnist8x8,format_mnist,format_svhn
from benchmarking import plot_accuracy_keras, confusion_matrix_keras


def get_data(dataset,plot=False):
    if(data_name=="MNIST"):
        return format_mnist(plot) #Hand-written digits, 28x28
    elif(data_name=="MNIST8x8"):
        return format_mnist8x8(plot) #Hand-written digits, 8x8
    elif(data_name=="SVHN"):
        return format_svhn(plot) #Housing numbers, 32x32


data_name="MNIST"
dataset = "MNIST8x8"
dataset = "SVHN"
X_train,X_test,y_train,y_test = get_data(data_name,plot=False)
X_train,X_test = X_train/255.0, X_test/255.0 #Normalize pixel values to be between 0 and 1 by dividing by 255.


print(X_train.shape, y_test.shape)




##=========== Hyperparameters ==================
epochs = 10
batch_size = 128

# epochs = 150
# batch_size = 20
eta = 0.01
#==============================================

# model = denseNet1_10_keras
# epochs,batch_size,eta = 80,128,0.01

# from models import denseKeras
from model_templates import denseKeras1_20
from model_templates import denseKeras3_128
from model_templates import convKeras

model = denseKeras1_20
epochs,batch_size,eta = 10,128,0.001
hyperparams = [epochs,batch_size,eta]

#Pick optimizer and cost function
opt =  keras.optimizers.SGD(eta)
model.compile(loss = categorical_crossentropy,
            optimizer = opt,
            metrics = ['accuracy'])


data = (X_train,X_test,y_train,y_test)
X_train,X_test,y_train,y_test = pick_sample(data,1000,500)

model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
                  steps_per_epoch = X_train.shape[0]//batch_size,
                  validation_data = (X_test, y_test),
                  validation_steps = X_test.shape[0]//batch_size, verbose = 1)

# history = model.history
plot_accuracy_keras(model,hyperparams,data_name)
y_pred = model.predict(X_test)
confusion_matrix_keras(model,X_test,y_test,hyperparams,data_name)
plt.show()

# def train_model(model,data, epochs = 20, batch_size = 128):
#     X_train,X_test,y_train,y_test = data
#     # Fitting the model on the training set
#     history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
#                       steps_per_epoch = X_train.shape[0]//batch_size,
#                       validation_data = (X_test, y_test),
#                       validation_steps = X_test.shape[0]//batch_size, verbose = 1)
#     # evaluating the model
#     _, acc = model.evaluate(X_test, y_test, verbose = 1)
#     print('%.3f' % (acc * 100.0))
#     print(history.history.keys())
#     plot_accuracy_keras(history,model)
#
# t_start = time.time()
# train_model(model,data, epochs, batch_size)
# t_train = time.time()-t_start
# print("Training time: ", t_train)
# ##============== Benchmark on test data ===============================
# y_pred = model.predict(X_test)
# acc_test,conf_test = confusion_matrix(y_pred,y_test)
# plt.show()



def gridsearch(model,data,hyperparams,data_name,iterable1_prov=[],iterable2_prov=[]):
    ##===================================================================================
    #                               Model selection
    #===================================================================================
    print("===========================Gridsearch======================================")
    X_train,X_test,y_train,y_test = data
    epochs,M,eta,lmbd = hyperparams

    epoch_vals= [50,80,100,150,200,250]      #Epochs
    M_vals = [10,20,50,80,100]             #Batch_sizes
    epoch_vals= [1,3,5,10]
    M_vals = [1,3,5,10]
    # eta_vals data_name="MNIST"

    # lmbd_vals = np.logspace(-6, 0, 7)
    # eta_vals = np.logspace(-2, -1, 2)
    # lmbd_vals = np.logspace(-2, -1, 2)
    # eta_vals = [0.0005,0.001,0.005,0.01,0.02]
    # eta_vals = [0.005,0.01,0.02,0.1,0.2]
    # lmbd_vals = [0.0001,0.0005,0.001,0.005]
    eta_vals = [0.001,0.01,0.05]
    eta_vals = [0.001]
    eta_vals = [1e-4,5e-4,1e-3,2e-3]
    lmbd_vals =[0,1e-4]

    valList = [epoch_vals,  M_vals, eta_vals,  lmbd_vals] #List containing values we want to loop over
    iterableStr = ['epochs','batch size','eta','lambda']
                #       0        1         2       3
    itIndices=[2,3]
    iterable1 = valList[itIndices[0]]
    iterable2 = valList[itIndices[1]]

    #Use the provided values if they exist
    if(len(iterable1_prov)>0 and len(iterable2_prov)>0):
        iterable1 = iterable1_prov
        iterable2 = iterable2_prov

    acc_test = np.zeros((len(iterable1), len(iterable2)))
    acc_train= np.zeros((len(iterable1), len(iterable2)))

    # X_train, y_train = X_train[0:50], y_train[0:50]

    for i, it1 in enumerate(iterable1):
        for j, it2 in enumerate(iterable2):
            #Set the pertinent elements of hyperparams
            hyperparams[itIndices[0]] = it1
            hyperparams[itIndices[1]] = it2
            epochs,M,eta,lmbd =hyperparams
            hyperparams = [epochs,M,eta,lmbd]

            model.initialize_network() #Needed to reset the weights
            model.train(X_train,y_train,hyperparams,verbose=False)

            y_pred = model.predict(X_test)
            y_tilde = model.predict(X_train)

            acc_train[i,j] = accuracy(y_tilde,y_train)
            acc_test[i,j] = accuracy(y_pred,y_test)

    #Create list of indices of the hyperparameters not looped over (to be used in title)
    indices_not = [x for x in range(len(iterableStr))]
    indices_not.remove(itIndices[0])
    indices_not.remove(itIndices[1])
    titleStr=''
    for i in range(len(indices_not)):
        if(i>0):
            titleStr+=", "
        titleStr+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_test,xticklabels = iterable2, yticklabels =iterable1,
             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title(f"Test accuracy(%) on {data_name} data using '{model.name}' with Sigmoid\n" +
    titleStr)

    ax.set_xlabel(iterableStr[itIndices[1]])
    ax.set_ylabel(iterableStr[itIndices[0]])


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_train,xticklabels = iterable2, yticklabels =iterable1,
            annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title(f"Training accuracy(%) on {data_name} data using '{model.name}' with Sigmoid\n" +
    titleStr)
    ax.set_xlabel(iterableStr[itIndices[1]])
    ax.set_ylabel(iterableStr[itIndices[0]])
    # plt.show()

#
# #                          Train the model
# #==============================================================================
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, epochs=10,
#                     validation_data=(X_test, y_test))
#
#
#
# ##                         Evaluate the model
# ##==============================================================================
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


#
# # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# # plt.figure(figsize=(10,10))
# # for i in range(25):
# #     plt.subplot(5,5,i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_images[i], cmap=plt.cm.binary)
# #     # The CIFAR labels happen to be arrays,
# #     # which is why you need the extra index
# #     plt.xlabel(class_names[train_labels[i][0]])
# # plt.show()
#
# #Create keras CNN
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
#
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
#
# model.summary()





# ##==============================================================================
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# assert x_train.shape == (50000, 32, 32, 3)
# assert x_test.shape == (10000, 32, 32, 3)
# assert y_train.shape == (50000, 1)
# assert y_test.shape == (10000, 1)
# print("cifar x_train.shape: ", x_train.shape)
# x_train = np.moveaxis(x_train,-1,1)
# fig = plt.figure(figsize=(2, 2))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()

##==================================================
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# # # assert x_train.shape == (50000, 32, 32, 3)
# # # assert x_test.shape == (10000, 32, 32, 3)
# # # assert y_train.shape == (50000, 1)
# # # assert y_test.shape == (10000, 1)
# # # print("cifar x_train.shape: ", x_train.shape)
# # assert 1==2
# # x_train = np.moveaxis(x_train,-1,1)
# # x_test = np.moveaxis(x_test,-1,1)
# #
# # print("y0: ",y_train[0])
#
# X_train,X_test,y_train,y_test = process_data((x_train[:10],x_test[:10],y_train[:10],y_test[:10]))
# print(X_train.shape)
# img = X_train.shape[0]
# plt.imshow(img)
# plt.show()
