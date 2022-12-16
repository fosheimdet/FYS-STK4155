import numpy as np

import seaborn as sns
import time
from scipy import signal
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

from neural_network import CNN
from dense import Dense
from convolutional import Conv2D
from flatten import Flatten
from max_pool import MaxPool
from functions import scale_data, accuracy, to_categorical_numpy, cross_corr
from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL
from finite_diff import finite_diff



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

# np.random.seed(0)




dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
X,y = preprocess(dataset,images=False,biases=False)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train, X_test = scale_data(X_train,X_test)

##============Hyperparameters=====================
epochs = 100
M = 20
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


# padding = "full"
# model = CNN(X.shape)
# model.addLayer(Conv2D((3,3),sigmoidL,padding))
# model.addLayer(Flatten())
# model.addLayer(Dense(10,softmaxL))
# #model.addLayer(Flatten())
#
# model.initialize_network()
#
# t_start = time.time()
# model.train(X_train,y_train,hyperparams)
# t_end = time.time()
# print("Training time: ", t_end-t_start)
#
#
# output_train, output_test = model.predict(X_train),model.predict(X_test)
#
# acc_train=accuracy(output_train,y_train)
# acc_test=accuracy(output_test,y_test)
# print("acc_train: ", acc_train)
# print("acc_test: ", acc_test)
#
#
#
# model.summary()


#
# # plt.imshow(m_image)
# plt.show()

# m_image = img.imread("nature.png")
# plt.subplot(2, 2,1)
# plt.axis('off')
# plt.imshow(m_image)
# plt.subplot(2, 2,2)
# plt.axis('off')
# plt.imshow(m_image[:,:,0])
# plt.subplot(2, 2,3)
# plt.axis('off')
# plt.imshow(m_image[:,:,1])
# plt.subplot(2, 2,4)
# plt.axis('off')
# plt.imshow(m_image[:,:,2])
# #plt.imshow(m_image, cmap=plt.cm.gray_r, interpolation='nearest')
# # plt.title("Label: %d" % digits.target[random_indices[i]])
# plt.show()


# fig = plt.figure(figsize=(2, 2))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()

##==================================================


input = X[0:2]
print("input: ")
print(input)
targets = y[0:2]
# model = CNN(X.shape)
model = CNN(input.shape)
model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( MaxPool((2,2),2,"same") )
# model.addLayer( Conv2D((3,3),sigmoidL,"same") )
# model.addLayer( MaxPool((2,2),2,"same") )
model.addLayer( Flatten() )
model.addLayer( Dense(10,softmaxL) )
#model.addLayer(Flatten())
model.initialize_network()
# output = model.predict(X[0:20])
t_start = time.time()
output = model.predict(input)
t_end = time.time()
print("Forward prop. time: ", t_end-t_start)

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

##===================================================================================
#                               Model selection
##===================================================================================

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-6, 0, 7)
# eta_vals = np.logspace(-2, -1, 2)
# lmbd_vals = np.logspace(-2, -1, 2)
acc_train = np.zeros((len(lmbd_vals), len(eta_vals)))
acc_test = np.zeros((len(lmbd_vals), len(eta_vals)))
gridsearchBool = True

model = CNN(X_train.shape)
model.addLayer( Conv2D((3,3),sigmoidL,"same") )
model.addLayer( Flatten() )
model.addLayer( Dense(y.shape[1],softmaxL) )
if(gridsearchBool):
    for i,eta in enumerate(eta_vals):
        for j,lmbd in enumerate(lmbd_vals):
            hyperparams = [epochs,M,eta,lmbd]

            model.initialize_network() #Needed to reset the weights
            model.train(X_train,y_train,hyperparams)

            y_pred = model.predict(X_test)
            y_tilde = model.predict(X_train)

            # theta_opt = logistic(X_train,y_train,epochs,M,eta,lmbd,gamma,False)
            # y_pred = np.floor(softmax(X_test@theta_opt)+0.5)
            # y_tilde = np.floor(softmax(X_train@theta_opt)+0.5)
            acc_train[i,j] = accuracy(y_tilde,y_train)
            acc_test[i,j] = accuracy(y_pred,y_test)

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_test,xticklabels = lmbd_vals, yticklabels =eta_vals,
             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title("Test accuracy(%) on MNIST data using CNN with SGD\n" +
    f"epochs={epochs}, batch size={M}")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_train,xticklabels = lmbd_vals, yticklabels =eta_vals,
            annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title("Training accuracy(%) on MNIST data using CNN with SGD \n" +
    f"epochs={epochs}, batch size={M}")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.show()
