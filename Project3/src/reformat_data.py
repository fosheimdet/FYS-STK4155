import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from keras.utils import to_categorical #Our numpy version is too slow
from sklearn.model_selection import train_test_split


from sklearn import datasets as sk_datasets
from keras.datasets import mnist
# from keras import datasets as keras_datasets

from functions import to_categorical_numpy

# digits = datasets.load_digits()
#
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

#Takes input in the form of (n_samples,height,width,n_channels)
def plot_images(images,labels):
    n_samples = images.shape[0]
    a = n_samples
    #Make appropriate number of subplots
    n_cols = int(np.ceil(np.sqrt(a)))
    n_rows =int(np.ceil(a/n_cols))

    if(images.shape[-1]<3):
        images = np.squeeze(images,-1)

    fig = plt.figure(figsize=(n_rows, n_cols))
    for n in range(1, n_samples+1):
        fig.add_subplot(n_rows, n_cols, n)
        plt.imshow(images[n-1])
        plt.title(f"label: {np.argmax(labels[n-1])}")
        plt.xticks([])
    # fig.suptitle(f"layer {layer}")
    plt.show()

#Grayscale images(28x28) of hand-written digits.
#Contains 60k training samples, 10k test samples.
#Original source: http://yann.lecun.com/exdb/mnist/
def format_mnist(plot=False):
    print("Using MNIST (28x28)")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train,X_test = X_train[:,:,:,np.newaxis], X_test[:,:,:,np.newaxis]
    y_train,y_test =to_categorical(y_train), to_categorical(y_test)

    if(plot):
        n_images = 30
        indices = np.arange(0,X_train.shape[0])
        selection = np.random.choice(indices,n_images,replace=False)
        plot_images(X_train[selection],y_train[selection])
    return X_train,X_test,y_train,y_test
# format_mnist()

#sklearn's mnist dataset.
#Contains 1797 8x8 images and corresponding labels. No train-test split provided.
def format_mnist8x8(plot=False):
    print("Using MNIST (8x8)")
    dataset = sk_datasets.load_digits() #Dowload MNIST dataset (8x8 pixels)

    X = dataset.images
    print(X.shape)
    y = dataset.target
    X = X[:,:,:,np.newaxis]
    y = to_categorical(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    if(plot):
        n_images = 30
        indices = np.arange(0,X_train.shape[0])
        selection = np.random.choice(indices,n_images,replace=False)
        plot_images(X_train[selection],y_train[selection])
    return X_train,X_test,y_train,y_test

#format_mnist8x8()


#Images (32x32x3) of digits from house numbers taken by google street view images, ranging from 0 to 9.
#Contains 73257 training samples, 26032 samples for testing.
#http://ufldl.stanford.edu/housenumbers/
def format_svhn(plot=False):
    print("Using SVHN (32x32)")
    train = loadmat("train_32x32.mat")
    test = loadmat("test_32x32.mat")

    t_start = time.time()
    #==========================
    X_train,X_test = train["X"], test["X"]
    y_train,y_test = train["y"], test["y"]
    #Samples originally stored in last axis (=3). Move to axis 0.
    X_train,X_test = np.moveaxis(X_train,-1,0), np.moveaxis(X_test,-1,0)

    y_train[y_train==10] = 0 #Zeros labeled as 10 in original data. Undo this
    y_test[y_test==10] = 0
    y_train,y_test = to_categorical(y_train), to_categorical(y_test)
    #==========================
    t_end=time.time()
    # print("Reformatting time: ", t_end-t_start, " s")
    if(plot):
        n_images = 30
        indices = np.arange(0,X_train.shape[0])
        selection = np.random.choice(indices,n_images,replace=False)
        plot_images(X_train[selection],y_train[selection])
    return X_train,X_test,y_train,y_test


# # #Prepare data
# def preprocess(classification_dataset,images=True, biases = False):
#     X = classification_dataset.data
#     if(images):
#         X = classification_dataset.images
#     y = classification_dataset.target
#     print("X.shape:", X.shape, ", y.shape:", y.shape)
#     if(biases):
#         n_inputs = X.shape[0]
#         n_features = X.shape[1]
#         #Add a one column to the design matrix
#         one_col = np.ones((n_inputs,1))
#         X = np.concatenate((one_col, X), axis = 1 )
#
#     y = to_categorical_numpy(y) #One-hot encode the labels
#
#     return X,y
#
#
#
# #Prepare data
# def preprocess_images(X,y):
#     #Assumes X.shape = (n_inputs,Height,Width,n_channels) or
#     #        datasetsX.shape = (n_inputs,n_channels,Height,Width) <---used by our CNN
#     X = X/255.0
#     y = to_categorical_numpy(y)
#     assert(len(X.shape)>=3)
#     if(len(X.shape)==3):
#         X = X[:,:,:,np.newaxis]
#
#     return X,y
#
# def preprocess(classification_dataset, biases = False):
#     X = classification_dataset.data
#     y = classification_dataset.target
#     print("X.shape:", X.shape, ", y.shape:", y.shape)
#     if(biases):
#         n_inputs = X.shape[0]
#         n_features = X.shape[1]
#         #Add a one column to the design matrix
#         one_col = np.ones((n_inputs,1))
#         X = np.concatenate((one_col, X), axis = 1 )
#
#     y = to_categorical_numpy(y) #One-hot encode the labels
#
#     return X,y
