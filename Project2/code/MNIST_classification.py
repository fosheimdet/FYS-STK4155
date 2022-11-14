import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

from functions import FrankeFunction, desMat, shuffle_in_unison, addNoise, getMSE, getR2, accuracy, scale_data
from neural_network import FFNN, FFNN2
from activation_functions import noActL, tanhL, sigmoidL,reluL,softmaxL, derCrossEntropy, derMSE



def to_categorical_numpy(integer_vector):
    """
    A function for one hot encoding categorical variables.
    """
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


def transformData(inputs,labels):
    n_inputs = len(inputs)
    X = np.zeros((n_inputs,64))
    for i in range(n_inputs):
        X[i,:] = np.ravel(inputs[i])

    onehotlabels = np.zeros((n_inputs,10))
    for i in range(n_inputs):
        for j in range(10):
            if labels[i]==j:
                onehotlabels[i][j] = 1

    return X, onehotlabels



# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

# inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size = 0.8)
#
#
# print("inputs_train size: ", inputs_train.shape)
# print("labels_train size: ", labels_train.shape)
# print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
# print("labels = (n_inputs) = " + str(labels.shape))
#
#
# X_train, y_train = transformData(inputs_train, labels_train)
# X_test, y_test = transformData(inputs_test, labels_test)

X,y = transformData(inputs,labels)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

X_train,X_test = scale_data(X_train,X_test)

print("X_train size: ", X_train.shape)
print("y_test size: ", y_test.shape)



# n0 = X_train.shape[1] #Number of input nodes
# nhidden = int(input("Please enter the number of hidden layers \n"))
# lengths = [n0]
# for i in range(nhidden):
#     lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
# nL = 10         #Number of output nodes
# lengths.append(nL)

sns.set()




epoch_vals = [10,30,50,80,100,150]
M_vals = [10,20,30,40,50,60,70,100]


epoch_vals= [20,50,80,100,150,200,250,300]
M_vals = [5,10,20,30,50,80,100,150]
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
# eta_vals = [0.1,0.05,0.01,0.005, 0.001, 1e-4]
# lmbd_vals = [1e-5,1e-4,1e-3,1e-2,1e-1,0]



valList = [epoch_vals, M_vals, eta_vals, lmbd_vals] #List containing values we want to loop over

iterableStr = ['epochs','batch size','eta','lambda']
            #       0        1         2       3

itIndices=[0,1] #Pick which variables to iterate over


iterable1 = valList[itIndices[0]]
iterable2 = valList[itIndices[1]]

# iterable1 = [0.1,0.01]
# iterable2 = [0.01,0.001]


epochs = 200
M = 150
eta = 0.1
lmbd = 0.01
lengths_h = [50] #List containing the lengths of the hidden layers
sklearnBool = False
heatmapBool = False #Train a single NN with the given parameters or iterate over two hyperparameters to make a heatmap
hyperparams = [epochs, M, eta, lmbd]

if(not heatmapBool):
    MLP = FFNN(X_train, y_train, lengths_h,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
    MLP.initializeNetwork()
    MLP.train()

    output_train, output_test = MLP.predict(X_train), MLP.predict(X_test)
    acc_train, acc_test = accuracy(output_train, y_train), accuracy(output_test, y_test)
    print(f"Training accuracy: {acc_train}")
    print(f"Test accuracy: {acc_test}")


if(heatmapBool):
    train_accuracies = np.zeros((len(iterable1), len(iterable2)))
    test_accuracies = np.zeros((len(iterable1), len(iterable2)))

    train_accuracies_sk = np.zeros((len(iterable2), len(iterable1)))
    test_accuracies_sk = np.zeros((len(iterable2), len(iterable1)))

    for i, it1 in enumerate(iterable1):
        for j, it2 in enumerate(iterable2):
            hyperparams[itIndices[0]] = it1
            hyperparams[itIndices[1]] = it2
            #Using our own NN
            MLP = FFNN(X_train, y_train, lengths_h,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
            MLP.initializeNetwork()
            MLP.train()

            output_train, output_test = MLP.predict(X_train), MLP.predict(X_test)
            acc_train, acc_test = accuracy(output_train, y_train), accuracy(output_test, y_test)
            train_accuracies[i,j] = acc_train
            test_accuracies[i,j] = acc_test

            if(sklearnBool):
            #Using sklearn's NN
                MLP_sklearn = MLPClassifier(hidden_layer_sizes = lengths_h, activation = 'logistic', solver = 'sgd',
                alpha = lmbd, batch_size = M, learning_rate_init = eta, max_iter = epochs, momentum = 0)
                MLP_sklearn.fit(X_train,y_train)
                output_train_sk, output_test_sk = MLP_sklearn.predict(X_train), MLP_sklearn.predict(X_test)
                acc_train_sk, acc_test_sk = accuracy(output_train_sk, y_train), accuracy(output_test_sk, y_test)
                train_accuracies_sk[i,j] = acc_train_sk
                test_accuracies_sk[i,j] = acc_test_sk



    def plot_data(data, x, y, title):
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(data,xticklabels = x, yticklabels =y , annot=True, ax=ax, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(iterableStr[itIndices[1]])
        ax.set_ylabel(iterableStr[itIndices[0]])

    plot_data(train_accuracies, valList[itIndices[1]], valList[itIndices[0]], f"MNIST training accuracy using our NN, lengths_h={lengths_h}")
    plot_data(test_accuracies, valList[itIndices[1]], valList[itIndices[0]], f"MNIST test accuracy using our NN, lengths_h={lengths_h}")

plt.show()
