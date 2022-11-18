import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from gridsearch import gridsearch
from neural_network import FFNN
from functions import scale_data, accuracy, to_categorical_numpy
from activation_functions import noActL, tanhL, sigmoidL,reluL, leakyReluL,softmaxL, derCrossEntropy, derMSE


#Prepare data
def preprocess(classification_dataset, biases = False):
    X = classification_dataset.data
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


np.random.seed(0)        #create same seed for random number every time


##=================== Set hyperparameters ==========================================
iterableStr = ['epochs','batch size','eta','lambda', 'lengths_h', 'activations']
            #       0        1         2       3           4            5
itIndices=[2,3] #Pick which variables to iterate over

epochs = 100
M = 20
eta = 0.01
lmbd = 1e-4
lengths_h = [50]
activation = "sigmoid"

cancerBool = True #Study cancer dataset or MNIST?
sklearnBool = True
regressionBool = False
#Set to true to perform gridsearch. If false, perform model assessment only.
modelSelection = False
##==================================================================================

hyperparams = [epochs,M,eta,lmbd, lengths_h, activation]
#Load chosen dataset
if(cancerBool):
    dataset = load_breast_cancer()   #Download breast cancer dataset
else:
    dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)

##===================================================================================
#                               Model selection
##===================================================================================
if(modelSelection):
    #Preprocess the data for our neural network
    X,y = preprocess(dataset,True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    X_train, X_test = scale_data(X_train,X_test)
    data = [X_train,X_test,y_train,y_test]
    #Perform gridsearch
    acc_train,acc_test, it1_opt, it2_opt = gridsearch(data,itIndices,hyperparams,regressionBool, sklearnBool,cancerBool)

    plt.show()

##===================================================================================
#                               Model asssessment
##===================================================================================
if(not modelSelection):
    nFolds = 5 #Number of folds for cross-validation
    kf = KFold(n_splits = nFolds, random_state=None, shuffle=False)
    X,y = preprocess(dataset,biases=True)

    acc_train,acc_test = 0,0
    acc_train_sk,acc_test_sk=0,0
    for train_inds,test_inds in kf.split(X):
        X_train, y_train= X[train_inds], y[train_inds]
        X_test, y_test = X[test_inds], y[test_inds]

        X_train, X_test = scale_data(X_train,X_test)

        #Using our code
        MLP = FFNN(X_train, y_train,softmaxL,derCrossEntropy,hyperparams,True)
        MLP.initializeNetwork()
        MLP.train()
        output_train, output_test = MLP.predict(X_train), MLP.predict(X_test)

        acc_train+=accuracy(output_train,y_train)
        acc_test+=accuracy(output_test,y_test)

        #Using sklearn
        MLP_sklearn = MLPClassifier(hidden_layer_sizes = lengths_h, activation = 'logistic', solver = 'sgd',
                    alpha = lmbd, batch_size = M, learning_rate_init = eta, max_iter = epochs, momentum = 0)

        MLP_sklearn.fit(X_train,y_train)
        output_train_sk, output_test_sk = MLP_sklearn.predict(X_train), MLP_sklearn.predict(X_test)

        acc_train_sk+=accuracy(output_train_sk,y_train)
        acc_test_sk+=accuracy(output_test_sk,y_test)

    acc_train = acc_train/nFolds
    acc_test = acc_test/nFolds
    acc_train_sk = acc_train_sk/nFolds
    acc_test_sk = acc_test_sk/nFolds
    print("Training accuracy: ", acc_train)
    print("Test accuracy: ", acc_test)
    print("Training accuracy, sklearn: ", acc_train_sk)
    print("Test accuracy, sklearn: ", acc_test_sk)
