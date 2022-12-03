import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split

from neural_network import FFNN
from layers import Dense
from functions import scale_data, accuracy, to_categorical_numpy
from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL



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

np.random.seed(0)

dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
X,y = preprocess(dataset,biases=False)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train, X_test = scale_data(X_train,X_test)

##============Hyperparameters=====================
epochs = 100
M = 20
eta = 0.01
lmbd = 1e-4
gridsearchBool = True

hyperparams = [epochs,M,eta,lmbd]

##===================================================================================
#                               Model assessment
##===================================================================================
model = FFNN(X_train,y_train,hyperparams)
##--------------------------------------------------------
model.addLayer( Dense(50,sigmoidL) ) #Add input shape as parameter?
# model.addLayer( Dense(5,reluL) )
model.addLayer( Dense(y.shape[1],softmaxL) )
##--------------------------------------------------------
model.initialize_network()

t_start = time.time()
model.train()
t_end = time.time()
print("Training time: ", t_end-t_start)


output_train, output_test = model.predict(X_train), model.predict(X_test)

acc_train=accuracy(output_train,y_train)
acc_test=accuracy(output_test,y_test)
print("acc_train: ", acc_train)
print("acc_test: ", acc_test)


##===================================================================================
#                               Model selection
##===================================================================================

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-6, 0, 7)
# eta_vals = np.logspace(-2, -1, 2)
# lmbd_vals = np.logspace(-2, -1, 2)
acc_train = np.zeros((len(lmbd_vals), len(eta_vals)))
acc_test = np.zeros((len(lmbd_vals), len(eta_vals)))

if(gridsearchBool):
    for i,eta in enumerate(eta_vals):
        for j,lmbd in enumerate(lmbd_vals):
            hyperparams = [epochs,M,eta,lmbd]
            model = FFNN(X_train,y_train,hyperparams)
            model.addLayer( Dense(50,sigmoidL) )
            model.addLayer( Dense(y.shape[1],softmaxL) )
            model.initialize_network()
            model.train()

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