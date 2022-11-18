import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


from functions import scale_data, accuracy, to_categorical_numpy
from activation_functions import sigm, softmax
from optimizers import gradientDescent,SGD,step_func


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



def logistic(X,y,epochs,M,eta,lmbd,gamma=0,adaptive=False):
    n = X.shape[0] #Number of training samples
    n_params = X.shape[1] #Number of parameters = n_features+1
    m = int(n/M) #Number of mini-batches in an epoch

    theta = np.random.normal(0,1,(n_params,1)) #Initialize the weights and biases

    v=0 #Initial momentum
    rng = np.random.default_rng()
    for e in range(epochs):
        indices = rng.permutation(np.arange(0,n))
        X,y = X[indices], y[indices]
        for i in range(m):
            random_index = M*np.random.randint(m)

            X_i = X[random_index:random_index+M]
            y_i = y[random_index:random_index+M]

            p_i = softmax(X_i@theta)

            grad = X_i.T@(p_i-y_i) + lmbd*theta

            if(adaptive):
                eta = step_func(e*m+i,t0,t1)

            v =gamma*v + eta*grad
            theta = theta -v

    return theta



##===================================================================================
#                               Set hyperparameters
##===================================================================================
# dataset = load_breast_cancer()   #Download breast cancer dataset
#dataset = datasets.load_digits() #Dowload MNIST dataset (8x8 pixelss)

epochs = 100
M = 20          #Batch size
eta = 1e-4
lmbd = 1e-3      #Regularization term
gamma = 0.0     #Momentum term
t0, t1 = 5,50   #For adaptive step
gridsearchBool = False
cancerBool = True

hyperparams=[epochs,M,eta,lmbd,gamma, t0, t1]

#Load chosen dataset
if(cancerBool):
    dataset = load_breast_cancer()   #Download breast cancer dataset
else:
    dataset = datasets.load_digits()  #Dowload MNIST dataset (8x8 pixelss)
##===================================================================================
#                               Model asssessment
##===================================================================================

nFolds = 5
kf = KFold(n_splits = nFolds, random_state=None, shuffle=False)
X,y = preprocess(dataset,True)

acc_train,acc_test = 0,0
acc_test_sk=0
for train_inds,test_inds in kf.split(X):
    X_train = X[train_inds]
    y_train = y[train_inds]

    X_test = X[test_inds]
    y_test = y[test_inds]
    X_train, X_test = scale_data(X_train,X_test)

    #Using our code
    theta_opt = logistic(X_train,y_train,epochs,M,eta,lmbd,gamma,False)
    y_pred = np.floor(softmax(X_test@theta_opt)+0.5)
    y_tilde = np.floor(softmax(X_train@theta_opt)+0.5)
    acc_train+=accuracy(y_tilde,y_train)
    acc_test+=accuracy(y_pred,y_test)
    #Using sklearn
    clf = LogisticRegression(penalty='l2').fit(X_train,np.argmax(y_train,axis=1))
    acc_test_sk+=clf.score(X_test,np.argmax(y_test,axis=1))

acc_train = acc_train/nFolds
acc_test = acc_test/nFolds
acc_test_sk = acc_test_sk/nFolds
print("Training accuracy: ", acc_train)
print("Test accuracy: ", acc_test)
print("Test accuracy, sklearn: ", acc_test_sk)
##===================================================================================


##===================================================================================
#                               Model selection
##===================================================================================

X,y = preprocess(dataset,True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train, X_test = scale_data(X_train,X_test)

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-6, 0, 7)
acc_train = np.zeros((len(lmbd_vals), len(eta_vals)))
acc_test = np.zeros((len(lmbd_vals), len(eta_vals)))

if(gridsearchBool):
    for i,lmbd in enumerate(lmbd_vals):
        for j,eta in enumerate(eta_vals):
            theta_opt = logistic(X_train,y_train,epochs,M,eta,lmbd,gamma,False)
            y_pred = np.floor(softmax(X_test@theta_opt)+0.5)
            y_tilde = np.floor(softmax(X_train@theta_opt)+0.5)
            acc_train[i,j] = accuracy(y_tilde,y_train)
            acc_test[i,j] = accuracy(y_pred,y_test)

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_test,xticklabels = lmbd_vals, yticklabels =eta_vals,
             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title("Test accuracy(%) on cancer data using logistic regression with SGD \n" +
    f"epochs={epochs}, batch size={M}, $\gamma$={gamma}")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_train,xticklabels = lmbd_vals, yticklabels =eta_vals,
            annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title("Training accuracy(%) on cancer data using logistic regression with SGD \n" +
    f"epochs={epochs}, batch size={M}, $\gamma$={gamma}")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.show()
