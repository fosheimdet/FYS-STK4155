import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from activation_functions import *
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from neural_network import FFNN
from functions import getMSE,getR2, accuracy

@ignore_warnings(category=ConvergenceWarning)
def gridsearch(data,itIndices, hyperparams, regression=True, sklearn=False, cancerBool=True):
    X_train,X_test,y_train,y_test = data

    #Values to iterate over
    epoch_vals= [20,50,80,100,150,200,250,300]      #Epochs
    M_vals = [5,10,20,30,50,80,100,150]             #Batch_sizes

    lengths_h_vals = [[10],[10,10],[10,10,10],[50],[50,50],[50,50,50], [80], [80,80]
    ,[100],[100,100]] #Number of neurons in the hidden layers
    lengths_h_vals = [[10],[20],[50],[100],[50,50],[100,50]]
    # lengths_h_vals = [[10],[10,10],[50],[50,50],[80],[80,80]
    # ,[100],[100,100]]
    activation_vals = ["sigmoid", "relu", "leaky relu","tanh"]     #Activation values

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-6, 0, 7)
    if(regression):
        eta_vals = np.logspace(-5,-1,5)
        lmbd_vals = np.logspace(-5,-1,5)
        eta_vals = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5]
        lmbd_vals = [0.0,1e-5,1e-4,1e-3,1e-2,1e-1]


    # eta_vals = [0.1, 0.01]
    # lmbd_vals = [0.01,0.001]



    valList = [epoch_vals,  M_vals, eta_vals,  lmbd_vals, lengths_h_vals, activation_vals] #List containing values we want to loop over

    iterableStr = ['epochs','batch size','eta','lambda', 'hidden layers', 'activation']
                #       0        1         2       3          4                 5


    iterable1 = valList[itIndices[0]]
    iterable2 = valList[itIndices[1]]

    #Arrays containing the performance scores of the network
    #In case of regression, score1=MSE, score2=R2.
    #In case of classification score1=accuracy, score2=0

    score1 = np.zeros((len(iterable1), len(iterable2)))
    score2= np.zeros((len(iterable1), len(iterable2)))

    score1_sk = np.zeros((len(iterable1), len(iterable2)))
    score2_sk = np.zeros((len(iterable1), len(iterable2)))

    #Find the parameters that gave the best result
    if(regression):
        MSE_test_opt = 1e20
    else:
        acc_test_opt = 0

    it1_opt,it2_opt = None, None

    for i, it1 in enumerate(iterable1):
        for j, it2 in enumerate(iterable2):
            #Set the pertinent elements of hyperparams
            hyperparams[itIndices[0]] = it1
            hyperparams[itIndices[1]] = it2
            epochs,M,eta,lmbd, lengths_h, activation =hyperparams
            #Using our own NN
            if(regression == True):
                MLP = FFNN(X_train, y_train,noActL,derMSE,hyperparams,False)
            else:
                MLP = FFNN(X_train, y_train,softmaxL,derCrossEntropy,hyperparams,True)
            MLP.initializeNetwork()
            MLP.train()

            output_train, output_test = MLP.predict(X_train), MLP.predict(X_test)


            if(regression==True):
                MSE_test = getMSE(y_test, output_test)
                R2_test = getR2(y_test, output_test)
                score1[i,j]= MSE_test
                score2[i,j] = R2_test
                if(MSE_test < MSE_test_opt):
                    it1_opt, it2_opt = it1,it2
                    MSE_test_opt = MSE_test
            else:
                acc_train = accuracy(y_train, output_train)
                acc_test = accuracy(y_test, output_test)
                score1[i,j]= acc_test
                #score2[i,j]=0
                if(acc_test > acc_test_opt):
                    it1_opt, it2_opt = it1,it2
                    acc_test_opt=acc_test


            #Using sklearn's NN
            if(sklearn):
                if(regression):
                    MLP_sklearn = MLPRegressor(hidden_layer_sizes = lengths_h, activation = 'logistic', solver = 'sgd',
                                alpha = lmbd, batch_size = M, learning_rate_init = eta, max_iter = epochs, momentum = 0)
                else:
                    MLP_sklearn = MLPClassifier(hidden_layer_sizes = lengths_h, activation = 'logistic', solver = 'sgd',
                                alpha = lmbd, batch_size = M, learning_rate_init = eta, max_iter = epochs, momentum = 0)

                MLP_sklearn.fit(X_train,y_train)
                output_train_sk, output_test_sk = MLP_sklearn.predict(X_train), MLP_sklearn.predict(X_test)
                if(regression):
                    score1_sk[i,j]=getMSE(output_test_sk, y_test)
                    score2_sk[i,j]=getR2(output_test_sk, y_test)
                else:
                    score1_sk[i,j]=accuracy(output_test_sk, y_test)
                    #score2[i,j]=getR2(output_test_sk, y_test)




##===================================================================================
#                         Set plotting options
##===================================================================================
    print("optimal parameters: ", iterableStr[itIndices[0]] + f"={it1_opt}",\
    iterableStr[itIndices[1]] + f"={it2_opt}")


    #Create list of indices of the hyperparameters not looped over (to be used in title)
    indices_not = [x for x in range(len(iterableStr))]
    indices_not.remove(itIndices[0])
    indices_not.remove(itIndices[1])

    titleStr1,titleStr2 = '','' #Titles for the plots
    cmapStr = 'viridis' #Color map

    dataStr = "Cancer"
    if(not cancerBool): dataStr = "MNIST"

    if(regression):
        titleStr1 = "MSE_test for the Franke function using our NN \n"
        titleStr2 = "R2_test for the Franke function using our NN \n"
        titleStr1_sk = "MSE_train for the Franke function using sklearn \n"
        titleStr2_sk= "MSE_test for the Franke function using sklearn \n"
    else:
        cmapStr = 'rocket'
        titleStr1 = dataStr+" Data Test Accuracy(%) using our NN \n"
        titleStr2 = dataStr+"  Data ?\n"
        titleStr1_sk = dataStr+" Data Test Accuracy(%) using sklearn \n"
        titleStr2_sk = dataStr+" ?  using sklearn \n"
    for i in range(len(indices_not)):
        if(i>0):
            titleStr1+=", "
            titleStr2+=", "
            titleStr1_sk+=", "
            titleStr2_sk+=", "
        titleStr1+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"
        titleStr2+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"
        titleStr1_sk+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"
        titleStr2_sk+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"

    sns.set(font_scale=1.2)
    annot_size = 13
    fmt_str = '.2g'

    if(not regression):
        score1,score2 = 100*score1,100*score2
        score1_sk,score2_sk = 100*score1_sk,100*score2_sk
        fmt_str = '.2f'
##===================================================================================
#                             Plotting
##===================================================================================

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(score1,xticklabels = iterable2, yticklabels = iterable1,
                fmt=fmt_str,annot=True, ax=ax, cmap=cmapStr,annot_kws={"size": annot_size})
    ax.set_title(titleStr1, fontsize = 13)
    ax.set_xlabel(iterableStr[itIndices[1]])
    ax.set_ylabel(iterableStr[itIndices[0]])

    if(regression):
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(score2,xticklabels = iterable2, yticklabels = iterable1 ,
                    fmt=fmt_str,annot=True, ax=ax, cmap=cmapStr,annot_kws={"size": annot_size})
        ax.set_title(titleStr2,fontsize = 13)
        ax.set_xlabel(iterableStr[itIndices[1]])
        ax.set_ylabel(iterableStr[itIndices[0]])

    if(sklearn):
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(score1_sk,xticklabels = iterable2, yticklabels = iterable1,
                    fmt=fmt_str,annot=True, ax=ax, cmap=cmapStr,annot_kws={"size": annot_size})
        ax.set_title(titleStr1_sk, fontsize = 13)
        ax.set_xlabel(iterableStr[itIndices[1]])
        ax.set_ylabel(iterableStr[itIndices[0]])

        if(regression):
            fig, ax = plt.subplots(figsize = (10, 10))
            sns.heatmap(score2_sk,xticklabels = iterable2, yticklabels = iterable1,
                        fmt=fmt_str,annot=True, ax=ax, cmap=cmapStr,annot_kws={"size": annot_size})
            ax.set_title(titleStr2_sk, fontsize = 13)
            ax.set_xlabel(iterableStr[itIndices[1]])
            ax.set_ylabel(iterableStr[itIndices[0]])


    return score1, score2, it1_opt, it2_opt
