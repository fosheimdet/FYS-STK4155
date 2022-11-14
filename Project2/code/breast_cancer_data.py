import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

from neural_network import FFNN, FFNN2
from functions import scale_data, accuracy
from activation_functions import noActL, tanhL, sigmoidL,reluL,softmaxL, derCrossEntropy, derMSE

def accuracy_score(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector




"""Load breast cancer dataset"""

np.random.seed(0)        #create same seed for random number every time

cancer=load_breast_cancer()      #Download breast cancer dataset

inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
outputs=cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
labels=cancer.feature_names[0:30]

print('The content of the breast cancer dataset is:')      #Print information about the datasets
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

X=inputs      #Reassign the Feature and Label matrices to other variables
y=outputs



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


X_train, X_test = scale_data(X_train,X_test)




y_train=to_categorical(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test=to_categorical(y_test)


# del temp1,temp2,temp

# Define tunable parameters"


# eta_vals=np.logspace(-3,-1,3)                    #Define vector of learning rates (parameter to SGD optimiser)
# lmbd=0.0                               #Define hyperparameter
# n_layers=2                                  #Define number of hidden layers in the model
# n_neuron=np.logspace(0,3,4,dtype=int)       #Define number of neurons per layer
# n_neuron = [2,10,20,30]
# epochs=100                                   #Number of reiterations over the input data
# M=10                              #Number of samples per gradient update

#Number of neurons in the hidden layers
lengths_h = [50,50]

epoch_vals= [20,50,80,100,150,200,250,300]
M_vals = [5,10,20,30,50,80,100,150]
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

epochs = 100
M = 20
eta = 0.1
lmbd = 1
hyperparams = [epochs,M,eta,lmbd]

sklearnBool = False

valList = [epoch_vals, M_vals, eta_vals, lmbd_vals]  #List containing values we want to loop over

iterableStr = ['epochs','batch size','eta','lambda']
            #       0        1         2       3

itIndices=[2,3] #Pick which variables to iterate over


iterable1 = valList[itIndices[0]]
iterable2 = valList[itIndices[1]]



train_accuracies=np.zeros((len(iterable1),len(iterable2)))      #Define matrices to store accuracy scores as a function
test_accuracies=np.zeros((len(iterable1),len(iterable2)))       #of learning rate and number of hidden neurons for


train_accuracies_sk=np.zeros((len(iterable1),len(iterable2)))
test_accuracies_sk=np.zeros((len(iterable1),len(iterable2)))

# def NN_architecture(n_features,n_categories,n_neuron):
#     lengths = [n_features]
#     for i in range(n_layers):
#         lengths.append(n_neuron[i])
#     lengths.append(n_categories)
#     return lengths


for i, it1 in enumerate(iterable1):     #run loops over hidden neurons and learning rates to calculate
    for j, it2 in enumerate(iterable2):      #accuracy scores
        # lengths = NN_architecture(X.shape[1],2,n_neuron)
        hyperparams[itIndices[0]] = it1
        hyperparams[itIndices[1]] = it2


        MLP=FFNN(X_train,y_train,lengths_h, sigmoidL, softmaxL, derCrossEntropy, True, hyperparams)
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
    sns.heatmap(100*data,xticklabels = x, yticklabels =y , annot=True, ax=ax, cmap="rocket", fmt = '.1f',cbar_kws={'format': '%.0f%%'})
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([0, 20, 40, 60, 80, 100])
    # cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_title(title)
    ax.set_xlabel(iterableStr[itIndices[1]])
    ax.set_ylabel(iterableStr[itIndices[0]])

indices = [x for x in range(len(valList))]
indices.remove(itIndices[0])
indices.remove(itIndices[1])

plot_data(train_accuracies, iterable1, iterable2, f"Cancer Data Training Accuracy(%) \n {iterableStr[indices[0]]}={hyperparams[indices[0]]},\
 {iterableStr[indices[1]]}={hyperparams[indices[1]]}, hidden layers={lengths_h}")
plot_data(test_accuracies, iterable1, iterable2, f"Cancer Data Test Accuracy(%) \n {iterableStr[indices[0]]}={hyperparams[indices[0]]}, \
 {iterableStr[indices[1]]}={hyperparams[indices[1]]}, hidden layers={lengths_h}")

plt.show()

#
# def plot_data(x,y,data,title=None):
#     # plot results
#     fontsize=16
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
#
#     cbar=fig.colorbar(cax)
#     cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
#     cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
#     cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])
#
#     # put text on matrix elements
#     for i, x_val in enumerate(np.arange(len(x))):
#         for j, y_val in enumerate(np.arange(len(y))):
#             c = "${0:.1f}\\%$".format( 100*data[j,i])
#             ax.text(x_val, y_val, c, va='center', ha='center')
#
#     # convert axis vaues to to string labels
#     x=[str(i) for i in x]
#     y=[str(i) for i in y]
#
#
#     ax.set_xticklabels(['']+x)
#     ax.set_yticklabels(['']+y)
#
#     ax.set_xlabel(f'{iterableStr[itIndices[0]]}',fontsize=fontsize)
#     ax.set_ylabel(f'{iterableStr[itIndices[1]]}',fontsize=fontsize)
#     if title is not None:
#         ax.set_title(title)
#
#     plt.tight_layout()
#
#
#
# plot_data(iterable1,iterable2,train_accuracies, 'training')
# plot_data(iterable1,iterable2,test_accuracies, 'testing')
# if(sklearnBool):
#     plot_data(iterable1,iterable2,train_accuracies_sk, 'training_sk')
#     plot_data(iterable1,iterable2,test_accuracies_sk, 'testing_sk')
# plt.show()
