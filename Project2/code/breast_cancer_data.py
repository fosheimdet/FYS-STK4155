import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from neural_network import FFNN, FFNN2
from activation_functions import noActL, tanhL, sigmoidL,reluL,softmaxL, derCrossEntropy, derMSE

def accuracy_score(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


def accuracy(output, targets, vertical):  #Assumes output to be softmax vector and targets one-hot-encoded
    score = 0
    if(vertical): #If data samples are stacked vertically
        n_data = output.shape[0]
        for i in range(n_data):
            if(np.argmax(output[i,:]) == np.argmax(targets[i,:])):
                score+=1
    else:
        n_data = output.shape[1]
        for i in range(n_data):
            if(np.argmax(output[:,i]) == np.argmax(targets[:,i])):
                score+=1

    return score/n_data



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

x=inputs      #Reassign the Feature and Label matrices to other variables
y=outputs



# Visualisation of dataset (for correlation analysis)
#
# plt.figure()
# plt.scatter(x[:,0],x[:,2],s=40,c=y,cmap=plt.cm.Spectral)
# plt.xlabel('Mean radius',fontweight='bold')
# plt.ylabel('Mean perimeter',fontweight='bold')
# plt.show()
#
# plt.figure()
# plt.scatter(x[:,5],x[:,6],s=40,c=y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean compactness',fontweight='bold')
# plt.ylabel('Mean concavity',fontweight='bold')
# plt.show()
#
#
# plt.figure()
# plt.scatter(x[:,0],x[:,1],s=40,c=y,cmap=plt.cm.Spectral)
# plt.xlabel('Mean radius',fontweight='bold')
# plt.ylabel('Mean texture',fontweight='bold')
# plt.show()
#
# plt.figure()
# plt.scatter(x[:,2],x[:,1],s=40,c=y,cmap=plt.cm.Spectral)
# plt.xlabel('Mean perimeter',fontweight='bold')
# plt.ylabel('Mean compactness',fontweight='bold')
# plt.show()


#Select features relevant to classification (texture,perimeter,compactness and symmetery)
#and add to input matrix

temp1=np.reshape(x[:,1],(len(x[:,1]),1))
temp2=np.reshape(x[:,2],(len(x[:,2]),1))
X=np.hstack((temp1,temp2))
temp=np.reshape(x[:,5],(len(x[:,5]),1))
X=np.hstack((X,temp))
temp=np.reshape(x[:,8],(len(x[:,8]),1))
X=np.hstack((X,temp))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)


y_train=to_categorical(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test=to_categorical(y_test)

del temp1,temp2,temp

# %%

# Define tunable parameters"

eta=np.logspace(-3,-1,3)                    #Define vector of learning rates (parameter to SGD optimiser)
lmbda=0.0                               #Define hyperparameter
n_layers=2                                  #Define number of hidden layers in the model
n_neuron=np.logspace(0,3,4,dtype=int)       #Define number of neurons per layer
n_neuron = [2,10,20,30]
epochs=100                                   #Number of reiterations over the input data
batch_size=100                              #Number of samples per gradient update



# n0 = X_train.shape[1] #Number of input nodes
# nhidden = int(input("Please enter the number of hidden layers \n"))
# lengths = [n0]
# for i in range(nhidden):
#     lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
# nL = 10         #Number of output nodes
# lengths.append(nL)


Train_accuracy=np.zeros((len(n_neuron),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(n_neuron),len(eta)))       #of learning rate and number of hidden neurons for


Train_accuracy2=np.zeros((len(n_neuron),len(eta)))
Test_accuracy2=np.zeros((len(n_neuron),len(eta)))


for i in range(len(n_neuron)):     #run loops over hidden neurons and learning rates to calculate
    for j in range(len(eta)):      #accuracy scores
        lengths = [X_train.shape[1]]
        for i in range(n_layers):
            lengths.append(n_neuron[i])
        lengths.append(2)
        hyperparams = [epochs,batch_size,eta[j],lmbda]

        MLP=FFNN(X_train.T,y_train.T,n_layers, lengths, reluL, softmaxL, derCrossEntropy, True, hyperparams)
        MLP.initializeNetwork()
        MLP.train()
        output_train, output_test = MLP.feedForward(X_train.T), MLP.feedForward(X_test.T)
        acc_train, acc_test = accuracy(output_train.T, y_train.T, False), accuracy(output_test.T, y_test.T, False)
        Train_accuracy[i,j] = acc_train
        Test_accuracy[i,j] = acc_test

        MLP2=FFNN2(X_train,y_train,n_layers, lengths, reluL, softmaxL, derCrossEntropy, True, hyperparams)
        MLP2.initializeNetwork()
        MLP2.train()
        output_train, output_test = MLP2.feedForward(X_train), MLP2.feedForward(X_test)
        acc_train, acc_test = accuracy(output_train, y_train, True), accuracy(output_test, y_test, True)
        Train_accuracy2[i,j] = acc_train
        Test_accuracy2[i,j] = acc_test



def plot_data(x,y,data,title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)

    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

plot_data(eta,n_neuron,Train_accuracy, 'training')
plot_data(eta,n_neuron,Test_accuracy, 'testing')

plot_data(eta,n_neuron,Train_accuracy2, 'training2')
plot_data(eta,n_neuron,Test_accuracy2, 'testing2')
