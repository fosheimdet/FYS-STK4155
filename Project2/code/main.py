import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns

from functions import FrankeFunction, desMat, shuffle_in_unison, addNoise, getMSE, getR2
from neural_network import FFNN, FFNN2
from activation_functions import noActL, tanhL, sigmoidL,reluL,softmaxL, derCrossEntropy, derMSE





# def accuracy(output, targets):  #Assumes output to be softmax vector and targets one-hot-encoded
#     score = 0
#     n_data = output.shape[0]
#     for i in range(n_data):
#         if(np.argmax(output[i,:]) == np.argmax(targets[i,:])):
#             score+=1
#     return score/n_data

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

def transformData(inputs,labels):
    n_inputs = len(inputs)
    X = np.zeros((64, n_inputs))
    for i in range(n_inputs):
        X[:,i] = np.ravel(inputs[i])

    onehotlabels = np.zeros((10,n_inputs))
    for i in range(n_inputs):
        for j in range(10):
            if labels[i]==j:
                onehotlabels[j][i] = 1

    return X, onehotlabels

def main():

    # n = 20
    # x = np.linspace(0,1,n)
    # y = np.linspace(0,1,n)
    # xx,yy = np.meshgrid(x,y)
    # Z = FrankeFunction(xx,yy)
    # z = np.ravel(Z).reshape(-1,1)
    #
    # xr = np.ravel(xx)
    # yr = np.ravel(yy)
    #
    # X = desMat(xr,yr,4)
    # X_train, X_test, z_train, z_test = train_test_split(X,z, train_size = 0.8)
    #
    # MSE, R2, betas, ztilde = SGD(X_train,X_test,z_train,z_test,100, 20 ,eta=0.1, lmd=0,gamma = 0)


    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size = 0.8)


    print("inputs_train size: ", inputs_train.shape)
    print("labels_train size: ", labels_train.shape)
    print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
    print("labels = (n_inputs) = " + str(labels.shape))


    X_train, y_train = transformData(inputs_train, labels_train)
    X_test, y_test = transformData(inputs_test, labels_test)

    print("X_train size: ", X_train.shape)
    print("y_test size: ", y_test.shape)


    # n_inputs = 100
    #
    # y = onehotlabels[:,0:n_inputs]


    n0 = X_train.shape[0] #Number of input nodes
    nhidden = int(input("Please enter the number of hidden layers \n"))
    lengths = [n0]
    for i in range(nhidden):
        lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
    nL = 10         #Number of output nodes
    lengths.append(nL)

    # epochs = 100
    # batch_size = 100
    # eta = 0.1
    # lmbd = 1e-3
    #
    # eta_vals = np.logspace(-5, 1, 7)
    # lmbd_vals = np.logspace(-8, -2, 7)

    sns.set()



    # store the models for later use
    # DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

    #Test classifier on one MNIST datasample
    #-------------------------------------------------
    # hyperparams = [epochs,batch_size, eta, lmbd]
    # MLP = FFNN(X_train,y_train,nhidden,lengths, sigmoidL, softmaxL, derCrossEntropy ,True,hyperparams)
    # MLP.initializeNetwork()
    # t_start = time.time()
    # MLP.train()
    # t_end = time.time()
    # output_test = MLP.feedForward(X_test)
    # test_acc = accuracy(output_test, y_test)
    # print(test_acc)
    # output = MLP.feedForward(X_test[:,20].reshape(-1,1))
    # MLP.displayNetwork()
    #
    # float_formatter = "{:.3f}".format
    # np.set_printoptions(formatter={'float_kind':float_formatter})
    # print(output.reshape(-1))
    # print(y_test[:,20].reshape(-1))
    # print("Training time: ", t_end-t_start)
    # #-------------------------------------------------


    #
    # for i, eta in enumerate(eta_vals):
    #     for j, lmbd in enumerate(lmbd_vals):
    #         hyperparams = [epochs, batch_size, eta, lmbd]
    #         MLP = FFNN2(X_train.T, y_train.T, nhidden, lengths,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
    #         MLP.initializeNetwork()
    #         MLP.train()
    #
    #         DNN_numpy[i][j] = MLP
    #
    #         output_train, output_test = MLP.feedForward(X_train.T), MLP.feedForward(X_test.T)
    #         acc_train, acc_test = accuracy(output_train, y_train.T), accuracy(output_test, y_test.T)
    #         train_accuracies[i,j] = acc_train
    #         test_accuracies[i,j] = acc_test

    epoch_vals = [10,30,50,80,100,150]
    M_vals = [10,20,30,40,50,60,70,100]
    batch_size = 5
    eta = 0.1
    lmbd = 0

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-8, -2, 7)


    train_accuracies = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracies = np.zeros((len(eta_vals), len(lmbd_vals)))

    train_accuracies2 = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracies2 = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i, eta in enumerate(eta_vals):
        for j, epochs in enumerate(epoch_vals):
            hyperparams = [epochs, batch_size, eta, lmbd]
            MLP = FFNN(X_train, y_train, nhidden, lengths,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
            MLP.initializeNetwork()
            MLP.train()

            # DNN_numpy[i][j] = MLP

            output_train, output_test = MLP.feedForward(X_train), MLP.feedForward(X_test)
            acc_train, acc_test = accuracy(output_train, y_train, False), accuracy(output_test, y_test, False)
            train_accuracies[i,j] = acc_train
            test_accuracies[i,j] = acc_test


            MLP = FFNN2(X_train.T, y_train.T, nhidden, lengths,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
            MLP.initializeNetwork()
            MLP.train()

            # DNN_numpy[i][j] = MLP

            output_train, output_test = MLP.feedForward(X_train.T), MLP.feedForward(X_test.T)
            acc_train, acc_test = accuracy(output_train, y_train.T, True), accuracy(output_test, y_test.T, True)
            train_accuracies2[i,j] = acc_train
            test_accuracies2[i,j] = acc_test


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracies,xticklabels = epoch_vals, yticklabels =eta_vals , annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy1")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("epochs")
    # ax.set_xlabel("$\lambda$")


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracies2,xticklabels = epoch_vals, yticklabels =eta_vals , annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy2")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("epochs")
    # ax.set_xlabel("$\lambda$")
    plt.show()

    # for i, eta in enumerate(eta_vals):
    #     for j, lmbd in enumerate(lmbd_vals):
    #         hyperparams = [epochs, batch_size, eta, lmbd]
    #         MLP = FFNN2(X_train.T, y_train.T, nhidden, lengths,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
    #         MLP.initializeNetwork()
    #         MLP.train()
    #
    #         DNN_numpy[i][j] = MLP
    #
    #         output_train, output_test = MLP.feedForward(X_train.T), MLP.feedForward(X_test.T)
    #         acc_train, acc_test = accuracy(output_train, y_train.T), accuracy(output_test, y_test.T)
    #         train_accuracies[i,j] = acc_train
    #         test_accuracies[i,j] = acc_testbels = lmbd_vals, yticklabels = eta_vals,  annot=True, ax=ax, cmap="viridis")


    # prediction = MLP.predict(X[:,0:100], y[:,0:100])
    # print(prediction)
    #
    # MLP.displayNetwork()
    #
    # print("Network prediction: ", np.argmax(prediction), "label: ", labels[datapoint])
    # print(np.sum(MLP.output))



    # print(MLP.output,"\t", onehotlabels[:,120].reshape(-1,1))
    # print(labels[120])






        # choose some random images to display
    # indices = np.arange(n_inputs)
    # random_indices = np.random.choice(indices, size=5)
    #
    # for i, image in enumerate(digits.images[random_indices]):
    #     plt.subplot(1, 5, i+1)
    #     plt.axis('off')
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title("Label: %d" % digits.target[random_indices[i]])
    # plt.show()











    # fig, ax = plt.subplots(figsize = (10, 10))
    # sns.heatmap(test_accuracies,xticklabels = lmbd_vals, yticklabels = eta_vals,  annot=True, ax=ax, cmap="viridis")
    # ax.set_title("Test Accuracy")
    # ax.set_ylabel("$\eta$")
    # ax.set_xlabel("$\lambda$")
    # plt.show()

    # x = np.concatenate((xr,yr),axis = 0)
    # x = xr.reshape(-1,1)
    # # x = np.array([[1],[2]])
    # print(x.shape)
    # # x = np.array([1])
    # n0 = len(x) #Number of input nodes
    # nhidden = int(input("Please enter the number of hidden layers \n"))
    # lengths = [n0]
    # for i in range(nhidden):
    #     lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
    # nL = len(z_noisy)         #Number of output nodes
    # lengths.append(nL)
    #
    #
    # MLP = NeuralNetworkMatrix(x,z_noisy,nhidden,lengths, 10, 10, 0.01)
    # MLP.initializeNetwork()
    # MLP.feedForward(x)
    #
    # MLP.backpropagate(z_noisy)
    # MLP.displayNetwork()
    # for i in range(1000):
    #     MLP.feedForward(x)
    #     MLP.backpropagate(z_noisy)
    # MLP.feedForward(x)
    # ztilde = MLP.output


    #
    # n0 = len(z_noisy) #Number of input nodes
    # nhidden = int(input("Please enter the number of hidden layers \n"))
    # lengths = [n0]
    # for i in range(nhidden):
    #     lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
    # nL = len(z_noisy)         #Number of output nodes
    # lengths.append(nL)
    #
    # MLP = NeuralNetworkMatrix(z_noisy,z_noisy,nhidden,lengths, 10, 10, 0.01)
    # MLP.initializeNetwork()
    # # MLP.feedForward(z_noisy)
    # # MLP.backpropagate()
    # for i in range(100):
    #     z_noisy = addNoise(z,sigma).reshape(-1,1)
    #     MLP.feedForward(z_noisy)
    #     MLP.backpropagate(z_noisy)
    # MLP.feedForward(addNoise(z,sigma).reshape(-1,1))
    # ztilde = MLP.output


    # X = np.array([[0,0], [0,1], [1,0], [1,1]]).T
    # y = np.array([0, 1, 1, 0])
    #
    # n0 = X.shape[0] #Number of input nodes
    # nhidden = int(input("Please enter the number of hidden layers \n"))
    # lengths = [n0]
    # for i in range(nhidden):
    #     lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
    # nL = 1         #Number of output nodes
    # lengths.append(nL)
    #
    # MLP = MLPxor(X, y, nhidden, lengths, 10, 20, 0.1)
    # MLP.initializeNetwork()
    # MLP.feedForward(X)
    # MLP.backpropagate()
    # for i in range(1000):
    #     MLP.feedForward(X)
    #     MLP.backpropagate()
    # MLP.feedForward(X)
    # print(MLP.output)
    # MLP.feedForward(np.array([[0],[1]]))
    # print(MLP.output)


    # n0 = z.shape[0] #Number of input nodes
    # nhidden = int(input("Please enter the number of hidden layers \n"))
    # lengths = [n0]
    # for i in range(nhidden):
    #     lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
    # nL = n0         #Number of output nodes    Ztilde = ztilde.reshape(n,n)









    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # # surf = ax.plot_surface(xx,yy,(addNoise(z,sigma).reshape(n,n)), cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
    # surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
    # plt.show()

    # lengths.append(nL)
    #
    # MLP = NeuralNetwork(z,nhidden,lengths)
    #
    # MLP.initializeNetwork()
    # MLP.feedForward(z_noisy)
    # for i in range(10):
    #     MLP.feedForward(addNoise(z,0.1))
    #     MLP.backpropagate()
    # MLP.feedForward(addNoise(z,0.1))
    # ztilde = MLP.output
    #
    #
    #
    #
    #





if __name__ =="__main__":
    main()
