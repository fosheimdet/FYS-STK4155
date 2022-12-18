import numpy as np
import time
import matplotlib.pyplot as plt

from activation_functions import sigmoidL,reluL,leakyReluL, tanhL
# from layers import Dense



#input_shape = (n_samples,Height,Width,n_channels) for CNNs
#and (n_samples,n_features) for DNNs
class CNN:
    def __init__(self,input_shape):
        self.shape_features = input_shape[1:]

        self.layers = []
        self.L = None #Number of layers

    def addLayer(self,layer):
        self.layers.append(layer)

    def changeAct(self,layer,actL):
        self.layers[layer].act = actL[0]
        self.layers[layer].d_act = actL[1]

    ##-------------Initialize weights and biases------
    def initialize_network(self):
        self.L = len(self.layers)
        self.layers[0].initialize(self.shape_features) #Need to pass the shape of previous layer to construct weights
        for l in range(1,self.L):
            self.layers[l].initialize(self.layers[l-1].shape)


    ##--------------Make prediction---------------------
    def predict(self, X):
        self.layers[0].feedForward(X)
        for l in range(1,self.L):
            self.layers[l].feedForward(self.layers[l-1].A)

        return self.layers[-1].A


    def backpropagate(self,y):
        ##----------------backpropagate error--------------------------

        AL = self.layers[-1].A
        deltaL = AL-y #Note that we got overflow errors when trying to implement this in the form f'(AL)*(partial C)/(partial AL)


        #deltaL = np.ones(self.layers[-1].A.shape)
        self.layers[-1].delta = deltaL
        error_s = deltaL@self.layers[-1].W.T #Succeeding error
        #error_s = deltaL
        # error_s = -y/AL

        for l in range(self.L-2,-1,-1):
            error = self.layers[l].backpropagate(error_s)
            error_s = error

        ##------------------------------------

        # layL = self.layers[-1]
        # prod=layL.d_act(layL.A[0,:])*(y[0,:]/layL.A[0,:])
        # print("AL-y: \n", layL.A[0,:]-y[0,:])
        # print("f'(AL)*dCdA: \n", prod)
        #
        # AL = self.layers[-1].A
        # error_s = -y/AL
        #
        # for l in range(self.L-1,-1,-1):
        #     error = self.layers[l].backpropagate(error_s)
        #     error_s = error



    def update(self,X, eta, lmbd):
        ##------------Update weights and biases ---------------------------------
        self.layers[0].update(X, eta, lmbd)
        for l in range(1,self.L):
            self.layers[l].update(self.layers[l-1].A, eta, lmbd)


    def train(self, X_train, y_train, hyperparams):
        epochs, batch_size, eta, lmbd = hyperparams

        n_inputs = X_train.shape[0]
        indices =np.arange(n_inputs)
        m = int(n_inputs/batch_size)
        t_start = time.time()
        for epoch in range(epochs):
            t_start_epoch = time.time()
            for i in range(m):
                batch_indices = np.random.choice(indices, batch_size, replace=False)
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                self.predict(X_batch)
                self.backpropagate(y_batch)
                self.update(X_batch,eta,lmbd)
            t_end_epoch = time.time()
            print(f"Finished epoch {epoch}/{epochs} in {t_end_epoch-t_start_epoch} s.")
        t_end = time.time()
        print(f"Completed training with {epochs} epochs in {t_end-t_start} s.")


##==============================================================================
##                     Visualization functions
##==============================================================================
    def summary(self,errors=False):
        print("A: " )
        print("----------------------------------------------------")
        print("Layer            Output shape            Param # ")
        print("====================================================")
        print('{:19s} {:22s} {}'.format('Input',str(self.shape_features),16))
        print("----------------------------------------------------")
        for l in range(self.L):
            layer = self.layers[l]
            # print(self.layers[l].A.shape)
            name = type(layer).__name__
            shape = str(layer.shape)
            n_params = layer.n_param
            print('{:19s} {:22s} {}'.format(name,shape,n_params))
            # print(type(layer).__name__, "\t\t", layer.shape )
            print("----------------------------------------------------")

    #Plots the feature maps of a convolutional layer. The input can
    #be provided as an optional argument for comparison.
    def plot_FMs(self,layer,sample,input=np.array([0])):
        m_one = 0 # Needs to be one if we want an additional subplot for the input
        if(len(input.shape)>1):
            m_one = 1
        output = self.layers[layer].A
        n_channels = output.shape[-1]
        a = m_one+n_channels
        #Make appropriate number of subplots
        n_cols = int(np.ceil(np.sqrt(a)))
        n_rows =int(np.ceil(a/n_cols))

        fig = plt.figure(figsize=(n_rows, n_cols))
        # plt.title("hey")
        if(len(input.shape)>1):
            fig.add_subplot(n_rows, n_cols, 1)
            plt.imshow(input[sample,:,:,1], cmap='gray')
            #plt.imshow(input[sample,:,:,:])
            plt.title(f"Input[{sample}]")
        for i in range(1, n_channels+1):
            fig.add_subplot(n_rows, n_cols, i+m_one)
            plt.imshow(output[sample,:,:,i-1], cmap='gray')
            plt.title(f"channel {i-1}")
        fig.suptitle(f"layer {layer}")
        #plt.show() needs to be included in main script, allowing multiple layers to be displayed at once.

    def plot_kernels(self,layer,channel):

        kernels = self.layers[layer].F[:,:,:,channel]
        n_filters = self.layers[layer].n_filters
        a = n_filters
        #Make appropriate number of subplots
        n_cols = int(np.ceil(np.sqrt(a)))
        n_rows =int(np.ceil(a/n_cols))

        fig = plt.figure(figsize=(n_rows, n_cols))
        # plt.title("hey")
        for f in range(0, n_filters):
            fig.add_subplot(n_rows, n_cols, f+1)
            plt.imshow(kernels[f,:,:], cmap='gray')
            plt.title(f"filter {f}")
        fig.suptitle(f"layer {layer}, channel {channel}")
        #plt.show() needs to be included in main script, allowing multiple layers to be displayed at once.

# #Classical way. X.shape = [n_inputs, n_features]
# #Here the weights are indexed with rows corresponding to the neurons of the previous layer and columns corresponding to nodes of the current layer
# class FFNN:
#     def __init__(self, X_full, y_full, hyperparams):
#         self.epochs,self.batch_size,self.eta,self.lmbd= hyperparams
#
#         self.X_full = X_full
#         self.y_full = y_full
#         self.n_inputs = X_full.shape[0]
#         self.n_features = X_full.shape[1]
#         self.n_categories = y_full.shape[1]
#
#         self.layers = []
#         self.L = None #Number of layers
#
#     def addLayer(self,layer):
#         self.layers.append(layer)
#
#     ##-------------Initialize weights and biases------
#     def initialize_network(self):
#         self.L = len(self.layers)
#
#         self.layers[0].initialize(self.X_full.shape) #Need to pass the shape of previous layer to construct weights
#         for l in range(1,self.L):
#             self.layers[l].initialize(self.layers[l-1].shape)
#
#     ##--------------Make prediction---------------------
#     def predict(self, X):
#         self.layers[0].feedForward(X)
#         for l in range(1,self.L):
#             self.layers[l].feedForward(self.layers[l-1].A)
#
#         return self.layers[-1].A
#
#
#     def backpropagate(self,X,y):
#         ##----------------backpropagate error--------------------------
#         AL = self.layers[-1].A
#
#         deltaL = AL-y #Note that we got overflow errors when trying to implement this in the form f'(AL)*(partial C)/(partial AL)
#
#         self.layers[-1].delta = deltaL
#         error_s = deltaL@self.layers[-1].W.T #Succeeding error
#
#         for l in range(self.L-2,-1,-1):
#             error = self.layers[l].backpropagate(error_s)
#             error_s = error
#
#
#         # for l in range(self.L-2,-1,-1):
#         #     self.layers[l].backpropagate(self.layers[l+1].delta,self.layers[l+1].W) #Here error(l) = delta(l+1)@W^T(l+1).
#
#
#         ##------------Update weights and biases ---------------------------------
#         self.layers[0].update(self.eta, self.lmbd, X)
#         for l in range(1,self.L):
#             self.layers[l].update(self.eta, self.lmbd, self.layers[l-1].A)
#
#
#     def train(self):
#         indices =np.arange(self.n_inputs)
#         m = int(self.n_inputs/self.batch_size)
#
#         for epoch in range(self.epochs):
#             for i in range(m):
#                 batch_indices = np.random.choice(indices, self.batch_size, replace=False)
#                 X_batch = self.X_full[batch_indices]
#                 y_batch = self.y_full[batch_indices]
#                 self.predict(X_batch)
#                 self.backpropagate(X_batch,y_batch)
#
#         print(f"Completed training with {self.epochs} epochs")
#
#
#
#     def displayNetwork(self,errors=False):
#
#         print("A: " )
#         print("-------------------------------")
#         for l in range(self.L):
#             print(self.layers[l].A.shape)
#         print("-------------------------------")
#         if(errors == True) :
#             print("Errors: " )
#             print("-------------------------------")
#         for l in range(self.L):
#             print(self.layers[l].delta.shape)
#             print("-------------------------------")
#             print("batch_size: ", self.batch_size)
