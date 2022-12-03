import numpy as np
from activation_functions import sigmoidL,reluL,leakyReluL, tanhL
from layers import Dense




#Classical way. X.shape = [n_inputs, n_features]
#Here the weights are indexed with rows corresponding to the neurons of the previous layer and columns corresponding to nodes of the current layer
class FFNN:
    def __init__(self, X_full, y_full, hyperparams):
        self.epochs,self.batch_size,self.eta,self.lmbd= hyperparams

        self.X_full = X_full
        self.y_full = y_full
        self.n_inputs = X_full.shape[0]
        self.n_features = X_full.shape[1]
        self.n_categories = y_full.shape[1]

        self.layers = []
        self.L = None #Number of layers

    def addLayer(self,layer):
        self.layers.append(layer)

    ##-------------Initialize weights and biases------
    def initialize_network(self):
        self.L = len(self.layers)

        self.layers[0].initialize(self.X_full.shape) #Need to pass the shape of previous layer to construct weights
        for l in range(1,self.L):
            self.layers[l].initialize(self.layers[l-1].shape)

    ##--------------Make prediction---------------------
    def predict(self, X):
        self.layers[0].feedForward(X)
        for l in range(1,self.L):
            self.layers[l].feedForward(self.layers[l-1].A)

        return self.layers[-1].A


    def backpropagate(self,X,y):
        ##----------------backpropagate error--------------------------
        AL = self.layers[-1].A

        deltaL = AL-y #Note that we got overflow errors when trying to implement this in the form f'(AL)*(partial C)/(partial AL)

        self.layers[-1].delta = deltaL
        for l in range(self.L-2,-1,-1):
            self.layers[l].backpropagate(self.layers[l+1].delta,self.layers[l+1].W) #Here error(l) = delta(l+1)@W^T(l+1).


        ##------------Update weights and biases ---------------------------------
        self.layers[0].update(self.eta, self.lmbd, X)
        for l in range(1,self.L):
            self.layers[l].update(self.eta, self.lmbd, self.layers[l-1].A)


    def train(self):
        indices =np.arange(self.n_inputs)
        m = int(self.n_inputs/self.batch_size)

        for epoch in range(self.epochs):
            for i in range(m):
                batch_indices = np.random.choice(indices, self.batch_size, replace=False)
                X_batch = self.X_full[batch_indices]
                y_batch = self.y_full[batch_indices]
                self.predict(X_batch)
                self.backpropagate(X_batch,y_batch)

        print(f"Completed training with {self.epochs} epochs")



    def displayNetwork(self,errors=False):

        print("A: " )
        print("-------------------------------")
        for l in range(self.L):
            print(self.layers[l].A.shape)
        print("-------------------------------")
        if(errors == True) :
            print("Errors: " )
            print("-------------------------------")
        for l in range(self.L):
            print(self.layers[l].delta.shape)
            print("-------------------------------")
            print("batch_size: ", self.batch_size)
