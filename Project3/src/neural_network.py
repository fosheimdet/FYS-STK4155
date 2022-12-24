import numpy as np
import time
import matplotlib.pyplot as plt

from functions import accuracy
from activation_functions import sigmoidL,reluL,leakyReluL, tanhL
# from layers import Dense




#input_shape = (n_samples,Height,Width,n_channels) for CNNs
#and (n_samples,n_features) for DNNs
class CNN:
    def __init__(self,name="CNN"):
        self.name = name
        self.shape_features = None

        self.layers = []
        self.L = None #Number of layers

        self.epochs,self.batch_size = None,None
        self.eta,self.lmbd = None,None
        self.scheme = None #Initialization scheme. Defaults to normal w. var=1 if provided nothing

    def addLayer(self,layer):
        self.layers.append(layer)

    def changeAct(self,layer,actL):
        self.layers[layer].act = actL[0]
        self.layers[layer].d_act = actL[1]

    ##-------------Initialize weights and biases------
    def initialize_weights(self,input_shape):
        self.shape_features = input_shape[1:]

        self.L = len(self.layers)
        self.layers[0].initialize(self.shape_features,self.scheme) #Need to pass the shape of previous layer to construct weights
        for l in range(1,self.L):
            self.layers[l].initialize(self.layers[l-1].shape,self.scheme)

    def set_hyperparams(self,hyperparams):
        epochs,batch_size,eta,lmbd = hyperparams
        self.epochs,self.batch_size,self.eta,self.lmbd = epochs,batch_size,eta,lmbd

    ##--------------Make prediction---------------------
    def predict(self, X):
        # t_start = time.time()
        self.layers[0].feedForward(X)
        for l in range(1,self.L):
            self.layers[l].feedForward(self.layers[l-1].A)

        # t_end = time.time()
        # print(f"Forward prop. time: {t_end-t_start:.5f}", )
        return self.layers[-1].A


    def backpropagate(self,y):
        ##----------------backpropagate error--------------------------
        #t_start = time.time()
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

        # t_end = time.time()
        # print(f"Backpropagation time: {t_end-t_start:.5f}", )
        return error_s

    #Previous layer's activations needed for updating
    def update(self,X):
        ##------------Update weights and biases ---------------------------------
        self.layers[0].update(X, self.eta, self.lmbd)
        for l in range(1,self.L):
            self.layers[l].update(self.layers[l-1].A, self.eta, self.lmbd)


    def train(self, X_train, y_train, X_val=None,y_val=None,verbose=True):
        epochs, batch_size, eta, lmbd = self.epochs,self.batch_size,self.eta,self.lmbd
        n_inputs = X_train.shape[0]
        #==========Validation stuff==========
        iter = 0 #Iteration number
        iter_step =int(X_train.shape[0]/batch_size)  #Evaluate every n'th batch
        evalBool = True
        if(type(X_val)!=type(X_train) or type(y_val)!=type(y_train)):
            evalBool = False #Perform no evaluations, speeds up training

        acc_val_l,acc_train_l =[],[]
        epoch_list=[] #Epochs in which the evaluations occured
        #====================================

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
                self.update(X_batch)

                if(iter%iter_step==0 and evalBool==True):
                    y_pred = self.predict(X_val)
                    y_tilde = self.predict(X_train)
                    acc_val= accuracy(y_pred,y_val)
                    acc_train = accuracy(y_tilde,y_train)
                    acc_val_l.append(acc_val)
                    acc_train_l.append(acc_train)
                    epoch_list.append(epoch+i/m)
                    if(verbose):
                        print(f"Epoch:{epoch+i/m:.3f} train_acc:{acc_train:.4f}  val_acc:{acc_val:.4f}")

                iter+=1
            t_end_epoch = time.time()
            # if(verbose):
            #     print(f"Finished epoch {epoch}/{epochs} in {t_end_epoch-t_start_epoch:.3f} s.")
        t_end = time.time()
        t_train = t_end-t_start
        print("======================================================================")
        print(f"      Completed training with {epochs} epochs in {t_train:.3f} s = {t_train//60:.0f}min,{t_train%60:.0f}sec.")
        print("======================================================================")
        return acc_val_l,acc_train_l,epoch_list,t_train
##==============================================================================
##                        Handy  functions
##==============================================================================
    #Does not account for validation history as of now
    def estimate_train_time(self, X_train, y_train,val_size):
        # from sklearn.model_selection import train_test_split
        #
        # X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size)

        epochs, batch_size, eta, lmbd = self.epochs,self.batch_size,self.eta,self.lmbd
        n_inputs = X_train.shape[0]
        indices =np.arange(n_inputs)
        m = n_inputs/batch_size
        t_start = time.time()
        print("Number of batches in 1 epoch: ", m)
        n_batches = 10
        # print(f"Timing {n_batches} batches = {(n_batches*batch_size)/n_inputs:.3f} epochs")
        print(f"Timing {n_batches} batches = {n_batches/m:.3f} epochs")

        for epoch in range(1):
            t_start = time.time()
            for i in range(n_batches):
                batch_indices = np.random.choice(indices, batch_size, replace=False)
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                self.predict(X_batch)
                self.backpropagate(y_batch)
                self.update(X_batch)

            t_end= time.time()
            self.initialize_weights(X_train.shape) #Undo mini-training

            t_batches = t_end-t_start
            # t_1epoch= (n_inputs/(n_batches*batch_size))*t_batches
            t_1epoch= (m/n_batches)*t_batches

            t_estimated = epochs*t_1epoch
            return t_estimated

    def summary(self,errors=False):
        n_tot = 0
        sum_str = ''
        sum_str+="----------------------------------------------------------\n"
        sum_str+="Layer                       Output shape          #Params  \n"
        sum_str+="==========================================================\n"
        sum_str+='{:27} {:22s} {}\n'.format('Input',str(self.shape_features),16)

        for l in range(self.L):
            sum_str+="----------------------------------------------------------\n"
            layer = self.layers[l]
            name = type(layer).__name__+layer.info()
            shape = str(layer.shape)
            n_params = layer.n_param
            sum_str+='{:27s} {:22s} {}  \n'.format(name,shape,n_params)

            n_tot+=n_params
        sum_str+="==========================================================\n"
        sum_str+=f'Total #parameters: {n_tot}\n'
        sum_str+="----------------------------------------------------------\n"
        return sum_str

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
