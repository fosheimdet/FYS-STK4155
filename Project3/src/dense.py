from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL
import numpy as np
from scipy import signal



class Dense:
    def __init__(self,n_l,actL):
        self.n_l = n_l            #Number of nodes in this layer
        self.act = actL[0]        #Activation function of this layers
        self.d_act = actL[1]      #Its derivative


        # self.n_samples = None   #Number of samples in our batch
        # self.n_features = None  #Number of features/nodes in previous layer
        self.shape = None       #Shape of this layer. Depends on number of samples in our batch
        self.W = None           #Weights for this layer
        self.b = None           #Biases for this layer
        self.A = None
        self.delta = None
        self.n_param = None

    def initialize(self,shape_prev,scheme=None): #Construct the weights and biases based on the shape of the input
        n_features = shape_prev[0] #original shape before pruning: (n_samples,n_features)

        n_prev = np.prod(shape_prev) #Number of nodes in previous layer
        var=1 #Variance of gaussian used for initializing weights
        if(scheme=="Xavier"):
            var = 1/n_prev
        elif(scheme=="He"):
            var = 2/n_prev
        self.W = np.random.normal(0,1/var,(n_features,self.n_l))
        self.b = np.repeat(0.0,self.n_l)
        # self.W = np.random.normal(0,1,(n_features,self.n_l)) #Initialize using normal distribution
        # self.b = np.repeat(0.01,self.n_l) #Will be automatically "reshaped" by numpy to a matrix in vector-matrix addition
        self.n_param = np.prod(self.W.shape)+len(self.b)

        self.shape = (self.n_l,)

    def feedForward(self, input): # Here input will be A^{l-1}
        self.A = self.act(input@self.W + self.b)
        return self.A


    def backpropagate(self, input): #input = delta^{l+1} @ W^{l+1}.T

        self.delta = self.d_act(self.A)*input

        return self.delta@self.W.T       #Passing only delta would require that the next layer aquire this layers weights to calculate its delta



    def update(self,A_prev,eta,lmbd):
        self.W -= eta*(A_prev.T@self.delta +lmbd*self.W)
        self.b -= eta*(np.ones(self.delta.shape[0])@self.delta + lmbd*self.b)

    def info(self):
        act_name = self.act.__name__
        return f'({self.n_l},{act_name})'
