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

    def initialize(self,shape_prev): #Construct the weights and biases based on the shape of the input
        n_features = shape_prev[0]
        self.W = np.random.normal(0,1,(n_features,self.n_l)) #Initialize using normal distribution
        self.b = np.repeat(0.01,self.n_l) #Will be automatically shaped to appropriately
        self.n_param = np.prod(self.W.shape)+len(self.b)

        self.shape = (self.n_l,)

    def feedForward(self, input): # Here input will be A^{l-1}
        self.A = self.act(input@self.W + self.b)
        return self.A


    def backpropagate(self, input): #input = delta^{l+1} @ W^{l+1}.T

        self.delta = self.d_act(self.A)*input

        return self.delta@self.W.T       #Passing only delta would require that the next layer aquire this layers weights to calculate its delta

    # def backpropagate2(self, delta_s, W_s): #input = delta^{l+1} @ W^{l+1}.T
    #     self.delta = self.d_act(self.A)*(delta_s@ W_s.T)
    #     return self.delta        #Passing only delta would require that the next layer aquire this layers weights to calculate its delta


    def update(self,eta,lmbd,A_p):
        self.W -= eta*(A_p.T@self.delta +lmbd*self.W)
        self.b -= eta*(np.ones(self.delta.shape[0])@self.delta + lmbd*self.b)
