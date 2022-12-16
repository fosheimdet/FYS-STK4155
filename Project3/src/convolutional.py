from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL
import numpy as np
from scipy import signal




class Conv2D:
    def __init__(self,shape_kernel,actL, padding="same"):
        self.r = int((shape_kernel[0]-1)/2) #For now only using square kernels
        self.p = None  #Padding size to be used on input from previous layer
        self.padding = padding
        self.act = actL[0]
        self.d_act = actL[1]

        self.shape = None       #Shape of this layer. Depends on number of samples in our batch
        self.shape_prev = None #Shape of previous layer. Needed for backpropagation
        self.K = None
        self.B = None
        self.A = None
        self.delta = None
        self.n_param = None

    def initialize(self,shape_prev):
        self.shape_prev = shape_prev
        self.K = np.random.normal(0,1,(2*self.r+1,2*self.r+1))
        self.B = 0.01
        self.n_param = np.prod(self.K.shape) + 1

        if(self.padding == "valid"):
            self.p = 0
        elif(self.padding == "same"):
            self.p = self.r
        elif(self.padding == "full"):
            self.p = 2*self.r


        Height_p,Width_p = shape_prev
        Height = Height_p - 2*self.r + 2*self.p
        Width = Width_p - 2*self.r + 2*self.p

        self.shape = (Height, Width)
        return self.shape

    def feedForward(self,input):
        n_input = input.shape[0]
        input_padded = np.pad(input,((0,0),(self.p,self.p),(self.p,self.p)))
        self.A = np.zeros((n_input,self.shape[0],self.shape[1]))

        for i in range(n_input):
            self.A[i,:,:] = self.act( signal.correlate2d(input_padded[i],self.K, mode="valid") )

        return self.A

    def backpropagate(self,input):
        n_samples = input.shape[0]
        self.delta = self.d_act(self.A)*input
        #Pad the delta
        p_delta = int( (self.shape_prev[0]-self.shape[0]+2*self.r)/2 )
        delta_padded = np.pad(self.delta,((0,0),(p_delta,p_delta),(p_delta,p_delta)))

        #Generate errors for next layer (delta^{l+1}*K^{l+1})
        error = np.zeros((n_samples,)+self.shape_prev)
        for i in range(n_samples):
            error[i,:,:] = signal.convolve2d(delta_padded[i,:,:], self.K, mode="valid")

        return error


    def update(self,eta,lmbd,A_p):
        p = self.p
        A_p_padded = np.pad(A_p,((0,0),(p,p),(p,p)))

        #As in the previous project, we update using the sum of the contributions
        #from each sample
        DK = np.zeros(self.K.shape)

        for n in range(A_p.shape[0]):
            DK+= signal.correlate2d(A_p_padded[n,:,:],self.delta[n,:,:],"valid")

        self.K -= eta*(DK + lmbd*self.K)
        self.B -= eta*(np.sum(self.delta) + lmbd*self.B)


        # self.W -= eta*(A_p.T@self.delta +lmbd*self.W)
        # self.b -= eta*(np.ones(self.delta.shape[0])@self.delta + lmbd*self.b)
