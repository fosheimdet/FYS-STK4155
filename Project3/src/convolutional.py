from activation_functions import noActL, sigmoidL, reluL, tanhL, softmaxL
import numpy as np
from scipy import signal




class Conv:
    def __init__(self,n_filters,shape_kernel,actL, padding="same", p=None):
        self.n_filters = n_filters
        self.k = shape_kernel[0] #For now only using square, even-sized kernels
        self.act = actL[0] #Actiation function
        self.d_act = actL[1] #Its derivative
        self.pad_mode = padding #Valid or same padding
        self.p = p  #Padding size to be used on input from previous layer



        self.shape = None      #Shape of this layer, (Height,Width,n_filters)
        self.shape_prev = None #Shape of previous layer. Needed for backpropagation
        self.F = None    #Filters for the layer. Shape: (n_filters,Heigh,Width,n_channels)
        self.b = None    #Biases. One for each filter.
        self.A = None    #Activation values
        self.delta = None #"Errors"
        self.n_param = None #Only used for printing the network

    def initialize(self,shape_prev, scheme=None):
        self.shape_prev = shape_prev #Shape of activations of previous layer (excluding sample axis)
        n_channels = shape_prev[-1] #Number of feature maps

        n_prev = np.prod(shape_prev) #Number of nodes in previous layer

        var=1 #Variance of gaussian used for initializing weights
        if(scheme=="Xavier"):
            var = 1/n_prev
        elif(scheme=="He"):
            var = 2/n_prev
        self.F = np.random.normal(0,var,(self.n_filters,self.k,self.k,n_channels))
        self.b = 0.0*np.ones(self.n_filters)
        # self.b = 0.01*np.ones(self.n_filters)
        self.n_param = np.prod(self.F.shape) + self.n_filters

        if(self.pad_mode == "valid"):
            self.p = 0
        elif(self.pad_mode == "same"):
            # self.p = self.r
            self.p = int((self.k-1)/2)
        elif(self.pad_mode == "full"):
            # self.p = 2*self.r
            self.p = self.k-1
        elif(self.pad_mode == "custom"):
            self.p = self.p

        Height_p,Width_p = shape_prev[0:2]
        Height = 1 + Height_p - self.k + 2*self.p
        Width =  1 + Width_p- self.k + 2*self.p

        self.shape = (Height, Width, self.n_filters) #Number of channels needed for initializing the first dense layer
        return self.shape

    def feedForward(self,input):
        n_input = input.shape[0]
        n_channels = input.shape[-1]
        input_padded = np.pad(input,((0,0),(self.p,self.p),(self.p,self.p),(0,0)))
        Z = np.zeros((n_input,)+self.shape)
        self.A = np.copy(Z)

        for n in range(n_input):
            for f in range(self.n_filters):
                for c in range(n_channels):
                    Z[n,:,:,f]+=signal.correlate2d(input_padded[n,:,:,c],self.F[f,:,:,c], mode="valid")
                Z[n,:,:,f]+=self.b[f]
        self.A = self.act(Z)
        return self.A

    def backpropagate(self,input):
        n_samples = input.shape[0]
        self.delta = self.d_act(self.A)*input
        #Pad the delta
        p_delta = int( (self.shape_prev[0]-self.shape[0]+(self.k-1))/2 )
        delta_padded = np.pad(self.delta,((0,0),(p_delta,p_delta),(p_delta,p_delta),(0,0)))

        #Generate errors for next layer (delta^{l+1}*K^{l+1})
        error = np.zeros((n_samples,)+self.shape_prev)
        n_channels_prev = self.shape_prev[-1]
        for n in range(n_samples):
            for c in range(n_channels_prev):
                for f in range(self.n_filters):
                    error[n,:,:,c] += signal.convolve2d(delta_padded[n,:,:,f],self.F[f,:,:,c], mode="valid")

        return error


    def update(self,A_prev,eta,lmbd):
        n_samples = A_prev.shape[0]
        n_channels = A_prev.shape[-1]
        p = self.p #Padding used on input during forward propagation
        A_prev_padded = np.pad(A_prev,((0,0),(p,p),(p,p),(0,0)))

        #As in the previous project, we update using the sum of the contributions
        #from each sample
        grad_F = np.zeros(self.F.shape) #Update to the filter = summed contribution from all samples

        for n in range(n_samples):
            for f in range(self.n_filters):
                for c in range(n_channels):
                    grad_F[f,:,:,c] += signal.correlate2d(A_prev_padded[n,:,:,c],self.delta[n,:,:,f],mode="valid")

        grad_b = np.zeros(len(self.b))
        for n in range(n_samples):
            for f in range(self.n_filters):
                grad_b[f]+= np.sum(self.delta[n,:,:,f])

        return (grad_F,grad_b) #For testing our implementation via finitie differences


    def info(self):
        #Conv(16,(5,5),act,padding="valid")
        act_name=self.act.__name__

        return f'({self.n_filters},({self.k},{self.k}),{act_name},"{self.pad_mode}")'
