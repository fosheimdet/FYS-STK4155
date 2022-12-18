import numpy as np



class Flatten:
    def __init__(self):
        self.act = np

        self.shape = None
        self.shape_prev = None
        self.A = None
        self.delta = None
        self.n_param = 0

    def initialize(self, shape_prev):
        self.shape_prev = shape_prev #Need to remember input shape for reconstruction during backpropgation
        Height, Width, n_channels = shape_prev
        self.shape = (Height*Width*n_channels,)
        #Just the number of nodes from in one channel. Will be multiplied by the number of
        #channels during propagation.
        return self.shape

    def feedForward(self,input):
        n_samples,Height,Width,n_channels= input.shape
        A = np.zeros((n_samples,Height*Width*n_channels))

        for n in range(n_samples):
            for c in range(n_channels):
                A[n,c*Height*Width:(c+1)*Height*Width] = np.ravel(input[n,:,:,c])
            # A[n,:] = np.ravel(input[n,:])
        self.A = A
        return self.A

    def backpropagate(self,input):
        n_samples = input.shape[0]
        Height_p,Width_p,n_channels = self.shape_prev
        # self.delta = np.zeros((input.shape[0],self.shape_prev[0],self.shape_prev[1]))
        self.delta = np.zeros((n_samples,)+self.shape_prev)
        for n in range(n_samples):
            for c in range(n_channels):
                self.delta[n,:,:,c] = input[n,c*Height_p*Width_p:(c+1)*Height_p*Width_p].reshape(Height_p,Width_p)
            # self.delta[i,:,:] = input[i,:].reshape(self.shape_prev)
        # print("delta<flat>:",self.delta)
        # print("delta_shape<flat>:", self.delta.shape)
        return self.delta

    def update(self,eta,lmbd,A_p):
        return None



    # def backpropagate(self,input):
    #     # self.delta = np.zeros((input.shape[0],self.shape_prev[0],self.shape_prev[1]))
    #     self.delta = np.zeros((input.shape[0],)+self.shape_prev)
    #     for i in range(input.shape[0]):
    #         self.delta[i,:,:] = input[i,:].reshape(self.shape_prev)
    #
    #     return self.delta
    #
    # def update(self,eta,lmbd,A_p):
    #     return None
