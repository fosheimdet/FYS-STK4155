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
        self.shape_prev = shape_prev
        Height, Width = shape_prev

        self.shape = (Height*Width,)
        return self.shape

    def feedForward(self,input):
        n_samples,Height,Width = input.shape
        A = np.zeros((n_samples,Height*Width))

        for i in range(n_samples):
            A[i,:] = np.ravel(input[i,:])
        self.A = A
        return self.A

    def backpropagate(self,input):
        self.delta = np.zeros((input.shape[0],self.shape_prev[0],self.shape_prev[1]))
        for i in range(input.shape[0]):
            self.delta[i,:,:] = input[i,:].reshape(self.shape_prev)

        return self.delta

    def update(self,eta,lmbd,A_p):
        return None
