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

    def initialize(self,shape_prev): #Construct the weights and biases based on the shape of the input
        n_features = shape_prev[0]
        self.W = np.random.normal(0,1,(n_features,self.n_l)) #Initialize using normal distribution
        self.b = np.repeat(0.01,self.n_l) #Will be automatically shaped to appropriately

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

    def show(self):
        print("--------W-------")
        print(self.W)
        # print("--------A-------")
        # print(self.A)




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


    def initialize(self,shape_prev):
        self.shape_prev = shape_prev
        self.K = np.random.normal(0,1,(2*self.r+1,2*self.r+1))
        self.B = 0.01

        print("random int: ",np.random.randint(10))

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

        error = np.zeros(self.A.shape)
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

        self.K -= eta*DK #+ lmbd*self.K)


        # self.W -= eta*(A_p.T@self.delta +lmbd*self.W)
        # self.b -= eta*(np.ones(self.delta.shape[0])@self.delta + lmbd*self.b)



class Flatten:
    def __init__(self):
        self.act = np

        self.shape = None
        self.shape_prev = None
        self.A = None
        self.delta = None


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










#
#
# class Dense:
#     def __init__(self,n_l,actL):
#         self.n_l = n_l            #Number of nodes in this layer
#         self.act = actL[0]        #Activation function of this layers
#         self.d_act = actL[1]      #Its derivative
#
#
#         # self.n_samples = None   #Number of samples in our batch
#         # self.n_features = None  #Number of features/nodes in previous layer
#         self.shape = None       #Shape of this layer. Depends on number of samples in our batch
#         self.W = None           #Weights for this layer
#         self.b = None           #Biases for this layer
#         self.A = None
#         self.delta = None
#
#     def initialize(self,shape_prev): #Construct the weights and biases based on the shape of the input
#         n_samples = shape_prev[0]
#         n_features = shape_prev[1]
#         self.W = np.random.normal(0,1,(n_features,self.n_l)) #Initialize using normal distribution
#         self.b = np.repeat(0.01,self.n_l) #Will be automatically shaped to appropriately
#
#         self.shape = (n_samples,self.n_l)
#
#     def feedForward(self, input): # Here input will be A^{l-1}
#         self.A = self.act(input@self.W + self.b)
#         return self.A
#
#     def backpropagate(self, input): #input = delta^{l+1} @ W^{l+1}.T
#         self.delta = self.d_act(self.A)*input
#         return self.delta@self.W.T         #Passing only delta would require that the next layer aquire this layers weights to calculate its delta
#
#     def update(self, eta, A_p):
#         self.W -= eta*A_p.T@self.delta
#         # print(self.delta.shape).reshape(-1,1)
#         # print(self.n_l)
#         self.b -= eta*np.ones(self.delta.shape[0])@self.delta
#
#     def show(self):
#         print("--------W-------")
#         print(self.W)
#         # print("--------A-------")
#         # print(self.A)








# def printNet():
#     print("--------A--------")
#     for l in range(0,len(layers)):
#         print("l=",l, layers[l].A.shape)
#     print("-----------------")
#
# layers = []
# layers.append(Dense(5,noActL))
# layers.append(Dense(10,sigmoidL))
# layers.append(Dense(3,softmaxL))
#
# L = len(layers)
#
# X = np.ones((3,3))
# y = np.array([[0,0,1], [0,1,0],[1,0,0]])
#
# ##-------------Initialize weights and biases------
# layers[0].initialize(X.shape)
# for l in range(1,L):
#     layers[l].initialize(layers[l-1].shape)
#
#
# ##--------------Feedforward---------------------
# layers[0].feedForward(X)
# for l in range(1,L):
#     layers[l].feedForward(layers[l-1].A)
#     # print("l = ", l, layers[l].A.shape)
#
# layers[L-1].show()
#
# printNet()
#
# ##----------------Backpropagation--------------------------
#
# eta = 0.01
#
# AL = layers[L-1].A
# delCdelY = -y*(1/AL)
#
#
# error = delCdelY
# # print("layer: ",L-1,delCdelY.shape)
# print("A[L-1] shape: ",layers[L-1].A.shape)
# print("errror shape: ", error.shape)
# layers[L-1].backpropagate(error)
# print("hey")
# for l in range(L-1,-1,-1):
#     print(l)
#     error_next = layers[l].backpropagate(error) #Here error(l) = delta(l+1)@W^T(l+1).
#     error = error_next
#     print("layer: ", l, layers[l].delta.shape)
#
#
# ##------------Update weights and biases ---------------------------------
# layers[0].update(eta,X)
# layers[0].show()
# for l in range(1,L):
#     layers[l].update(eta,layers[l-1].A)
#     layers[l].show()


















# delCdelY
# layers[L-1].backpropagate(delCdelY)
# for l in range(L-2,-1,-1):
#     layers[l].backpropagate(layers[l+1].delta@layers[l+1].W.T)
#     print("layer: ", l, layers[l].delta.shape)






# class Dense:
#     def __init__(self,input,n_l,actL):
#         self.n_l = n_l            #Number of nodes in this layer
#         self.act = actL[0]        #Activation function of this layers
#         self.d_act = actL[1]      #Its derivative
#
#         self.input = input        #Input to layer. A^{l-1} if forward, delta^{l+1} if backward propagation
#
#         self.W = np.random.normal(0,1,(input.shape[1],n_l)) #Initialize using normal distribution
#         self.b = np.repeat(0.01,n_l) #Will be automatically shaped to appropriately
#
#         self.A = None
#         self.delta = None
#
#         def feedForward(self):
#             self.A = act(self.input@self.W + self.b)
#             return self.A
#
#         def backpropagate(self):
#             self.delta = d_act(A)*input  #input = delta^{l+1} @ W^{l+1}.T
#             return self.delta@self.W.T
#
#         def update(self):
#             self.W -= self.eta*self.input.T@self.delta
