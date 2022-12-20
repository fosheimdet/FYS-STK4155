import autograd.numpy as np
import numpy as np
# To do elementwise differentiation:
from autograd import elementwise_grad as egrad


#No activation function
def noAct(x):
    return x
def derNoAct(x):
    return 1
noActL = [noAct,derNoAct]

#The hyperbolic tangent
def tanh(x):
    return np.tanh(x)

def derTanh(x):
    return 1-np.tanh(x)**2

tanhL = [tanh,derTanh]

#The Sigmoid function
def sigm(x):
    return 1/(1+np.exp(-x)) #Doesn't overflow for large negative x

# def sigm(x):
#     if x>0:
#         return np.exp(x)/(1+np.exp(x)) #Doesn't overflow for large positive x
#     else:
#         return 1/(1+np.exp(-x)) #Doesn't overflow for large negative x

def derSigm(a):
    return a-a*a

sigmoidL = [sigm,derSigm]

# def sigm(x):
#     if x>0:
#         return np.exp(x)/(1+np.exp(x)) #Doesn't overflow for large positive x
#     else:
#         return 1/(1+np.exp(-x)) #Doesn't overflow for large negative x


#The Relu function
def relu(x):
    return np.maximum(0,x) #If k is a vector or a matrix, the elementwise maximum will be taken

def derRelu(x):
    return np.heaviside(x,0) #Here the second parameter specifies what to return when the first argument is 0
    #return np.where(x <= 0, 0, 1)
reluL = [relu,derRelu]

# def ReLU(x):
#     return x * (x > 0)
#
# def dReLU(x):
#     return 1. * (x > 0)
# reluL = [ReLU,dReLU]

#Leaky relu
def leakyRelu(x):
    return np.where(x < 0, 0.01*x, x) #Return 0.01*x if x<0, x otherwise

def derLeakyRelu(x):
    return np.where(x<0, 0.01, x)

leakyReluL = [leakyRelu, derLeakyRelu]


#The softmax function
def softmax(A):
    exp_term = np.exp(A)
    probabilities = exp_term/np.sum(exp_term, axis=1, keepdims = True)
    return probabilities

#The derivative of the softmax
def derSoftmax(a):
    return a*(1-a)   #We can set the delta equal to one, since we are only going to be differentiating activations w.r.t. their own z
    # f_grad = egrad(softmax)
    # return f_grad(a)
softmaxL = [softmax,derSoftmax]



def derCrossEntropy(AL,y):
    return -y*(1/AL)

def derMSE(a,y):
        return 2*(a-y)/y.shape[0]
