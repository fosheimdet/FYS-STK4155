import numpy as np


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
def sigm(k):
    # return np.exp(k)/(1+np.exp(k))
    # k = np.clip(k,-500,500)
    return 1/(1+np.exp(-k))

def derSigm(a):
    return a-a*a

sigmoidL = [sigm,derSigm]

#The Relu function
def relu(k):
    return np.maximum(0,k) #If k is a vector or a matrix, the elementwise maximum will be taken


def derRelu(a):
    return np.heaviside(a,0) #Here the second parameter specifies what to return when the first argument is 0

reluL = [relu,derRelu]

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

softmaxL = [softmax,derSoftmax]

def derCrossEntropy(AL,y):
    return -y*(1/AL)

def derMSE(a,y):
        return (a-y)
