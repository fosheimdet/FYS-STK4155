import numpy as np



def noAct(x):
    return x
def derNoAct(x):
    return 1
noActL = [noAct,derNoAct]

def tanh(x):
    return np.tanh(x)

def derTanh(x):
    return 1-np.tanh(x)**2

tanhL = [tanh,derTanh]

def sigm(k):
    # return np.exp(k)/(1+np.exp(k))
    return 1/(1+np.exp(-k))

def derSigm(a):
    return a-a*a

sigmoidL = [sigm,derSigm]

def relu(k):
    return np.maximum(0,k) #If k is a vector or a matrix, the elementwise maximum will be taken


def derRelu(a):
    return np.heaviside(a,0) #Here the second parameter specifies what to return when the first argument is 0

reluL = [relu,derRelu]

def softmax(A):
    sm = A
    exp = np.exp(A)
    for i in range(A.shape[1]):
        sm[:,i] =exp[:,i]/np.sum(exp[:,i])
    return sm

def derSoftmax(a):
    return a*(1-a)   #We can set the delta equal to one, since we are only going to be differentiating activations w.r.t. their own activations functions

softmaxL = [softmax,derSoftmax]

def derCrossEntropy(AL,y):
    return -y*(1/AL)

def derMSE(a,y):
        return (a-y)
