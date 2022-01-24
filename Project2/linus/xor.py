
import numpy as np
import random



def sigm(k):
    return np.exp(k)/(1+np.exp(k))

def derSigm(a):
    return a-a*a

#Training data
y = np.array([0,1,1,0])
X = np.array([[0,0], [0,1], [1,0], [1,1]])

#Forward propagation

x = np.array([[0],[0]]) #Column vector
x = X[0,:].reshape(2,1)

W1 = np.random.normal(0,1,(2,2)) #Matrix
b1 = np.random.normal(0,1,(2,1)) #Column vector
b2 = np.random.normal(0,1)       #Scalar
W2 = np.random.normal(0,1,(1,2)) #Row vector

a0 = x
a1 = sigm(b1+W1@a0)
a2 = sigm(b2+W2@a1)

ytilde=a2

print(ytilde)

#Backpropagation


phi1 = np.insert(a1,0,1,0) #Insert 1 at the beginning of a1
phi2 = np.insert(a2,0,1,0)

Theta1 = np.concatenate((b1,W1),1)
Theta2 = np.concatenate((b2,W2),1)
print(Theta1)
print(Theta2)



def delCdely(ytilde,k):
    return ytilde-y[k]

# def Del2(p,q):
#     delCdely(ytilde)*derSigm(ytilde)*phi2[q]s
