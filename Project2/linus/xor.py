
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
b2 = np.array([b2])
W2 = np.random.normal(0,1,2) #Row vector

a0 = x
a1 = sigm(b1+W1@a0)
a2 = sigm(b2+W2@a1)

ytilde=a2

print("ytilde: ")
print(ytilde)

#Backpropagation


phi1 = np.insert(a0,0,1,0) #Insert 1 at the beginning of a0
phi2 = np.insert(a1,0,1,0)

Theta1 = np.concatenate((b1,W1),1)
Theta2 = np.concatenate((b2,W2),0)
print("Theta1: ")
print(Theta1)
print("Theta2: ")
print(Theta2)

print(Theta1.shape)


def delCdelTheta1(p,q,ytilde,k):
    Delta1 = (ytilde-y[k])*derSigm(ytilde)*W2[p]*derSigm(a1[p])*phi1[q]
    return Delta1

def delCdelTheta2(p,q,ytilde,k):
    Delta2 = (ytilde-y[k])*derSigm(ytilde)*phi2[q]
    return Delta2

eta = 10

# print("testing:")
# print(ytilde-y[0])
# print(derSigm(ytilde))
# print(W2)
# print(eta*delCdelTheta1(0,1,ytilde,0))

for p in range(0,Theta1.shape[0]):
    for q in range(0,Theta1.shape[1]):
        Theta1[p,q] = Theta1[p,q] -eta*delCdelTheta1(p,q,ytilde,0)

print("Theta1 after: ")
print(Theta1)

for q in range(0,Theta2.shape[0]):
        Theta2[q] = Theta2[q] - eta*delCdelTheta2(p,q,ytilde,0)

print("Theta2 after: ")
print(Theta2)

def delCdely(ytilde,k):
    return ytilde-y[k]

# def Del2(p,q):
#     delCdely(ytilde)*derSigm(ytilde)*phi2[q]s
