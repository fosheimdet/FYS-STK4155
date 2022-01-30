
import numpy as np
import random
from sklearn import datasets


def sigm(k):
    return np.exp(k)/(1+np.exp(k))

def derSigm(a):
    return a-a*a

def softmax(aL,n):
    aLp1 = np.zeros((n,1))
    denom = 0
    for two in range(n):
        denom = denom + np.exp(Theta2[two,1:]@aL+Theta2[two,0])

    for two in range(n):
        aLp1[two] = np.exp(Theta2[two,1:]@aL+Theta2[two,0])/denom
    return aLp1


def delCdelTheta1(p,q,ytilde,k):
    Delta1 = 0
    for j in range(0,n2):
        Delta1 = (ytilde[j]-y[j,k])*derSigm(ytilde[j])*Theta2[:,1:][j][p]*derSigm(a1[p])*phi1[q]
         #Delta1 = (ytilde[j]-y[j,k])*derSigm(ytilde[j])
    return Delta1

def delCdelTheta2(p,q,ytilde,k):
    Delta2 = 0
    for j in range(0,n2):
        Delta2 = (ytilde[j]-y[j,k])*derSigm(ytilde[j])*phi2[q]
    return Delta2

#Training data

digits = datasets.load_digits()
images = digits.images    #Array of dimension 3 containing all the images in greyscale values
labels = digits.target

#Constructing design matrix. Pixels of the images are stored as column vectors
X = np.zeros((64,len(images)))

for i in range(0,len(images)):
    X[:,i] = np.ravel(digits.images[i])


n0 = X.shape[0] #Number of input nodes
n1 = 50         #Number of nodes in hidden layer
n2 = 10         #Number of output nodes

#One-hot encode the labels
y = np.zeros((10,X.shape[1]))
for i in range(0,X.shape[1]):
    for j in range(0,n2):
        if(j==labels[i]):
            y[j,i] = 1


#Initializing parameters
np.random.seed(0) #To always get the same results
W1 = np.random.normal(0,1,(n1,n0)) #Matrix
b1 = np.random.normal(0,1,(n1,1)) #Column vector
W2 = np.random.normal(0,1,(n2,n1))
b2 = np.random.normal(0,1,(n2,1))



Theta1 = np.concatenate((b1,W1),1)
Theta2 = np.concatenate((b2,W2),1)
print("Theta1: ")
print(Theta1)
print("Theta2: ")
print(Theta2)


eta = 1
a0 = np.ones((2,1))
a1= np.ones((2,1))
a2 = np.ones(1)
indices = np.arange(X.shape[0])
M = 20

# b3 = np.array([b2])
for i in range(0,20): #epochs
    minibatch = np.random.choice(indices,M) #Choose SGD minibatch
    for k in minibatch:

        #------------------------------------------------------------
        #----------------------Feed-forward--------------------------
        a0 = X[:,k]
        a1 = sigm(Theta1[:,0]+Theta1[:,1:]@a0)
        a2 = softmax(a1,n2)
        ytilde = a2

        phi1 = np.insert(a0,0,1,0) #Insert 1 at the beginning of a0
        phi2 = np.insert(a1,0,1,0)

        #------------------------------------------------------------
        #----------------------back-propagation----------------------
        #Adjusting parameters
        for p in range(0,Theta1.shape[0]):
            for q in range(0,Theta1.shape[1]):
                Theta1[p,q] = Theta1[p,q] -eta*delCdelTheta1(p,q,ytilde,k)


        for q in range(0,Theta2.shape[0]):
                Theta2[q] = Theta2[q] - eta*delCdelTheta2(p,q,ytilde,k)
        #------------------------------------------------------------

#------------------------------------------------------------
#---------------testing the network--------------------------

a0 = X[:,0]
a1 = sigm(Theta1[:,0]+Theta1[:,1:]@a0)
a2 = softmax(a1,n2)
ytilde = a2
print("target: ", labels[0])
print("ytilde: ", ytilde)

a0 = X[:,5]
a1 = sigm(Theta1[:,0]+Theta1[:,1:]@a0)
a2 = softmax(a1,n2)
ytilde = a2
print("target: ", labels[5])
print("ytilde: ", ytilde)

a0 = X[:,9]
a1 = sigm(Theta1[:,0]+Theta1[:,1:]@a0)
a2 = softmax(a1,n2)
ytilde = a2
print("target: ", labels[9])
print("ytilde: ", ytilde)



# a0 = X[0,:].reshape(2,1)
# a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
# a2 = sigm(Theta2[0]+Theta2[1:]@a1)
# ytilde = a2
# print("ytilde: ", ytilde)


Theta1 = 0
Theta2 = 0
#
#
# for p in range(0,Theta1.shape[0]):
#     for q in range(0,Theta1.shape[1]):
#         Theta1[p,q] = Theta1[p,q] -eta*delCdelTheta1(p,q,ytilde,0)
#
# print("Theta1 after: ")
# print(Theta1)
#
# for q in range(0,Theta2.shape[0]):
#         Theta2[q] = Theta2[q] - eta*delCdelTheta2(p,q,ytilde,0)
#
# print("Theta2 after: ")
# print(Theta2)
#
# a0 = X[1,:].reshape(2,1)
# a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
# a2 = sigm(Theta2[0]+Theta2[1:]@a1)
# ytilde = a2
#
# print("ytilde: ", ytilde)

# print("Theta2[0]: ", Theta2[0])
# print("Theta2[1:]: ", Theta2[1:])
# print("Theta2[1:]@a1:", Theta2[1:]@a1)
# print(Theta1[:,0])
# print(Theta1[:,1:]@a0)




# def Del2(p,q):
#     delCdely(ytilde)*derSigm(ytilde)*phi2[q]s
