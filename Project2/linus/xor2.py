
import numpy as np
import random



def sigm(k):
    return np.exp(k)/(1+np.exp(k))

def derSigm(a):
    return a-a*a


def delCdelTheta1(p,q,ytilde,k,n3,n2):
    Delta1 = 0
    for three in range(0,n3):
        for two in range(0,n2):
            # Delta1 = Delta1 + (ytilde-y[k])*derSigm(ytilde)*W3[three,two]*derSigm(a2[two])*W2[two,p]*derSigm(a1[p])*phi1[q]
             Delta1 = Delta1 + (ytilde-y[k])*derSigm(ytilde)*W3[two]*derSigm(a2[two])*W2[two,p]*derSigm(a1[p])*phi1[q]
    return Delta1

def delCdelTheta2(p,q,ytilde,k):
    Delta2 = (ytilde-y[k])*derSigm(ytilde)*W3[p]*derSigm(a2[p])*phi2[q]
    return Delta2

def delCdelTheta3(p,q,ytilde,k):
    Delta3 = (ytilde-y[k])*derSigm(ytilde)*phi3[q]
    return Delta3


#Initializing parameters
np.random.seed(420) #To always get the same results

# Theta1 = np.random.normal(0,1,(2,3))
# Theta2 = np.random.normal(0,1,3)
W1 = np.random.normal(0,1,(2,2)) #Matrix
b1 = np.random.normal(0,1,(2,1)) #Column vector
W2 = np.random.normal(0,1,(2,2)) #Matrix
b2 = np.random.normal(0,1,(2,1)) #Column vector
b3 = np.random.normal(0,1)       #Scalar
b3 = np.array([b3])
W3 = np.random.normal(0,1,2) #Row vector

#Training data
y = np.array([0,1,1,0])
X = np.array([[0,0], [0,1], [1,0], [1,1]])


#Forward propagation
# x = np.array([[0],[0]]) #Column vector
# x = X[0,:].reshape(2,1)

# a0 = x
# a1 = sigm(b1+W1@a0)
# a2 = sigm(b2+W2@a1)
#
# ytilde=a2
# #-------------------------------------------------------------------------------


Theta1 = np.concatenate((b1,W1),1)
Theta2 = np.concatenate((b2,W2),1)
Theta3 = np.concatenate((b3,W3),0)
print("Theta1: ")
print(Theta1)
print("Theta2: ")
print(Theta2)


eta = 10
a0 = np.ones((2,1))
a1= np.ones((2,1))
a2 = np.ones((2,1))
a3 = np.ones(1)



for i in range(0,5000):  #Epochs
    for k in range(0,X.shape[0]):  #Iterate over training data
        #FeedForward
        a0 = X[k,:].reshape(2,1)
        a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
        a2 = sigm(Theta2[:,0].reshape(2,1)+Theta2[:,1:]@a1)
        a3 = sigm(Theta3[0]+Theta3[1:]@a2)
        ytilde = a3
        # print("ytilde: ", ytilde)

        phi1 = np.insert(a0,0,1,0) #Insert 1 at the beginning of a0
        phi2 = np.insert(a1,0,1,0)
        phi3 = np.insert(a2,0,1,0)


        #Adjusting parameters
        for p in range(0,Theta1.shape[0]):
            for q in range(0,Theta1.shape[1]):
                Theta1[p,q] = Theta1[p,q] -eta*delCdelTheta1(p,q,ytilde,k, a3.shape[0], a2.shape[0])


        for p in range(0,Theta2.shape[0]):
            for q in range(0,Theta2.shape[1]):
                Theta2[p,q] = Theta2[p,q] -eta*delCdelTheta2(p,q,ytilde,k)

        # print("Theta1 after: ")
        # print(Theta1)

        for q in range(0,Theta3.shape[0]):
                Theta3[q] = Theta3[q] - eta*delCdelTheta3(p,q,ytilde,k)

        # print("Theta2 after: ")
        # print(Theta2)

a0 = X[0,:].reshape(2,1)
a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
a2 = sigm(Theta2[:,0].reshape(2,1)+Theta2[:,1:]@a1)
a3 = sigm(Theta3[0]+Theta3[1:]@a1)
ytilde = a3
print("ytilde: ", ytilde)


a0 = X[1,:].reshape(2,1)
a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
a2 = sigm(Theta2[:,0].reshape(2,1)+Theta2[:,1:]@a1)
a3 = sigm(Theta3[0]+Theta3[1:]@a1)
ytilde = a3
print("ytilde: ", ytilde)

a0 = X[2,:].reshape(2,1)
a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
a2 = sigm(Theta2[:,0].reshape(2,1)+Theta2[:,1:]@a1)
a3 = sigm(Theta3[0]+Theta3[1:]@a1)
ytilde = a3
print("ytilde: ", ytilde)

a0 = X[3,:].reshape(2,1)
a1 = sigm(Theta1[:,0].reshape(2,1)+Theta1[:,1:]@a0)
a2 = sigm(Theta2[:,0].reshape(2,1)+Theta2[:,1:]@a1)
a3 = sigm(Theta3[0]+Theta3[1:]@a1)
ytilde = a3
print("ytilde: ", ytilde)
