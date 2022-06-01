import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import shuffle_in_unison,FrankeFunction,addNoise,desMat,getR2,getMSE





t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)


def SGD(X_train,X_test,z_train,z_test, M, epochs,gamma = 0, lmd=0):
    np.random.seed(420)
    MSE = []
    R2 = []
    nbeta = X_train.shape[1] #Polynomial degree
    betas = np.random.normal(0,1,(nbeta,1))
    N = X_train.shape[0] #Total number of datapoints in training sample
    m = int(N/M) #Number of minibatches

    v = 0 #momentum term
    # eta = 0.01

    for e in range(epochs):
        X_train, z_train = shuffle_in_unison(X_train,z_train)
        for i in range(m):
            random_index = M*np.random.randint(m)  #Up to but not including m
            xi = X_train[random_index:random_index + M]
            zi = z_train[random_index:random_index + M]
            gradients = (2.0/M)* xi.T @ ((xi @ betas)-zi.reshape(-1,1)) + 2*lmd*betas

            eta = learning_schedule(e*m+i)
            v = gamma*v + eta*gradients
            betas -= v
            # betas = betas - eta*gradients #sum horizontally

        ztilde = X_test@betas

        # Ztilde = ztilde.reshape(n,n)
        MSE.append(getMSE(z_test,ztilde))
        R2.append(getR2(z_test,ztilde))

    return MSE, R2, betas



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# plt.show()
