import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from functions import FrankeFunction,desMat,addNoise

from SGD import SGD






def main():


    n = 40 #Number of ticks on x and y-axes
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)

    xx,yy = np.meshgrid(x,y)
    Z = FrankeFunction(xx,yy)
    z = np.ravel(Z)
    z = addNoise(z,0.2)
    xr,yr = np.ravel(xx),np.ravel(yy)
    X = desMat(xr,yr,4)

    M = 20       #Minibatch size
    # epochL =[i for i in range(1,100,5)]
    epochs = 100
    X_train,X_test,z_train,z_test  = train_test_split(X,z,test_size=0.2)

    MSE,R2,betas = SGD(X_train,X_test,z_train,z_test, M, epochs, gamma=0, lmd=0)

    epochArr = np.arange(0,epochs)

    fig0 = plt.figure()
    plt.plot(epochArr,MSE)
    plt.xlabel("epoch")
    plt.ylabel("MSE")

    fig1 = plt.figure()
    plt.plot(epochArr,R2)
    plt.xlabel("epoch")
    plt.ylabel("R2-score")

    z_fitted = X@betas
    Z_fitted = z_fitted.reshape(n,n)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,Z_fitted, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
    plt.show()







# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# plt.show()














if __name__ =="__main__":
    main()
