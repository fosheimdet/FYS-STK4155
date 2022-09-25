import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import seaborn as sns

from functions import shuffle_in_unison, FrankeFunction, desMat, getR2, getMSE, addNoise

def step_func(t, t0=1.0, t1=10):
    """
    Adaptive method for learning rate.
    """
    return t0/(t+t1)


def SGD(betas,hyperparams, adaptive):
    epochs,M,eta,lmbd,gamma, t0, t1 = hyperparams
    v = 0
    # MSE = []
    # R2 = []
    for e in range(epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X_train[random_index:random_index+M]
            zi = z_train[random_index:random_index+M]

            gradC = (2/M)*Xi.T@(Xi@betas-zi) + 2*lmbd*betas

            if(adaptive):
                eta = step_func(e*m+i)

            v =gamma*v + eta*gradC

            betas = betas -v

    ztilde = X_test@betas
    MSE = getMSE(z_test,ztilde)
    R2 = getR2(z_test,ztilde)
    # MSE.append(getMSE(z_test,ztilde))
    # R2.append(getR2(z_test,ztilde))
    return MSE,R2,betas,ztilde
    # return np.average(MSE),np.average(R2),betas,ztilde


n = 40  #Number of ticks on x- and y-axes
p = 8  #Degree of intepolating polynomial

M = 20 #Size of minibatches
m = int((n*n)/M) #Number of minibatches

epochs = 100 #An epoch consists of m SGD steps

eta = 0.1 #learning rate


x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
xx, yy = np.meshgrid(x,y)
Z = FrankeFunction(xx,yy)
z = np.ravel(Z).reshape(-1,1)
z_noisy = addNoise(z,0.1).reshape(-1,1)
20
xr, yr = np.ravel(xx), np.ravel(yy)

X = desMat(xr,yr,p)

X_train, X_test, z_train, z_test = train_test_split(X,z,test_size = 0.2)

nbetas = X.shape[1] #Number of parameters of our model# eta_vals = np.logspace(-4,-1,8)
# lmbd_vals = [1e-5,1e-4,1e-3,1e-2,1e-1,0]
betas = np.random.normal(0,1,(nbetas,1))





epoch_vals = [10,30,50,70,100, 120, 200]
epoch_vals = [20*(i+1) for i in range(2,15,1)]
eta_vals = [1e-5,1e-4,1e-3,1e-2,1e-1]
eta_vals = [0.5*1e-4,1e-4,0.5*1e-3,1e-3,0.5*1e-2,1e-2,0.5*1e-1,1e-1]

lmbd_vals = [1e-5,0.5*1e-4,1e-4,0.5*1e-3,1e-3,0.5*1e-2,1e-2,0.5*1e-1,1e-1,0]
#lmbd_vals = np.logspace(-5,0,12)
M_vals = [5,10,20,30,50,80,100,150]
gamma_vals = [0.2,0.3,0.5,0.7,0.8, 0.85, 0.9]
#gamma_vals = np.linspace(0.7,0.91,20)
t0 = np.linspace(0,10,11)
t1 = np.linspace(0,100,11)

# hParDict = {'epochs': epoch_vals, 'batch size': M_vals, 'eta': eta_vals, 'lambda': lmbd_vals, 'gamma': gamma_vals}
hParList = [epoch_vals, M_vals, eta_vals, lmbd_vals,  gamma_vals, t0, t1]
hParStrings = ['epochs', 'batch size', 'eta', 'lambda', 'gamma', 't0', 't1']
#                 0            1         2        3        4       5     6
itIndices = [5,6]

len1 = len(hParList[itIndices[0]])
len2 = len(hParList[itIndices[1]])

MSE = np.zeros((len1,len2))
R2 =np.zeros((len1,len2))

epochs = 280
M = 10
eta = 0.1
lmbd = 1e-4
gamma = 0.88
t0 = 5
t1 = 50
hyperparams = [epochs,M,eta,lmbd,gamma, t0, t1]

for i, val1 in enumerate(hParList[itIndices[0]]):
    for j, val2 in enumerate(hParList[itIndices[1]]):
        hyperparams[itIndices[0]] =val1
        hyperparams[itIndices[1]] =val2
        MSE[i,j], R2[i,j] = SGD(betas,hyperparams, True)[0:2]


fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(MSE,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax)
ax.set_title("MSE_test for Ridge with $\gamma = 0.88$ and adaptive learning rate")
ax.set_ylabel(hParStrings[itIndices[0]])
ax.set_xlabel(hParStrings[itIndices[1]])

#ax.set_xlabel("$\eta$")
plt.show()#
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(R2,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax)
ax.set_title("R2_test for Ridge with $\gamma = 0.88$ and adaptive learning rate")
ax.set_ylabel(hParStrings[itIndices[0]])
ax.set_xlabel(hParStrings[itIndices[1]])
#ax.set_xlabel("$\eta$")
plt.show()




#
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(R2,xticklabels = varMat[1], yticklabels = varMat[0], annot=True, ax=ax)
# ax.set_title("R2_test for Ridge using SGD without momentum")
# ax.set_ylabel("epochs")
# #ax.set_ylabel("$\lambda$")
#
# ax.set_xlabel("mini-batch size")
# #ax.set_xlabel("$\eta$")
# plt.show()
#
#
# epochs = 300
# M=10
# eta = 0.1
# #lmbd = 1e-3
# lmbd = 0
# betas = SGD(betas,epochs,M,eta, lmbd, 0)[2]
# ztilde = X@betas
#
# Ztilde = ztilde.reshape(n,n)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# #surf = ax.plot_surface(xx,yy,Z, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# plt.show()
#
#
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(R2,xticklabels = varMat[1], yticklabels = varMat[0], annot=True, ax=ax)
# ax.set_title("R2_test for Ridge using SGD without momentum")
# ax.set_ylabel("epochs")
# #ax.set_ylabel("$\lambda$")
#
# ax.set_xlabel("mini-batch size")
# #ax.set_xlabel("$\eta$")
# plt.show()
#
#
# epochs = 300
# M=10
# eta = 0.1
# #lmbd = 1e-3
# lmbd = 0
# betas = SGD(betas,epochs,M,eta, lmbd, 0)[2]
# ztilde = X@betas
#
# Ztilde = ztilde.reshape(n,n)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# #surf = ax.plot_surface(xx,yy,Z, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# plt.show()
