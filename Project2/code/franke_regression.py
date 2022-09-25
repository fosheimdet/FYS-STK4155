import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from imageio import imread

from functions import FrankeFunction, addNoise, getMSE, getR2, desMat
from activation_functions import sigmoidL,noActL,tanhL, reluL,softmaxL, derCrossEntropy, derMSE
from neural_network import FFNN,FFNN2

# noAct = [noAct,derNoAct]
# tanh = [tanh, derTanh]
# sigmoid = [sigm,derSigm] #Create list containing the sigmoid activation function and its derivative
# relu = [relu, derRelu]
# softmax = [softmax,derSoftmax]

n = 20 #ticks on x- and y-axis
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
xx,yy = np.meshgrid(x,y)
Z = FrankeFunction(xx,yy)
z = Z.ravel()
print(z)

sigma = 0.1
z_noisy = addNoise(z,sigma).reshape(-1,1)



# terrain = imread('SRTM_data_Norway_1.tif')
# terrain = terrain[:n,:n]
# x = np.linspace(0,1, np.shape(terrain)[0])
# y = np.linspace(0,1, np.shape(terrain)[1])
# xx, yy = np.meshgrid(x,y)           # Creates mesh of image pixels
# xr, yr = np.ravel(xx), np.ravel(yy)
# Z = terrain
# z_noisy = np.ravel(Z)#Has inherent noise

#
xr, yr = np.ravel(xx), np.ravel(yy)

def designMat(xr,yr,scaled):
    X = np.ones((2,len(xr)))
    X[0,:] = xr
    X[1,:] = yr
    if(scaled):
        XT= X.T
        scaler = StandardScaler()
        scaler.fit(XT)
        XTscaled = scaler.transform(XT)
        return XTscaled
    else:
        return X.T


X = designMat(xr,yr,False)
print(X.shape)
print(z_noisy.shape)

X_train,X_test, z_train, z_test = train_test_split(X,z_noisy,train_size = 0.8)


#Uncomment to use FFNN2 instead of FFNN2
#-------------------------------------
# X = X.T
# z_noisy = z_noisy.reshape(-1,1)
#-------------------------------------
n0 = 2 #Number of input nodes
nhidden = int(input("Please enter the number of hidden layers \n"))
lengths = [n0]
for i in range(nhidden):
    lengths.append(int(input(f"Please state the number of neurons in layer {i+1} \n")))
nL = 1         #Number of output nodes
lengths.append(nL)
#y = z_noisy.reshape(1,-1)


epoch_vals= [20,50,80,100,150,200,250,300]
M_vals = [5,10,20,30,50,80,100,150]
eta_vals = [0.1,0.05,0.01,0.005, 0.001, 1e-4]
lmbd_vals = [1e-5,1e-4,1e-3,1e-2,1e-1,0]

valDict = {'epochs': epoch_vals,'batch size': M_vals, 'eta': eta_vals, 'lambda': lmbd_vals} #Dict containing values we want to loop over

iterableStr = ['epochs','batch size','eta','lambda']
            #       0        1         2       3
itIndices=[2,3]

iterable1 = valDict[iterableStr[itIndices[0]]]
iterable2 = valDict[iterableStr[itIndices[1]]]

MSE = np.zeros((len(iterable2), len(iterable1)))
R2 = np.zeros((len(iterable2), len(iterable1)))

print("z_noisy shape: ", z_noisy.shape)

#Default values
epochs = 250
M = 10
eta = 0.01
lmbd = 0
hyperparams =[epochs,M,eta,lmbd]

for i, it2 in enumerate(iterable2):
    for j, it1 in enumerate(iterable1):
        hyperparams[itIndices[0]] = it1
        hyperparams[itIndices[1]] = it2

        MLP = FFNN2(X_train, z_train, nhidden, lengths,sigmoidL,noActL,derMSE,False,hyperparams)
        MLP.initializeNetwork()
        MLP.train()
        MLP.feedForward(X_test)
        MLP.displayNetwork()
        ztilde = MLP.output
        MSE[i,j] = getMSE(z_test,ztilde)
        R2[i,j] = getR2(z_test,ztilde)

MLP.displayNetwork(True)

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(MSE,xticklabels = iterable1, yticklabels = iterable2 , annot=True, ax=ax, cmap="viridis")
ax.set_title("MSE_test using MLP on Franke function")
ax.set_xlabel(iterableStr[itIndices[0]])
ax.set_ylabel(iterableStr[itIndices[1]])
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(R2,xticklabels = iterable1, yticklabels =iterable2 , annot=True, ax=ax, cmap="viridis")
ax.set_title("R2_test using MLP on Franke function")
ax.set_xlabel(iterableStr[itIndices[0]])
ax.set_ylabel(iterableStr[itIndices[1]])
plt.show()

Ztilde = ztilde.reshape(n,n)



fig = plt.figure()
ax = fig.gca(projection='3d')

# surf = ax.plot_surface(xx,yy,(addNoise(z,sigma).reshape(n,n)), cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
plt.show()
