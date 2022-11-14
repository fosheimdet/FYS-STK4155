import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from imageio import imread
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

from functions import FrankeFunction, addNoise, getMSE, getR2, desMat, gridsearch, scale_data
from activation_functions import sigmoidL,noActL,tanhL, reluL,softmaxL, derCrossEntropy, derMSE
from neural_network import FFNN,FFNN2



np.random.seed(10)

n = 20 #ticks on x- and y-axis
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
xx,yy = np.meshgrid(x,y)
Z = FrankeFunction(xx,yy)
z = Z.ravel()


sigma = 0.1
z_noisy = addNoise(z,sigma).reshape(-1,1)

xr, yr = np.ravel(xx), np.ravel(yy)

def designMat(xr,yr,scaled):
    X = np.ones((len(xr),2))
    X[:,0] = xr
    X[:,1] = yr
    if(scaled):
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        return X_scaled
    else:
        return X


X = designMat(xr,yr,True)


X_train,X_test, z_train, z_test = train_test_split(X,z_noisy,train_size = 0.8)
# X_train, X_test = scale_data(X_train,X_test)


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

MSE2 = np.zeros((len(iterable2), len(iterable1)))
R22 = np.zeros((len(iterable2), len(iterable1)))

print("z_noisy shape: ", z_noisy.shape)

#Default values
epochs = 200
M = 10
eta = 0.005
lmbd = 1e-05
hyperparams =[epochs,M,eta,lmbd]
print("X_train shape", X_train.shape)
print("z_train shape", z_train.shape)

##===================================================================================
##                         Heatmap
##===================================================================================
MSE_min = 1e4
it1_opt, it2_opt = 0,0
R2_max = -1e4

# networks = np.zeros( (len(iterable2),len(iterable1)), dtype=object)

MLP_opt = -1e8*np.zeros(1,dtype=object)


for i, it2 in enumerate(iterable2):
    for j, it1 in enumerate(iterable1):
        hyperparams[itIndices[0]] = it1
        hyperparams[itIndices[1]] = it2

        MLP = FFNN(X_train, z_train, lengths,sigmoidL,noActL,derMSE,False,hyperparams)
        MLP.initializeNetwork()
        MLP.train()
        MLP.feedForward(X_test)
        # MLP.displayNetwork()
        ztilde = MLP.output
        assert(ztilde.shape == z_test.shape)
        MSE[i,j] = getMSE(z_test,ztilde)
        R2[i,j] = getR2(z_test,ztilde)
        if(MSE[i,j]<MSE_min):
            MSE_min = MSE[i,j]
            it1_opt,it2_opt = it1,it2
            MLP_opt = MLP



# MLP.displayNetwork(True)

# MSE_train, MSE_test, it1_opt,it2_opt =gridsearch(itIndices, hyperparams, True)

print("optimal parameters: ", iterableStr[itIndices[0]] + f"={it1_opt}", iterableStr[itIndices[1]] + f"={it2_opt}")


itStr_trimmed = [x for x in range(len(iterableStr))]
itStr_trimmed.remove(itIndices[0])
itStr_trimmed.remove(itIndices[1])


fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(MSE,xticklabels = iterable1, yticklabels = iterable2 , annot=True, ax=ax, cmap="viridis")
ax.set_title("MSE$_{test}$ using our NN on the Franke function."+
f" {iterableStr[itStr_trimmed[0]]}={hyperparams[itStr_trimmed[0]]}, {iterableStr[itStr_trimmed[1]]}={hyperparams[itStr_trimmed[1]]} ")
ax.set_xlabel(iterableStr[itIndices[0]])
ax.set_ylabel(iterableStr[itIndices[1]])

# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(R2,xticklabels = iterable1, yticklabels = iterable2 , annot=True, ax=ax, cmap="viridis")
# ax.set_title("R2$_{test}$ using our NN on the Franke function."+
# f" {iterableStr[itStr_trimmed[0]]}={hyperparams[itStr_trimmed[0]]}, {iterableStr[itStr_trimmed[1]]}={hyperparams[itStr_trimmed[1]]} ")
# ax.set_xlabel(iterableStr[itIndices[0]])
# ax.set_ylabel(iterableStr[itIndices[1]])


#Set the parameters of the grid search to their optimal values
hyperparams[itIndices[0]] = it1_opt
hyperparams[itIndices[1]] = it2_opt

epochs,M,eta,lmbd = hyperparams

##===================================================================================
##===================================================================================


# Using our own NN
##===================================================================================
# MLP = FFNN(X_train, z_train, nhidden, lengths,sigmoidL,noActL,derMSE,False,hyperparams)
# MLP.initializeNetwork()
# MLP.train()
MLP_opt.feedForward(X)

ztilde = MLP_opt.output
Ztilde = ztilde.reshape(n,n)

##===================================================================================
# Using sklearn
##===================================================================================
MLP_sk = MLPRegressor(hidden_layer_sizes=lengths[1:-1], activation='logistic', solver='sgd',
alpha=0,batch_size=M, learning_rate_init=eta, max_iter=epochs, momentum=0,
tol = 0, nesterovs_momentum=False).fit(X_train,np.ravel(z_train))

ztilde_sk = MLP_sk.predict(X)
Ztilde_sk = ztilde_sk.reshape(n,n)
##===================================================================================


# Using Keras
##===================================================================================
model = Sequential()
for i in range(len(lengths)-2):
    if(i==0):
        model.add(Dense(units=lengths[i+1], input_dim=lengths[0], activation = 'sigmoid'))
    else:
        model.add(Dense(units=lengths[i+1], activation = 'sigmoid'))

model.add(Dense(units=1))
model.compile(loss='mean_squared_error',
              optimizer='sgd')
model.fit(X_train, z_train, epochs=epochs, batch_size=M, verbose=0)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)
ztilde_keras = model.predict(X)
Ztilde_keras = ztilde_keras.reshape(n,n)
##===================================================================================

cmapStr = 'viridis'
# cmapStr = 'magma'
# cmapStr = 'icefire_r'
azimuth = 20
theta = 30
Lwidth = 0.3
alpha_value = 0.9

fig= plt.figure(figsize=(8,4))
#First plot
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(xx,yy,Z, cmap='viridis', linewidth=Lwidth, alpha =alpha_value, edgecolor = 'k')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.view_init(azimuth, theta)
ax.set_title('Franke function')
# Add a color bar which maps values to colors
#fig.colorbar(surf, shrink=0.5, aspect = 5)
#Second plot
ax = fig.add_subplot(122, projection='3d')
#ax.scatter(xs = heights, ys = weights, zs = ages)
surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=Lwidth, alpha = alpha_value, edgecolor = 'k')
ax.set_title(f'Our NN with hidden layers = {lengths[1:-1]}')



fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
plt.title("Our own NN")


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx,yy,Ztilde_sk, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
plt.title("sklearn")

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx,yy,Ztilde_keras, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
plt.title("keras")


fig = plt.figure()
ax = fig.gca(projection='3d')
# surf = ax.plot_surface(xx,yy,(addNoise(z,sigma).reshape(n,n)), cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
surf = ax.plot_surface(xx,yy,Z, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
plt.title("Original")





# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(R22, xticklabels = iterable1, yticklabels =iterable2 , annot=True, ax=ax, cmap="viridis")
# ax.set_title("R2_test (FFNN2) using MLP on Franke function")
# ax.set_xlabel(iterableStr[itIndices[0]])
# ax.set_ylabel(iterableStr[itIndices[1]])


#
# MLP_opt.feedForward(X)
# ztilde = MLP_opt.output
# Ztilde = ztilde.reshape(n,n)
#
# epochs,M,eta,lmbd = hyperparams
# MLP_sk = MLPRegressor(hidden_layer_sizes=lengths[1:-1], activation='logistic', solver='sgd',
# batch_size=M, learning_rate_init=eta, max_iter=epochs, momentum=0, nesterovs_momentum=False).fit(X_train,np.ravel(z_train))
# ztilde_sk = MLP_sk.predict(X)
# Ztilde_sk = ztilde_sk.reshape(n,n)
#
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # surf = ax.plot_surface(xx,yy,(addNoise(z,sigma).reshape(n,n)), cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# surf = ax.plot_surface(xx,yy,Ztilde, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # surf = ax.plot_surface(xx,yy,(addNoise(z,sigma).reshape(n,n)), cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# surf = ax.plot_surface(xx,yy,Ztilde_sk, cmap = 'viridis',linewidth=0.1, alpha = 0.9, edgecolor = 'k')
# plt.title("sklearn")

plt.show()
