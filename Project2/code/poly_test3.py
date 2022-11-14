import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from functions import getMSE, getR2
import random




def desMat1D(x,p):
    N = len(x)
    X = np.ones((N,p))
    for i in range(1,p):
        X[:,i] = x**i
    return X


def step_func(t):
    return t0/(t+t1)


def gradientDescent(betas,eta,iterations,gamma=0,threshold = 1e-3):
    X,y = X_train,y_train
    N = X.shape[0] #number of datapoints
    MSE = []
    R2 = []
    v=0
    beta0 = []
    beta1 = []
    gradients = []
    print(f"betas for gamma={gamma}: ", betas)
    for i in range(iterations):
        beta0.append(betas[0])
        beta1.append(betas[1])

        cost_prev = getMSE(y,X@betas)
        gradC = (2/N)*X.T@(X@betas -y)
        v = gamma*v + eta*gradC
        betas = betas-v

        cost = getMSE(y, X@betas)

        MSE.append(getMSE(y_test, X_test@betas))
        R2.append(getR2(y_test,X_test@betas))
        gradients.append(gradC)

        if(abs(cost - cost_prev) <= threshold):
            print(f"stopped after {i} iterations")
            break

    last_iter = i #last iteration

    return  MSE, R2, betas, last_iter, beta0, beta1



#def SGD(X,y,betas,eta,epochs, M , lmbd, gamma=0, adaptive=True, threshold = 1e-4):
def SGD(betas,hyperparams, momentum = False, adaptive=True, threshold = 1e-4):
    X,y = X_train,y_train
    epochs,M,eta,lmbd,gamma, t0, t1 = hyperparams
    if(momentum == False): gamma=0
    v = 0
    MSE = []
    R2 = []
    beta0 = []
    beta1 = []
    rng = np.random.default_rng()
    cost_prev = getMSE(y,X@betas)

    for e in range(epochs):
        indices = rng.permutation(np.arange(0,len(y)))
        X,y = X[indices], y[indices]

        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradC = (2/M)*Xi.T@(Xi@betas-yi) + 2*lmbd*betas

            if(adaptive):
                eta = step_func(e*m+i)
            v =gamma*v + eta*gradC
            betas = betas -v

        ytilde = X_test@betas
        MSE.append(getMSE(y_test,ytilde))
        R2.append(getR2(y_test,ytilde))
        beta0.append(betas[0])
        beta1.append(betas[1])

        cost = getMSE(y,X@betas)
        if(abs(cost_prev-cost)<=threshold):
            break
        cost_prev = cost

    last_iter=e

    return MSE, R2,betas, last_iter


#Scaling factor for the step size for Adagrad
def adagrad_sf(s_prev, grad):
    epsilon = 1e-8
    s = s_prev + grad**2
    sf = 1/(epsilon+np.sqrt(s))
    return s, sf

#Scaling factor for the step size for RMSprop
def RMSprop_sf(s_prev,grad):
    epsilon = 1e-8
    beta = 0.9
    s = beta*s_prev + (1-beta)*grad**2
    return s, 1/(np.sqrt(epsilon+s))

#Scaling factor for the step size for Adam
def adam_sf(r_prev,s_prev,grad,t):
    epsilon = 1e-8
    beta1, beta2 = 0.9, 0.99
    r = beta1*r_prev + (1-beta1)*grad
    s = beta2*s_prev + (1-beta2)*grad**2

    r = r/(1-beta1**t)
    s = s/(1-beta2**t)
    sf = r/(np.sqrt(s)+epsilon)
    return r,s,sf


def optimize(method,betas,hyperparams,momentum=False,threshold = 1e-3):
    X,y = X_train,y_train
    epochs,M,eta,lmbd,gamma, t0, t1 = hyperparams
    if(momentum == False): gamma = 0
    n = X.shape[0]
    v = 0 #momentum
    r=0   #Adam
    s = 0

    MSE = []
    R2 = []
    t = 1 #Step counter, Adam

    rng = np.random.default_rng()
    cost_prev = getMSE(y,X@betas)

    for e in range(epochs):
        indices = rng.permutation(np.arange(0,n))
        X,y = X[indices], y[indices]
        ytilde = X_test@betas
        MSE.append(getMSE(y_test,ytilde))
        R2.append(getR2(y_test,ytilde))
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradC = (2/M)*Xi.T@(Xi@betas-yi) + 2*lmbd*betas

            if(method == "Adagrad"):
                s,sf = adagrad_sf(s, gradC)
            if(method == "RMSprop"):
                s,sf = RMSprop_sf(s,gradC)
            if(method == "Adam"):
                r,s,sf = adam_sf(r,s,gradC,t)

            v = gamma*v + sf*eta*gradC
            betas = betas - v
            t+=1

        cost = getMSE(y,X@betas)
        if(abs(cost_prev-cost)<=threshold):
            break
        cost_prev = cost

    last_iter=e

    return  MSE, R2,betas, last_iter



np.random.seed(0)
n = 100
p = 5

x = np.linspace(0,2,n)
#x = 2*np.random.rand(n)
#y = 2*x + 10*np.random.random_sample(len(x)) - 10/2
a0 = 4
a1 = 3
a2 = 6
y = a0 + a1*x + a2*x**2 + np.random.randn(n)
y = y.reshape(-1,1)

X = desMat1D(x,p)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

betas = np.random.normal(0,1,(p,1))

H = (2.0/n)* X.T @ X
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

iterations = 1000

gamma = 0.9

eta = 1.0/np.max(EigValues)
eta = 1e-1
epochs = 500
#t0, t1 = 5,50
t0, t1 = 0.05,0.5
M = 5       #Batch size
m = int(n/M)  #Number of mini-batches in an epoch
lmbd = 0
threshold = 1e-3

#Optimization methods
methods = ["GD", "SGD","Adagrad", "RMSprop", "Adam"]
#            0     1       2          3         4

method_int = 2

method= methods[method_int]

hyperparams = [epochs,M,eta,lmbd,gamma, t0, t1]

if(method == "GD"):
    MSE,R2,betas_opt,last_iter,beta0, beta1 = gradientDescent(betas,eta,iterations,0)
    MSE_m, R2_m, betas_opt_m, last_iter_m, beta0_m, beta1_m = gradientDescent(betas,eta,iterations,gamma)
elif(method == "SGD"):
    MSE,R2,betas_opt, last_iter = SGD(betas,hyperparams, False,True, threshold) #without momentum
    MSE_m, R2_m, betas_opt_m, last_iter_m = SGD(betas,hyperparams,True, True, threshold) #With momentum
else:
    MSE,R2,betas_opt,last_iter = optimize(method,betas,hyperparams,False,threshold)
    MSE_m, R2_m, betas_opt_m, last_iter_m = optimize(method,betas,hyperparams, False,threshold)

ytilde = X@betas_opt
ytilde_m = X@betas_opt_m


# epoch_vals = [20*(i+1) for i in range(2,15,1)]
# eta_vals = [1e-5,1e-4,1e-3,1e-2,1e-1]
# eta_vals = [0.5*1e-4,1e-4,0.5*1e-3,1e-3,0.5*1e-2,1e-2,0.5*1e-1,1e-1]
#
# lmbd_vals = [1e-5,0.5*1e-4,1e-4,0.5*1e-3,1e-3,0.5*1e-2,1e-2,0.5*1e-1,1e-1,0]
# M_vals = [5,10,20,30,50,80,100]
# gamma_vals = [0.2,0.3,0.5,0.7,0.8, 0.85, 0.9]
# t0_vals = np.linspace(0,10,11)
# t1_vals = np.linspace(0,100,11)
#
# hParList = [epoch_vals, M_vals, eta_vals, lmbd_vals,  gamma_vals, t0_vals, t1_vals]
# hParStrings = ['epochs', 'batch size', 'eta', 'lambda', 'gamma', 't0', 't1']
# #                 0            1         2        3        4       5     6
# itIndices = [0,1]
#
# len1 = len(hParList[itIndices[0]])
# len2 = len(hParList[itIndices[1]])
#
# #Heatmaps:
# MSE_hm = np.zeros((len1,len2))
# R2_hm =np.zeros((len1,len2))
#
# hyperparams = [epochs,M,eta,lmbd,gamma, t0, t1]
#
# #Calculating MSE and R2 values for each of the two chosen iterables
# for i, val1 in enumerate(hParList[itIndices[0]]):
#     for j, val2 in enumerate(hParList[itIndices[1]]):
#         hyperparams[itIndices[0]] =val1
#         hyperparams[itIndices[1]] =val2
#         MSE_l,R2_l = SGD(betas,hyperparams,False,True,threshold)[0:2]
#         #MSE_l,R2_l = optimize(method,betas,hyperparams,False,threshold)[0:2]
#         MSE_hm[i,j], R2_hm[i,j] = MSE_l[-1], R2_l[-1]
#
# print(R2_hm)
#
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(MSE_hm,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax)
# ax.set_title("MSE_test for Ridge with $\gamma = 0.88$ and adaptive learning rate")
# ax.set_ylabel(hParStrings[itIndices[0]])
# ax.set_xlabel(hParStrings[itIndices[1]])
#
# #ax.set_xlabel("$\eta$")
# float_formatter = "{:.3f}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})
#
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(R2_hm,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax)
# ax.set_title("R2_test for Ridge with $\gamma = 0.88$ and adaptive learning rate")
# ax.set_ylabel(hParStrings[itIndices[0]])
# ax.set_xlabel(hParStrings[itIndices[1]])
# #ax.set_xlabel("$\eta$")
# plt.show()



#======================================================
#                    Plotting
#======================================================

plt.figure()
plt.title("Linear regression with "+f"$p={p}$"+ " using "+method+" with "+ f"$\eta={eta:.3f}$")
plt.scatter(x,y, color = 'r', label = f"${a0}+{a1}x+{a2}x^2+\epsilon$")
iter_str = "epochs"
if(method == "GD"):
    iter_str = "it."
plt.plot(x,ytilde, 'b', label = f"$\gamma=0$, "+iter_str+f"$={last_iter}$, MSE={MSE[-1]:.3f}")
plt.plot(x,ytilde_m,'g', label = f"$\gamma={gamma}$, "+iter_str+f"$={last_iter_m}$, MSE={MSE_m[-1]:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.legend()

plt.figure()
plt.title("R2 score for "+method+" with and without momentum")
plt.plot(np.arange(0,last_iter+1), R2[0:],'b', label=f"$\gamma = 0$")
plt.plot(np.arange(0,last_iter_m+1), R2_m[0:],'g', label=f"$\gamma = {gamma}$")
if(method == "GD"):
    plt.xlabel("Iterations")
else:
    plt.xlabel("Epochs")
plt.ylabel("R2-score")
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.tight_layout()
# plt.arrow(300,R2[0], -300, 0)
plt.legend()

plt.figure()
plt.title("MSE score for "+method+ " with and without momentum")
plt.plot(np.arange(0,last_iter+1), MSE[0:],'b', label=f"$\gamma = 0$")
plt.plot(np.arange(0,last_iter_m+1), MSE_m[0:],'g', label=f"$\gamma = {gamma}$")
if(method == "GD"):
    plt.xlabel("Iterations")
else:
    plt.xlabel("Epochs")
plt.ylabel("MSE-score")
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.tight_layout()
# plt.arrow(300,MSE[0], -300, 0)
plt.legend()

##====================================================================================

# plt.figure()
# plt.plot(beta0,beta1,'b')
# plt.scatter(beta0,beta1,c='b', label=f"$\gamma = 0$, it.$={last_iter}$")
# plt.plot(beta0_m,beta1_m,'g')
# plt.scatter(beta0_m,beta1_m,c='g', label=f"$\gamma = {gamma}$, it.$={last_iter_m}$")
# plt.plot(beta0_m, beta1_m,'g', label=f"$\gamma = {gamma}$")
# plt.xlabel("beta0")
# plt.ylabel("beta1")
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
# plt.tight_layout()



# b0 = np.arange(-1,5, 0.1)
# b1 = np.arange(-1,-9,0.1)
#
# b0m, b1m= np.meshgrid(b0,b1)
# b0m+b1m*x
#z = getMSE(yy, b0m+b1m*xx)
# z = b0m**2+b1m**2
# print(z)
# plt.contourf(b0,b1,z)

# def MSE(b0,b1):
#     return np.sum((y-(b0+b1*x))**2)/n

# b0 = np.arange(-1,8, 0.1)
# b1 = np.arange(-1.5,6,0.1)
# b0m, b1m = np.meshgrid(b0, b1, sparse=True)
# # z = b0m**2 + b1m**2
# errors = np.zeros((len(b1), len(b0)))
#
# for i in range(len(b0)):
#     for j in range(len(b1)):
#         betas = np.array([b0[i],b1[j]]).reshape(-1,1)
#         errors[j,i] = getMSE(y, X@betas)
# # z = MSE(b0m,b1m)
# h = plt.contour(b0,b1,errors, 200)
# plt.scatter(4,3,c='r', label= "(beta0,beta1)=(4,3)")
# plt.legend()

# for i in range(len(gradients)):
#     plt.arrow(np.double(beta0[i]),np.double(beta1[i]),-eta*np.double(gradients[i][0]),-eta*np.double(gradients[i][1]),width = 0.03)


plt.show()
