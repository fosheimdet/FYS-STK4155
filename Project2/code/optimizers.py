import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from functions import getMSE, getR2, scale_data, desMat1D
import random






def step_func(t,t0,t1):
    return t0/(t+t1)



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
    epochs,M,eta,lmbd,gamma, t0, t1was used  = hyperparams
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

            if(method == "Adam"):
                v = gamma*v + sf*eta
            else:
                v = gamma*v + sf*eta*gradC
            betas = betas - v
            t+=1

        cost = getMSE(y,X@betas)
        if(abs(cost_prev-cost)<=threshold):
            break
        cost_prev = cost

    last_iter=e

    return  MSE, R2,betas, last_iter


def gradientDescent(betas,hyperparams,iterations,gamma=0,threshold = 1e-3, analysis = True):
    eta, lmbd =hyperparams[2:4]
    X,y = X_train,y_train
    N = X.shape[0] #number of datapoints
    MSE = []
    R2 = []
    v=0
    beta0 = []
    beta1 = []
    gradients = []

    for i in range(iterations):
        beta0.append(betas[0])
        beta1.append(betas[1])

        cost_prev = getMSE(y,X@betas)
        gradC = (2/N)*X.T@(X@betas -y)+2*lmbd*betas
        v = gamma*v + eta*gradC
        betas = betas-v

        cost = getMSE(y, X@betas)
        if(analysis):
            MSE.append(getMSE(y_test, X_test@betas))
            R2.append(getR2(y_test,X_test@betas))
            gradients.append(gradC)

        if(abs(cost - cost_prev) <= threshold):
            print(f"stopped after {i} iterations")
            break

    last_iter = i #last iteration

    if(analysis):
        return  MSE, R2, betas, last_iter, beta0, beta1
    else:
        return betas



#def SGD(X,y,betas,eta,epochs, M , lmbd, gamma=0, adaptive=True, threshold = 1e-4):
def SGD(betas,hyperparams,threshold, momentum = False, adaptive=True, analysis = False):
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

    m = int(X.shape[0]/M)
    for e in range(epochs):
        indices = rng.permutation(np.arange(0,len(y)))
        X,y = X[indices], y[indices]

        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradC = (2/M)*Xi.T@(Xi@betas-yi) + 2*lmbd*betas

            if(adaptive):
                eta = step_func(e*m+i,t0,t1)
            v =gamma*v + eta*gradC
            betas = betas -v

        #Store MSE and R2 for each epoch
        if(analysis):
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
    if(analysis):
        return MSE, R2,betas, last_iter
    else:
        return betas




if __name__ == '__main__':


    np.random.seed(0)
    n = 100
    p = 5

    x = np.linspace(0,1,n)
    #x = 2*np.random.rand(n)
    #y = 2*x + 10*np.random.random_sample(len(x)) - 10/2
    a0 = 4
    a1 = 3
    a2 = 6
    y = a0 + a1*x + a2*x**2 + np.random.randn(n)
    y = y.reshape(-1,1)

    X = desMat1D(x,p)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # X_train, X_test = scale_data(X_train,X_test,True)

    betas = np.random.normal(0,1,(p,1))

    print(X.shape,betas.shape)

    H = (2.0/n)* X.T @ X
    EigValues, EigVectors = np.linalg.eig(H)
    print(f"Eigenvalues of Hessian Matrix:{EigValues}")

    iterations = 200

    gamma = 0.9

    eta = 1.0/np.max(EigValues)
    eta = 1e-2
    lmbd = 1e-2
    eta_m = 1e-4    #For momentum
    lmbd_m = 1e-2
    epochs = 500
    #t0, t1 = 5,50
    t0, t1 = 0.05,0.5
    M = 10       #Batch size
    m = int(n/M)  #Number of mini-batches in an epoch

    threshold = 1e-3
    heatmapBool = False

    #Optimization methods
    methods = ["GD", "SGD","Adagrad", "RMSprop", "Adam"]
    #            0     1       2          3         4
    method_int = 0

    method= methods[method_int]

    hyperparams = [epochs,M,eta,lmbd,gamma, t0, t1]
    hyperparams_m = [epochs,M,eta_m,lmbd_m,gamma, t0, t1]

    if(method == "GD"):
        MSE,R2,betas_opt,last_iter,beta0, beta1 = gradientDescent(betas,hyperparams,iterations,gamma=0, threshold = threshold)
        MSE_m, R2_m, betas_opt_m, last_iter_m, beta0_m, beta1_m = gradientDescent(betas,hyperparams_m,iterations,gamma)
    elif(method == "SGD"):
        MSE,R2,betas_opt, last_iter = SGD(betas,hyperparams,threshold,momentum=False,adaptive=True,analysis=True) #without momentum
        MSE_m, R2_m, betas_opt_m, last_iter_m = SGD(betas,hyperparams_m,threshold,momentum=True,adaptive=True,analysis=True) #With momentum
    else:
        MSE,R2,betas_opt,last_iter = optimize(method,betas,hyperparams,momentum=False,threshold=threshold)
        MSE_m, R2_m, betas_opt_m, last_iter_m = optimize(method,betas,hyperparams_m,momentum=True,threshold=threshold)

    ytilde = X@betas_opt
    ytilde_m = X@betas_opt_m
    #===========================================================================
    #                         Produce heatmap
    #===========================================================================

    def heatmap():
        epoch_vals = [20*(i+1) for i in range(2,15,1)]
        eta_vals = [1e-5,1e-4,1e-3,1e-2,1e-1]
        #eta_vals = [1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10]
        lmbd_vals = [1e-5,1e-4,1e-3,1e-2,1e-1]


        M_vals = [5,10,20,30,50,80,100]
        gamma_vals = [0.2,0.3,0.5,0.7,0.8, 0.85, 0.9]
        t0_vals = np.linspace(0,10,11)
        t1_vals = np.linspace(0,100,11)

        hParList = [epoch_vals, M_vals, eta_vals, lmbd_vals,  gamma_vals, t0_vals, t1_vals]
        hParStrings = ['epochs', 'batch size', 'eta', 'lambda', 'gamma', 't0', 't1']
        #                 0            1         2        3        4       5     6
        itIndices = [3,2]

        len1 = len(hParList[itIndices[0]])
        len2 = len(hParList[itIndices[1]])

        #Heatmaps:
        MSE_hm = np.zeros((len1,len2))
        R2_hm =np.zeros((len1,len2))

        hyperparams = [epochs,M,eta,lmbd,gamma, t0, t1]

        #Calculating MSE and R2 values for each of the two chosen iterables
        n_heatmaps = 5 #Average over multiple heatmaps for more reliable results
        for n in range(n_heatmaps):
            for i, val1 in enumerate(hParList[itIndices[0]]):
                for j, val2 in enumerate(hParList[itIndices[1]]):
                    hyperparams[itIndices[0]] =val1
                    hyperparams[itIndices[1]] =val2
                    if(method == "GD"):
                        MSE_l,R2_l = gradientDescent(betas,hyperparams,iterations,gamma=gamma, threshold = -1)[0:2]
                    elif(method == "SGD"):
                        MSE_l,R2_l = SGD(betas,hyperparams,threshold,momentum=True,adaptive=True,analysis=True)[0:2]
                    else:
                        MSE_l,R2_l=optimize(method,betas,hyperparams,momentum=True,threshold=threshold)[0:2]

                    MSE_hm[i,j]+= MSE_l[-1]
                    R2_hm[i,j]+= R2_l[-1]

        MSE_hm = MSE_hm/n_heatmaps
        R2_hm = R2_hm/n_heatmaps
        fig, ax = plt.subplots(figsize = (10, 10))
        #sns.heatmap(MSE_hm,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax, cmap = "viridis",fmt = '.1f',cbar_kws={'format': '%.0f%'})
        sns.heatmap(MSE_hm,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax, cmap = "viridis")
        ax.set_title("MSE$_{test}$" +f" for polynomial fit using {method} with $\gamma={gamma}$")
        ax.set_ylabel(hParStrings[itIndices[0]])
        ax.set_xlabel(hParStrings[itIndices[1]])


        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(R2_hm,xticklabels = hParList[itIndices[1]], yticklabels = hParList[itIndices[0]], annot=True, ax=ax, cmap = "viridis")
        ax.set_title("R2$_{test}$"+ f" for polynomial fit using {method} with $\gamma={gamma}$")
        ax.set_ylabel(hParStrings[itIndices[0]])
        ax.set_xlabel(hParStrings[itIndices[1]])
        #ax.set_xlabel("$\eta$")
        # plt.show()

    if(heatmapBool):
        heatmap()
    #======================================================
    #                    Plotting
    #======================================================

    pltCols = ['orange','brown','pink','gray','olive','cyan']
    gamma_vals = [0.8,0.85,0.90,0.92,0.95,0.98]
    ytilde_gammas = []
    last_iter_vals = []
    MSE_vals = []
    for i, g in enumerate(gamma_vals):
        hyperparams = [epochs,M,eta,lmbd,g, t0, t1]
        # hyperparams_m = [epochs,M,eta_m,lmbd_m,gamma, t0, t1]
        if(method == "GD"):
            MSE_m, R2_m, betas_opt_m, last_iter_m, beta0_m, beta1_m = gradientDescent(betas,hyperparams,iterations,g)
        elif(method == "SGD"):
            MSE_m, R2_m, betas_opt_m, last_iter_m = SGD(betas,hyperparams,threshold,momentum=True,adaptive=True,analysis=True) #With momentum
        else:
            MSE_m, R2_m, betas_opt_m, last_iter_m = optimize(method,betas,hyperparams,momentum=False,threshold=threshold)
        ytilde_gammas.append(X@betas_opt_m)
        last_iter_vals.append(last_iter_m)
        MSE_vals.append(MSE_m[-1])

    plt.figure()
    plt.title("Polynomial fit with "+f"$p={p}$"+ " using "+method+" with "+ f"$\eta={eta}$, $\lambda={lmbd}$")
    plt.scatter(x,y, color = 'r', label = f"${a0}+{a1}x+{a2}x^2+\epsilon$")
    iter_str = "epochs"
    if(method == "GD"):
        iter_str = "it."
    plt.plot(x,ytilde,'b', label = f"$\gamma=0$"+f", MSE = {MSE[-1]:.2f}")
    if(method!="Adam"):
        for i in range(len(ytilde_gammas)):
            ytilde_m = ytilde_gammas[i]
            plt.plot(x,ytilde_m,c=pltCols[i], ls='--',label = f"$\gamma={gamma_vals[i]}$" + f", MSE={MSE_vals[i]:.2f}")
    #plt.plot(x,ytilde_m,'g', label = f"$\gamma={gamma}$, "+iter_str+f"$={last_iter_m}$, MSE={MSE_m[-1]:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.title("R2 score for "+method+" with and without momentum")
    if(method=="Adam"):
        plt.title("R2 score for "+method)
    plt.plot(np.arange(0,last_iter+1), R2[0:],'b', label=f"$\gamma = 0$, "+iter_str+f"$={last_iter}$, R2={R2[-1]:.3f} \n" +
    f"($\eta={eta}$, $\lambda={lmbd}$)")
    if(method!="Adam"):
        plt.plot(np.arange(0,last_iter_m+1), R2_m[0:],'g', label=f"$\gamma = {gamma}$, "+iter_str+f"$={last_iter_m}$, R2={R2_m[-1]:.3f} \n" +
        f"($\eta={eta_m}$, $\lambda={lmbd_m}$)")
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
    if(method=="Adam"):
        plt.title("MSE score for "+method)
    plt.plot(np.arange(0,last_iter+1), MSE[0:],'b', label=f"$\gamma = 0$, "+iter_str+f"$={last_iter}$, MSE={MSE[-1]:.3f} \n" +
    f"($\eta={eta}$, $\lambda={lmbd}$)")
    if(method!="Adam"):
        plt.plot(np.arange(0,last_iter_m+1), MSE_m[0:],'g', label=f"$\gamma = {gamma}$, "+iter_str+f"$={last_iter_m}$, MSE={MSE_m[-1]:.3f} \n" +
        f"($\eta={eta_m}$, $\lambda={lmbd_m}$)")
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
