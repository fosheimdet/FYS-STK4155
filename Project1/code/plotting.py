import numpy as np
import matplotlib.pyplot as plt

from functions import FrankeFunction, addNoise, desMat
from regression_methods import OLS
from resampling_methods import crossValidation
from misc import bias_var_noResamp


def plotAgainstOrder(task):
    #Setting common parameters across methods
    sklearn = False
    plotting = True
    maxOrder = task[1]           #Maximum order to iterate to
    sigma =np.array(task[2])     #Vector of various sigma values
    savePlot=task[3]
    scaling=task[4]
    #Parameters specific for each method
    if(task[0] == 'bias_var'):
        xr,yr,Z,sklearn = task[5],task[6],task[7],task[8] #Z here is without noise

    elif(task[0]=='crossval'):
        xr,yr,z,f,K,shuffle = task[5],task[6],task[7],task[8],task[9],task[10]


#Loop through the various sigma's and make a plot for each
    for i in range(0,len(sigma)):
        orders     = np.arange(0,maxOrder,1) #To use as x-axis
        bias       = np.zeros(maxOrder)
        variance   = np.zeros(maxOrder)
        cov        = np.zeros(maxOrder)
        MSEtest    = np.zeros(maxOrder)
        MSEtrain   = np.zeros(maxOrder)
        R2test     = np.zeros(maxOrder)
        R2train    = np.zeros(maxOrder)

        if(task[0]=='crossval'):
            z = addNoise(z,sigma[i])

        for order in range(0,maxOrder):
            if(task[0]=='bias_var'):
                cov[order],bias[order],variance[order],X,Z_orig,Z_tilde,\
                beta_hat,MSEtest[order],R2test[order],MSEtrain[order],R2train[order], var_beta\
                = OLS(xr,yr,Z,order,sigma[i],scaling,sklearn)
            elif(task[0]=='bootstrap'):
                print('My name is mr.Bootstrap')
            elif(task[0]=='crossval'):
                 X = desMat(xr,yr,order)
                 MSEtest[order],bias[order],variance[order],R2test[order],cov[order]\
                  = crossValidation(X,z,f,K,shuffle)


        Sigma2_est = MSEtest - bias - variance + cov
        Sigma2_true = (sigma[i]**2)*np.ones(len(bias))

        fig=plt.figure()
        plt.plot(orders,bias,'firebrick', label='bias',linestyle='--', dashes=(2,1))
        plt.plot(orders,variance,'steelblue', label = 'variance',linestyle='--')
        plt.plot(orders,MSEtest, 'darkgreen', label='MSE_test')
        plt.plot(orders,Sigma2_est,'black', label='sigma2_est')
        plt.plot(orders,Sigma2_true,'black',label = 'sigma2_true', linestyle = 'dotted')
        # plt.plot(xaxis,MSEtrain_vec, 'steelblue', label='MSE_train')
        plt.ylabel('MSE')
        plt.xlabel('Polynomial degree')

        # plt.xlabel(r"$\eta$")
        # plt.ylabel(r"$E_L$")
        # plt.title(r"Bias and Variance of our model on test data")
        plt.title(f"Bias-Variance decomposition for Var($\epsilon$)={sigma[i]}$^2$")

        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.tight_layout()
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([orders[0], orders[len(orders)-1]])
        if(savePlot):
            plt.savefig(f'bias_var_tradeoff_noresamp/biVar_nore_sigma={sigma}_scaleing={scaling}_sklearn={sklearn}.png')
        plt.legend()
        plt.show()


def plotssVSOrder(xr,yr,z,maxOrder,sigma2,K):

    maxOrder = 21
    sigma2 = 0.8
    orders= np.arange(0,maxOrder)
    MSPE_vec=np.zeros(maxOrder)
    bias_vec = np.zeros(maxOrder)
    var_vec = np.zeros(maxOrder)
    cov_vec = np.zeros(maxOrder)
    R2_vec = np.zeros(maxOrder)

    f = z
    z_n = addNoise(z,sigma2)
    shuffle = True
    for order in range(0,maxOrder):
        X = desMat(xr,yr,order)
        #Return result from each of the K folds in the form of vectors, then take the average
        #and store it as the i'th element of the vectors
        X_train,X_test,z_train,z_test = train_test_split(X,z_n,test_size = 0.2,random_state=0)
        MSPE,bias,var,R2,cov = crossValidation(X,z_n,K,f,shuffle)
        MSPE_vec[order] = np.mean(MSPE)
        bias_vec[order] = np.mean(bias)
        var_vec[order] = np.mean(var)
        cov_vec[order] = np.mean(cov)
        R2_vec[order] = np.mean(R2)

    Sigma2_est = MSPE_vec-bias_vec-var_vec+cov_vec
    fig=plt.figure()
    plt.plot(orders,bias_vec,'firebrick', label='bias',linestyle='--', dashes=(2,1))
    plt.plot(orders,var_vec,'steelblue', label = 'variance',linestyle='--', dashes=(2,1))
    # plt.plot(orders,Sigma2_est, 'black', label='Sigma2_est')
    # plt.plot(orders,np.ones(len(Sigma2_est))*sigma2, 'black', label='Sigma2_true',\
    # linestyle='--')
    #plt.plot(xaxis,bias_vec+variance_vec+sigma2, 'green', label='Sum')
    # plt.plot(xaxis,MSEtrain_vec, 'steelblue', label='MSE_train')
    plt.ylabel('MSE')
    plt.xlabel('Polynomial degree')

    # plt.xlabel(r"$\eta$")
    # plt.ylabel(r"$E_L$")
    # plt.title(r"Bias and Variance of our model on test data")
    plt.title(f"Bias-Variance decomposition for $\sigma$={sigma2}")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([orders[0], orders[len(orders)-1]])
    # if(savePlot):
    #     plt.savefig(f'bias_var_tradeoff_noresamp/biVar_nore_sigma2={sigma2}_scaleing={scaling}_sklearn={sklearn}.png')
    plt.legend()
    plt.show()
#===============================================================================
