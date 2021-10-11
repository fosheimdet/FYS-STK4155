import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from functions import addNoise,desMat
from regression_methods import OLS
from resampling_methods import crossValidation, bootstrap
from sklearn.model_selection import train_test_split



#===============================================================================
#Makes surface plots of both the fitted function and the original Franke function,
#side by side
#===============================================================================
def FrankePlotter(xx,yy,Z_orig, Z_tilde, order, sigma):
    fig = plt.figure(figsize=(8,4))

    #First plot
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(xx,yy,Z_orig, cmap = cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(f"Original data, var($\epsilon$) = {sigma}$^2$")
    #Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(20, 30)
    # Add a color bar which maps values to colors
    #fig.colorbar(surf, shrink=0.5, aspect = 5)

    #Second plot
    ax = fig.add_subplot(122, projection='3d')
    #ax.scatter(xs = heights, ys = weights, zs = ages)
    surf = ax.plot_surface(xx,yy,Z_tilde, cmap = cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(f'OLS fit with polynomial of order {order}')
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(20, 30)
    #fig.colorbar(surf, shrink=0.5, aspect = 5)
    plt.savefig(f'surfacePlot/OLS_surface_order={order}_sigma={sigma}.png')
    plt.show()
#===============================================================================



#===============================================================================
#====Calculates the confidence intervals of beta_hat and plots them=============
#===============================================================================
def beta_CI(beta_hat,var_beta,alpha,order,save):
    Z = norm.ppf(1-alpha/2)
    #95% confidence intervals of beta
    beta_lower = beta_hat - Z*np.sqrt(var_beta)
    beta_upper = beta_hat + Z*np.sqrt(var_beta)
    beta_CI=[Z*np.sqrt(var_beta),Z*np.sqrt(var_beta)]
    beta_indices = np.arange(0,len(beta_hat),1)

    fig = plt.figure()
    plt.errorbar(beta_indices,beta_hat,beta_CI, fmt='o',color = 'darkgreen',
    ecolor = 'darkgreen', capsize = 3,capthick = 2 ,barsabove = True, ms=5)

    plt.xlabel(r"$\beta$ index")
    plt.ylabel(r"$\beta$ value")
    orderStr= 'th'
    if(order==1):
        orderStr = 'st'
    elif(order==2):
        orderStr = 'nd'
    elif(order==3):
        orderStr = 'rd'
    plt.title(f"Regression coefficients with their 95% CI for {order}"+orderStr+ " order pol.")
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.tight_layout()
    #plt.savefig(datafile_path+ofile+"_"+solv+"_"+dist+'_nh_'+str(nh)+"_RBM_"+str(RBM)+'_.pdf')
    #plt.legend()
    if(save == True):
        plt.savefig(f'CI_beta_order{order}.png')
    plt.show()
#===============================================================================



#===============================================================================
#===========Bias-Variance trade-off without resampling==========================
#===============================================================================
def bias_var_noResamp(xr,yr,Z,maxOrder,sigma,scaling,sklearn,plotting,savePlot):
    xaxis = np.arange(0,maxOrder,1)

    bias_vec = np.zeros(maxOrder)
    variance_vec = np.zeros(maxOrder)
    MSEtest_vec = np.zeros(maxOrder)
    MSEtrain_vec = np.zeros(maxOrder)

    #cov_f_ytilde_vec = []


    for order in range(0,maxOrder):
        cov_f_ytilde,bias,variance,X,Z_orig,Z_tilde,beta_hat,MSE_test,\
        R2_test, MSE_train, R2_train, var_beta= OLS(xr,yr,Z,order,sigma,scaling,sklearn)

        bias_vec[order] = bias
        variance_vec[order] = variance
        MSEtest_vec[order] = MSE_test+cov_f_ytilde
        #print(cov_f_ytilde)

        MSEtrain_vec[order] = MSE_train

    # print('Bias: ', bias_vec)
    # print('Variances: ', variance_vec)
    if(plotting==True):
        fig=plt.figure()

        plt.plot(xaxis,bias_vec,'firebrick', label='bias',linestyle='--', dashes=(2,1))
        plt.plot(xaxis,variance_vec,'steelblue', label = 'variance',linestyle='--')
        plt.plot(xaxis,MSEtest_vec, 'darkgreen', label='MSE_test')
        plt.plot(xaxis,MSEtest_vec-(bias_vec+variance_vec),'black', label='sigma2_est')
        plt.plot(xaxis,np.ones(len(bias_vec))*sigma**2,'black',label = 'sigma2_true', linestyle = 'dotted')
        # plt.plot(xaxis,MSEtrain_vec, 'steelblue', label='MSE_train')
        plt.ylabel('MSE')
        plt.xlabel('Polynomial degree')

        # plt.xlabel(r"$\eta$")
        # plt.ylabel(r"$E_L$")
        # plt.title(r"Bias and Variance of our model on test data")
        plt.title(f"Bias-Variance decomposition for Var($\epsilon$)={sigma}$^2$")

        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.tight_layout()
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([xaxis[0], xaxis[len(xaxis)-1]])
        if(savePlot):
            plt.savefig(f'bias_var_tradeoff_noresamp/biVar_nore_sigma={sigma}_scaleing={scaling}_sklearn={sklearn}.png')
        plt.legend()
        plt.show()
#===============================================================================


#===============================================================================
#Compute prediction error and R2-score for various complexities, no resampling
#===============================================================================
def MSPE_R2(xr,yr,Z,maxOrder,sigma_vec,scaled,sklearn,savePlot,MSPE):
    beta_indices = np.arange(0,maxOrder,1)
    MSEtest_vec = np.zeros(maxOrder)
    MSEtrain_vec = np.zeros(maxOrder)
    R2test_vec = np.zeros(maxOrder)
    R2train_vec = np.zeros(maxOrder)
    for i in range(0,len(sigma_vec)):
        for order in range(0,maxOrder):
            cov_f_ytilde,bias,variance,X,X_orig,Z_tilde,beta_hat,MSE_test,R2_test, \
            MSE_train, R2_train, var_beta = OLS(xr,yr,Z,order, sigma_vec[i],scaled,sklearn)
            MSEtest_vec[order] = MSE_test
            MSEtrain_vec[order] = MSE_train
            R2test_vec[order] = R2_test
            R2train_vec[order] = R2_train
        #Make plot of MSE_test and MSE_train as a function of order
        if(MSPE==True):
            fig=plt.figure()
            plt.plot(beta_indices,MSEtest_vec, 'firebrick', label='MSE_test')
            plt.plot(beta_indices,MSEtrain_vec, 'steelblue', label='MSE_train')
            plt.ylabel('MSE', fontsize = 11)
            plt.xlabel('Polynomial degree', fontsize = 11)
            plt.title(f"Prediction error vs. complexity for Var($\epsilon$)={sigma_vec[i]}$^2$")

            # Show the minor grid lines with very faint and almost transparent grey lines
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.tight_layout()
            plt.legend()
            axes = plt.gca()
            axes.set_xlim([beta_indices[0], beta_indices[len(beta_indices)-1]])
            # plt.legend(prop={'size':11})
            if(savePlot):
                plt.savefig(f'test_vs_train_MSPE/sigma_0.01_to_1/MSPE_Sigma={sigma_vec[i]}_maxOrder={maxOrder-1}.png')

            plt.show()
        else:
            fig=plt.figure()
            plt.plot(beta_indices,R2test_vec, 'firebrick', label=r'$R^2$_test')
            plt.plot(beta_indices,R2train_vec, 'steelblue', label=r'$R^2$_train')
            plt.plot(beta_indices,np.ones(len(beta_indices)), 'black', linestyle = '--')
            plt.plot(beta_indices,np.zeros(len(beta_indices)), 'black', linestyle = '--')
            plt.ylabel(r'$R^2$-score', fontsize =11)
            plt.xlabel('Polynomial degree', fontsize = 11)
            plt.title(f"$R^2$-score vs. complexity for Var($\epsilon$)={sigma_vec[i]}$^2$")

            # Show the minor grid lines with very faint and almost transparent grey lines
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.legend()
            axes = plt.gca()
            axes.set_xlim([beta_indices[0], beta_indices[len(beta_indices)-1]])
            plt.tight_layout()
            #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            # plt.legend(prop={'size':12})
            if(savePlot):
                plt.savefig(f'test_vs_train_R2/sigma_0.01_to_1/R2_Sigma={sigma_vec[i]}_maxOrder={maxOrder-1}.png')

            plt.show()
#===============================================================================



#===============================================================================
#Bias-variance tradeoff analysis using bootstrapping and OLS for the Franke function
#===============================================================================
def bootstrap_Franke_OLS(xr,yr,z,nBootstrap,maxOrder,sigma2,scaled,sklearn):
    n = np.max(xr.shape) #Number of elements in xr
    epsilon = np.random.normal(0,np.sqrt(sigma2),n) #vector of length n*n containing draws from N(0,1)
    bootstrap_bool=True
    if(bootstrap_bool==True):
        z = z +epsilon
    # z_train,z_test = train_test_split(z,0.2)
    order_indices = np.arange(0,maxOrder,1)
    bias_vec = np.zeros(maxOrder)
    variance_vec = np.zeros(maxOrder)
    MSEtest_vec = np.zeros(maxOrder)
    cov_vec = np.zeros(maxOrder)

    for order in range(0,maxOrder):
         MSEtest_mean,MSEtest_var,bias_mean,bias_var,variance_mean,variance_var\
         ,covariance_mean= bootstrap(xr,yr,z,nBootstrap,order,sigma2,scaled,sklearn)

         bias_vec[order] = bias_mean
         variance_vec[order] = variance_mean
         MSEtest_vec[order] = MSEtest_mean
         cov_vec[order] = covariance_mean


    Sigma_est = MSEtest_vec-bias_vec-variance_vec+cov_vec
    sum = MSEtest_vec + bias_vec + variance_vec

    fig=plt.figure()
    plt.plot(order_indices,bias_vec, 'firebrick',linestyle='--', dashes=(2,1), label='bias')
    plt.plot(order_indices,variance_vec, 'steelblue',linestyle='--', label='variance')
    plt.plot(order_indices,MSEtest_vec, 'darkgreen', label='MSE')
    plt.plot(order_indices,Sigma_est, 'black', label='Sigma2_est')
    plt.plot(order_indices,np.ones(len(Sigma_est))*sigma2, 'black', label='Sigma2_true',linestyle='--', dashes=(2,1))
    #plt.plot(order_indices,sumBiasVar, 'orange', label='Sum')
    plt.ylabel('MSE', fontsize = 11)
    plt.xlabel('Polynomial degree', fontsize = 11)
    plt.title(f"Bias-Variance tradeoff using bootstrap with var($\epsilon$)={sigma2}")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.tight_layout()
    axes=plt.gca()
    axes.set_xlim([order_indices[0], order_indices[len(order_indices)-1]])
    #axes.set_yscale('log')
    plt.legend()
    # plt.legend(prop={'size':11})
    plt.show()
#===============================================================================
