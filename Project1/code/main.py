import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn as sk
from sklearn import svm
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from random import randrange


 #import function for constructing the design matrix
# from functions import FrankeFunction,desMat,getMSE,getR2,StandardPandascaler,plotter,beta_CI
from functions import FrankeFunction,desMat,getMSE,addNoise
from regression_methods import OLS
from resampling_methods import bootstrap,crossValidation

from plotting import plotAgainstOrder
from misc import FrankePlotter,beta_CI,bias_var_noResamp, MSPE_R2, bootstrap_Franke_OLS



def main():
    np.random.seed(2021)

    # dx = 0.05
    # x = np.arange(0,1,dx)
    # y = np.arange(0,1,dx)
    n = 20
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)

    # x = np.random.uniform(low=0, high=1, size=(n,))
    # y = np.random.uniform(low=0, high=1, size=(n,))

    xx, yy = np.meshgrid(x,y)
    Z = FrankeFunction(xx,yy)
    z = np.ravel(Z)     #Original z values of Franke's function without noise
    # print('z: ', z)

    xr = np.ravel(xx)
    yr = np.ravel(yy)



    #===============================================================================
    #===============================================================================
    n = 20 #Number of ticks for x and y axes

    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    xx, yy = np.meshgrid(x,y)
    Z = FrankeFunction(xx,yy)
    z = np.ravel(Z)         #Original z values of Franke's function without noise
    f = z
    xr = np.ravel(xx)
    yr = np.ravel(yy)




    maxOrder = 11
    sigma_0 = np.array([0.01,0.1,0.2,0.5,1])
    sigma_1 =np.array([0.01,0.1,0.2,0.5,1])
    sigma_2 =np.array([0.01,0.1,0.2,0.5,1])

    savePlot = False
    scaling = True

    sklearn = False  #Use sklearn in OLS?


    nBoot = 10

    K = 5           #Number of folds in cross validation alg.
    shuffle = True  #Shuffle the data before performing folds?

    bias_var = ['bias_var', maxOrder,sigma_0,savePlot,scaling,xr,yr,Z,sklearn]
    bootstrap= ['bootstrap',maxOrder,sigma_1,savePlot,scaling,nBoot]
    crossval = ['crossval', maxOrder,sigma_2,savePlot,scaling,xr,yr,z,f,K,shuffle]

    tasks = [bias_var, bootstrap, crossval]   #Nested list

    plotAgainstOrder(tasks[2])


    #===============================================================================
    #===============================================================================

    # sigma2 = 0.1
    # order = 2
    # z=addNoise(z,sigma2)
    # print('z_shape: ',z.shape)
    # X = desMat(xr,yr,order)
    #
    # LR = LinearRegression()
    # LR.fit(X,z)
    # z_pred = LR.predict(X)
    # print(z_pred.shape)
    # Z_tilde = z_pred.reshape(n,n)
    #
    # print('reg coeff: ', LR.coef_)


    # plotVSOrder(xr,yr,z,21,0.8,5)

    # order = 10
    # sigma = 0.3
    # scaled = False
    # sklearn = False
    #
    # cov,bias,variance,X,Z_orig,Z_tilde,beta_hat,MSE_test,R2_test, \
    # MSE_train, R2_train, var_beta = OLS(xr,yr,Z,order,sigma,scaled,sklearn)
    #
    # FrankePlotter(xx,yy,Z_orig,Z_tilde,order,sigma)

    #===============================================================================
    #Cross validation
    #===============================================================================

    # K = 5
    # nEl = 25
    # z = np.arange(0,nEl,1)
    # b = np.arange(0,nEl*4)
    # X = b.reshape(nEl,4)
    # crossValidation(X,z,K)
    #
    # K = 5
    # maxOrder = 21
    # sigma2 = 0.8
    # #crossValidation(X,z,K)
    # orders= np.arange(0,maxOrder)
    # MSPE_vec=np.zeros(maxOrder)
    # bias_vec = np.zeros(maxOrder)
    # var_vec = np.zeros(maxOrder)
    # cov_vec = np.zeros(maxOrder)
    # R2_vec = np.zeros(maxOrder)
    #
    # f = z
    # z_n = addNoise(z,sigma2)
    #
    # shuffle = True
    #
    # for order in range(0,maxOrder):
    #     X = desMat(xr,yr,order)
    #     #Return result from each of the K folds in the form of vectors, then take the average
    #     #and store it as the i'th element of the vectors
    #     X_train,X_test,z_train,z_test = train_test_split(X,z_n,test_size = 0.2,random_state=0)
    #     MSPE,bias,var,R2,cov = crossValidation(X,z_n,K,f,shuffle)
    #     MSPE_vec[order] = np.mean(MSPE)
    #     bias_vec[order] = np.mean(bias)
    #     var_vec[order] = np.mean(var)
    #     cov_vec[order] = np.mean(cov)
    #     R2_vec[order] = np.mean(R2)
    #
    # Sigma2_est = MSPE_vec-bias_vec-var_vec+cov_vec
    # fig=plt.figure()
    # plt.plot(orders,bias_vec,'firebrick', label='bias',linestyle='--', dashes=(2,1))
    # plt.plot(orders,var_vec,'steelblue', label = 'variance',linestyle='--', dashes=(2,1))
    # # plt.plot(orders,Sigma2_est, 'black', label='Sigma2_est')
    # # plt.plot(orders,np.ones(len(Sigma2_est))*sigma2, 'black', label='Sigma2_true',\
    # # linestyle='--')
    # #plt.plot(xaxis,bias_vec+variance_vec+sigma2, 'green', label='Sum')
    # # plt.plot(xaxis,MSEtrain_vec, 'steelblue', label='MSE_train')
    # plt.ylabel('MSE')
    # plt.xlabel('Polynomial degree')
    #
    # # plt.xlabel(r"$\eta$")
    # # plt.ylabel(r"$E_L$")
    # # plt.title(r"Bias and Variance of our model on test data")
    # plt.title(f"Bias-Variance decomposition for Var($\epsilon$)={sigma2}")
    #
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.tight_layout()
    # plt.legend()
    # axes = plt.gca()
    # axes.set_xlim([orders[0], orders[len(orders)-1]])
    # # if(savePlot):
    # #     plt.savefig(f'bias_var_tradeoff_noresamp/biVar_nore_sigma2={sigma2}_scaleing={scaling}_sklearn={sklearn}.png')
    # plt.legend()
    # plt.show()



    #===============================================================================
    ############Comparing with sklearn's cross validation

    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, z_train)
    # clf.score(X_test, z_test)

    # linreg = LinearRegression()
    # linreg.fit(X, z)
    ##ztilde2 = linreg.predict(X)
    # kfold = KFold(n_splits = K, shuffle=True)
    # scores_test = cross_val_score(linreg, X, z, cv = kfold)
    # print("R2-Score sk_OLS: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std() * 2))
    #
    # scores_mse_test = cross_val_score(linreg, X, z, scoring='neg_mean_squared_error', cv=kfold)
    # estimated_mse_test = np.mean(-scores_mse_test)
    # print('MSE-score sk_OLS:', estimated_mse_test)
    #===============================================================================



    #===============================================================================


    #===============================================================================
    #Bias-var tradeoff for Franke function using OLS w. bootstrap
    #===============================================================================

    # maxOrder = 14 #Plot MSPE or R2 scores against polynomials up to this order
    # sigma_vec = [0.01,0.1,0.5,1,2]
    # nBootstrap = 20
    # scaled = True
    # sklearn = False
    # #sigma2_vec = [0.01,0.1,0.5,1] #Vector containing the values of noise to use
    # # savePlot = False
    # # MSPE = False #True to compare MSPE and false to compare R2-score
    # for i in range(0,len(sigma_vec)):
    #     bootstrap_Franke_OLS(xr,yr,z,nBootstrap,maxOrder,sigma_vec[i],scaled,sklearn)

    #===============================================================================


    #===============================================================================
    #Comparing MSPE and R2 in training- and test data for various complexities,
    # without resampling
    #===============================================================================

    # maxOrder = 16 #Plot MSPE or R2 scores against polynomials up to this order
    # sigma_vec = np.array([0.01,0.1,0.5,1])
    # #sigma_vec = np.array([0.001,0.01,0.05,0.1,1])
    # #sigma2_vec = sigma_vec**2 #Vector containing the values of noise to use
    # scaled = True
    # sklearn = True
    # savePlot = False
    # MSPE = False #True to compare MSPE and false to compare R2-score
    # MSPE_R2(xr,yr,Z,maxOrder,sigma_vec,scaled,sklearn,savePlot,MSPE)

    #===============================================================================


    #===============================================================================
    #========Bias-Variance tradeoff without resampling for OLS======================
    #===============================================================================
    #

    # maxOrder = 10
    # sigma = [0.01,0.1,0.2,0.5,1]
    # #sigma = [0.1]
    # scaling=True
    # sklearn = False
    # plotting = True
    # savePlot=False
    # random = 1
    # n = 20
    # if(random==0):
    #     x = np.linspace(0,1,n)
    #     y = np.linspace(0,1,n)
    # else:
    #     x = np.random.uniform(low=0, high=1, size=(n,))
    #     y = np.random.uniform(low=0, high=1, size=(n,))
    # xx, yy = np.meshgrid(x,y)
    # Z = FrankeFunction(xx,yy)
    # z = np.ravel(Z)     #Original z values of Franke's function without noise
    # # print('z: ', z)
    #
    # xr = np.ravel(xx)
    # yr = np.ravel(yy)
    # for i in range(0,len(sigma)):
    #     bias_var_noResamp(xr,yr,Z,maxOrder,sigma[i],scaling,sklearn,plotting,savePlot)

    #===============================================================================


    #===============================================================================
    #===============================Surface plots ==================================
    #===============================================================================
    # order = 21
    # sigma2 = 0.1
    # sklearn = True
    # scaled = True
    # bias,variance,X,Z_orig,Z_tilde,beta_hat, MSE_test, R2_test, MSE_train,\
    # R2_train, var_beta= OLS(xr,yr,Z,order,sigma2,scaled,sklearn)
    #
    # plotter(xx,yy,Z_orig,Z_tilde, order, sigma2) #Plots the Franke function with noise and the fitted function

    #===============================================================================










if __name__ =="__main__":
    main()






#B = n*n#Number of bootstrap samples
# def bootstrap_Franke_OLS(xr,yr,nBootstrap,maxOrder,sigma2):
#     n = np.max(xr.shape) #Number of elements in xr
#     epsilon = np.random.normal(0,sigma2,n*n) #vector of length n*n containing draws from N(0,1)
#     bootstrap_bool=True
#     if(bootstrap_bool==True):
#         z = z +epsilon
#     # z_train,z_test = train_test_split(z,0.2)
#     # B = 10
#     # maxOrder = 21
#     order_indices = np.arange(0,maxOrder,1)
#     MSE_vec = np.zeros(maxOrder)
#     bias_vec = np.zeros(maxOrder)
#     variance_vec = np.zeros(maxOrder)
#
#     for i in range(0,maxOrder):
#         MSE_vec[i],MSE_var,bias_vec[i],bias_var,variance_vec[i],variance_var = bootstrap(xr,yr,z,B,i)
#
#     # MSEb,biasb,varianceb = bootstrap(xr,yr,z,B,10)
#     # for i in range(0,len(MSEb)):
#     #     print('MSEb:\t', MSEb[i], 'biasb:\t', biasb[i],'varianceb:\t', varianceb[i])
#     #sumBiasVarSig = bias_vec + variance_vec + vareps
#     Sigma_est = MSE_vec-bias_vec-variance_vec
#
#     fig=plt.figure()
#     plt.plot(order_indices,bias_vec, 'firebrick', label='bias')
#     plt.plot(order_indices,variance_vec, 'steelblue', label='variance')
#     plt.plot(order_indices,MSE_vec, 'darkgreen', label='MSE')
#     plt.plot(order_indices,Sigma_est, 'black', label='Sigma_est')
#     #plt.plot(order_indices,sumBiasVar, 'orange', label='Sum')
#     plt.ylabel('MSE', fontsize = 11)
#     plt.xlabel('Polynomial degree', fontsize = 11)
#     plt.title(f"Bias-Variance tradeoff using bootstrap with var($\epsilon$)={vareps}")
#
#     # Show the minor grid lines with very faint and almost transparent grey lines
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#     plt.tight_layout()
#     plt.legend()
#     # plt.legend(prop={'size':11})
#     # if(savePlot):
#     #     plt.savefig(f'test_vs_train_MSPE/sigma2_0.01_to_1/MSPE_sigma2={sigma2_vec[i]}_maxOrder={maxOrder-1}.png')
#     plt.show()


#
# def MSPE_R2(maxOrder,sigma2_vec,savePlot,MSPE):
#     beta_indices = np.arange(0,maxOrder,1)
#     MSEtest_vec = np.zeros(maxOrder)
#     MSEtrain_vec = np.zeros(maxOrder)
#     R2test_vec = np.zeros(maxOrder)
#     R2train_vec = np.zeros(maxOrder)
#     for i in range(0,len(sigma2_vec)):
#         for order in range(0,maxOrder):
#             bias,variance,X,X_orig,Z_tilde,beta_hat,MSE_test,R2_test, \
#             MSE_train, R2_train, var_beta = OLS(xr,yr,Z,order, True, sigma2_vec[i])
#             MSEtest_vec[order] = MSE_test
#             MSEtrain_vec[order] = MSE_train
#             R2test_vec[order] = R2_test
#             R2train_vec[order] = R2_train
#         #Make plot of MSE_test and MSE_train as a function of order
#         if(MSPE==True):
#             fig=plt.figure()
#             plt.plot(beta_indices,MSEtest_vec, 'firebrick', label='MSE_test')
#             plt.plot(beta_indices,MSEtrain_vec, 'steelblue', label='MSE_train')
#             plt.ylabel('MSE', fontsize = 11)
#             plt.xlabel('Polynomial degree', fontsize = 11)
#             plt.title(f"Prediction error vs. complexity for Var($\epsilon$)={sigma2_vec[i]}")
#
#             # Show the minor grid lines with very faint and almost transparent grey lines
#             plt.minorticks_on()
#             plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#             plt.tight_layout()
#             plt.legend()
#             axes = plt.gca()
#             axes.set_xlim([beta_indices[0], beta_indices[len(beta_indices)-1]])
#             # plt.legend(prop={'size':11})
#             if(savePlot):
#                 plt.savefig(f'test_vs_train_MSPE/sigma2_0.01_to_1/MSPE_Sigma2={sigma2_vec[i]}_maxOrder={maxOrder-1}.png')
#
#             plt.show()
#         else:
#             fig=plt.figure()
#             plt.plot(beta_indices,R2test_vec, 'firebrick', label=r'$R^2$_test')
#             plt.plot(beta_indices,R2train_vec, 'steelblue', label=r'$R^2$_train')
#             plt.plot(beta_indices,np.ones(len(beta_indices)), 'black', linestyle = '--')
#             plt.plot(beta_indices,np.zeros(len(beta_indices)), 'black', linestyle = '--')
#             plt.ylabel(r'$R^2$-score', fontsize =11)
#             plt.xlabel('Polynomial degree', fontsize = 11)
#             plt.title(f"$R^2$-score vs. complexity for Var($\epsilon$)={sigma2_vec[i]}")
#
#             # Show the minor grid lines with very faint and almost transparent grey lines
#             plt.minorticks_on()
#             plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#             plt.legend()
#             axes = plt.gca()
#             axes.set_xlim([beta_indices[0], beta_indices[len(beta_indices)-1]])
#             plt.tight_layout()
#             #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#             # plt.legend(prop={'size':12})
#             if(savePlot):
#                 plt.savefig(f'test_vs_train_R2/sigma2_0.01_to_1/R2_Sigma2={sigma2_vec[i]}_maxOrder={maxOrder-1}.png')
#
#             plt.show()


#===============================================================================
#Make plots of the optimal regression parameters, beta_hat,and their confidence
#intervals for various orders of the fitting polynomial
#===============================================================================
# for order in range(0,10):
#     bias,variance,X,Z_orig,Z_tilde,beta_hat, MSE_test, R2_test, MSE_train,\
#      R2_train, var_beta= OLS(xr,yr,Z,order,True,0.5)
#
#     beta_CI(beta_hat,var_beta,0.05, order, True)
#===============================================================================




# sigma2_vec = [0.01,0.1,0.3,0.5,1,2]
# #sigma2_vec = [1]
# maxOrder = 21 #We iterate through all polynomial degrees up to this order
# MSPE_R2(maxOrder,sigma2_vec,False,True) #Use True for first bool to save plot,
# ##True for second to plot calculate MSPE and False for R2v





# #===============================================================================
# #===========Bias-Variance trade-off without resampling==========================
# def bias_var_noResamp(xr,yr,Z,maxOrder,sigma2,scaling):
#     xaxis = np.arange(0,maxOrder,1)
#     bias_vec = np.zeros(maxOrder)
#     variance_vec = np.zeros(maxOrder)
#     bias_vec2 = np.zeros(maxOrder)
#     variance_vec2 = np.zeros(maxOrder)
#     MSEtest_vec = np.zeros(maxOrder)
#     MSEtrain_vec = np.zeros(maxOrder)
#
#     for order in range(0,maxOrder):
#         bias_vec[order],variance_vec[order],X,Z_orig,Z_tilde,beta_hat, MSE_test,\
#         R2_test, MSE_train, R2_train, var_beta= OLS(xr,yr,Z,order,True,sigma2)
#         MSEtrain_vec[order] = MSE_train
#         MSEtest_vec[order] = MSE_test
#
#     # print('Bias: ', bias_vec)
#     # print('Variances: ', variance_vec)
#
#     fig=plt.figure()
#
#     plt.plot(xaxis,bias_vec,'r', label='bias',linestyle='--')
#     plt.plot(xaxis,variance_vec,'b', label = 'variance',linestyle='--')
#     plt.plot(xaxis,MSEtest_vec, 'firebrick', label='MSE_test')
#     #plt.plot(xaxis,bias_vec+variance_vec+sigma2, 'green', label='Sum')
#     # plt.plot(xaxis,MSEtrain_vec, 'steelblue', label='MSE_train')
#     plt.ylabel('MSE')
#     plt.xlabel('Polynomial degree')
#
#     # plt.xlabel(r"$\eta$")
#     # plt.ylabel(r"$E_L$")
#     # plt.title(r"Bias and Variance of our model on test data")
#     plt.title(f"Bias-Variance decomposition for Var($\epsilon$)={sigma2}")
#
#     # Show the minor grid lines with very faint and almost transparent grey lines
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#     plt.tight_layout()
#     plt.savefig(f'BiasVarDecomp_noBoot_sigma2={sigma2}.png')
#     plt.legend()
#     plt.show()
# #===============================================================================
