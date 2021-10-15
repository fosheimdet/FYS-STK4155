import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import norm

from functions import addNoise,desMat
from regression_methods import linReg
from resampling_methods import crossValidation, bootstrap
from sklearn.model_selection import train_test_split




#===============================================================================
#Makes surface plots of both the fitted function and the original Franke function,
#side by side
#===============================================================================
# def FrankePlotter(savePlot,xx,yy,Z_orig, Z_tilde, order,sigma,regMeth,lamda):
def surfacePlotter(xx,yy,Z_orig,Z_tilde=0,savePlot=False,order=5,sigma=0,regMeth="OLS",lmd=0):
    n = Z_orig.shape[0]

    fig = plt.figure(figsize=(8,4))
    #First plot
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(xx,yy,Z_orig, cmap = cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title(r"Original data"+"\n"+ r"$\sigma_{\epsilon}$" + f"= {sigma}, n={n}")
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
    surf = ax.plot_surface(xx,yy,Z_tilde, cmap = cm.coolwarm, alpha=1,\
    linewidth=0, antialiased=False)
    titleStr = ''
    if(regMeth=='ridge' or regMeth=='lasso'):
        titleStr = f' and $\lambda=${lamda}'
    ax.set_title(regMeth+f' fit, p = {order}'+titleStr)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(20, 30)
    #fig.colorbar(surf, shrink=0.5, aspect = 5)
    if(savePlot):
        plt.savefig(f'surfacePlot/linReg_surface_order={order}_Var($\epsilon$)={vareps}_n={n}.png')
    plt.show()
#===============================================================================

#
# #===============================================================================
# #Makes surface plots of both the fitted function and the original Franke function,
# #side by side
# #===============================================================================
# def FrankePlotter(savePlot,xx,yy,Z_orig, Z_tilde, order,sigma,regMeth,lam):
#     n = Z_orig.shape[0]
#
#     fig = plt.figure(figsize=(8,4))
#     #First plot
#     ax = fig.add_subplot(121, projection='3d')
#     surf = ax.plot_surface(xx,yy,Z_orig, cmap = cm.coolwarm, linewidth=0, antialiased=False)
#     ax.set_title(r"Original data"+"\n"+ r"$\sigma_{\epsilon}$" + f"= {sigma}, n={n}")
#     #Customize the z axis.
#     ax.set_zlim(-0.10, 1.40)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.view_init(20, 30)
#     # Add a color bar which maps values to colors
#     #fig.colorbar(surf, shrink=0.5, aspect = 5)
#
#     #Second plot
#     ax = fig.add_subplot(122, projection='3d')
#     #ax.scatter(xs = heights, ys = weights, zs = ages)
#     surf = ax.plot_surface(xx,yy,Z_tilde, cmap = cm.coolwarm, alpha=1,\
#     linewidth=0, antialiased=False)
#     titleStr = ''
#     if(regMeth=='ridge' or regMeth=='lasso'):
#         titleStr = f' and $\lambda=${lam}'
#     ax.set_title(regMeth+f' fit, p = {order}'+titleStr)
#     ax.set_zlim(-0.10, 1.40)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.view_init(20, 30)
#     #fig.colorbar(surf, shrink=0.5, aspect = 5)
#     if(savePlot):
#         plt.savefig(f'surfacePlot/linReg_surface_order={order}_Var($\epsilon$)={vareps}_n={n}.png')
#     plt.show()
# #===============================================================================

#===============================================================================
#====Calculates the confidence intervals of beta_hat and plots them=============
#===============================================================================
def beta_CI(beta_hat,var_beta,alpha,order,save):
    Z_CI = norm.ppf(1-alpha/2)
    #95% confidence intervals of beta
    sigBeta=np.sqrt(var_beta)
    beta_lower = beta_hat - Z_CI*sigBeta
    beta_upper = beta_hat + Z*sigBeta
    beta_CI=[Z_CI*sigBeta,Z_CI*sigBeta]
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


#
# #===============================================================================
# #====Calculates the confidence intervals of beta_hat and plots them=============
# #===============================================================================
# def beta_CI(beta_hat,var_beta,alpha,order,save):
#     Z = norm.ppf(1-alpha/2)
#     #95% confidence intervals of beta
#     sigBeta=np.sqrt(var_beta)
#     beta_lower = beta_hat - Z*sigBeta
#     beta_upper = beta_hat + Z*sigBeta
#     beta_CI=[Z*sigBeta,Z*sigBeta]
#     beta_indices = np.arange(0,len(beta_hat),1)
#
#     fig = plt.figure()
#     plt.errorbar(beta_indices,beta_hat,beta_CI, fmt='o',color = 'darkgreen',
#     ecolor = 'darkgreen', capsize = 3,capthick = 2 ,barsabove = True, ms=5)
#
#     plt.xlabel(r"$\beta$ index")
#     plt.ylabel(r"$\beta$ value")
#     orderStr= 'th'
#     if(order==1):
#         orderStr = 'st'
#     elif(order==2):
#         orderStr = 'nd'
#     elif(order==3):
#         orderStr = 'rd'
#     plt.title(f"Regression coefficients with their 95% CI for {order}"+orderStr+ " order pol.")
#     # Show the minor grid lines with very faint and almost transparent grey lines
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
#     plt.tight_layout()
#     #plt.savefig(datafile_path+ofile+"_"+solv+"_"+dist+'_nh_'+str(nh)+"_RBM_"+str(RBM)+'_.pdf')
#     #plt.legend()
#     if(save == True):
#         plt.savefig(f'CI_beta_order{order}.png')
#     plt.show()
# #===============================================================================
