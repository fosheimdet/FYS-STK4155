import numpy as np
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn as sk
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from random import randrange
import seaborn as sns
from imageio import imread


# from functions import FrankeFunction,desMat,getMSE,getR2,StandardPandascaler,plotter,beta_CI
from functions import OLS,ridge,lasso,FrankeFunction,desMat,getMSE,getR2,addNoise
from regression_methods import linReg
from resampling_methods import bootstrap,crossValidation

from plotting import MasterFunction
from misc import surfacePlotter,beta_CI



def getScores(*args):
    allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']]
    scores = allScores[args[0]]
    for i in range(1,len(args)):
        scores += allScores[args[i]]
    for s in scores:
        while(scores.count(s)>1):
            scores.pop(scores.index(s))
    return scores

def setIndVars(n,maxOrder,order_s, lambda_s, sigmasBool, plotTypeInt):
    #                          ordersBool,lambdasBool
    only_sigmas  =   [sigmasBool,  False,  False]
    errorVSorder  =  [sigmasBool,  True,   False]
    errorVSlambda = [sigmasBool,  False,  True]
    heatmap =       [sigmasBool,  True,   True]

    plotTypes = [only_sigmas,errorVSorder,errorVSlambda,heatmap]

    ordersBool  = plotTypes[plotTypeInt][1]
    lambdasBool = plotTypes[plotTypeInt][2]

    #S = 4/(n**2) #Is equal to 0.01 when n=20
    S = 40/(n**2) #Is equal to 0.1 when n=20
    sigma_v, sigma_s    = [0.5*S,S,2*S,5*S], [S]
    order_v, order_s   = np.arange(0,maxOrder+1,1), order_s
    lexp_v, lexp_s = np.double(np.arange(-15,15+1,1)), lambda_s
    #lambdas = np.logspace(-4, 0, nlambdas)

    sigmas = sigma_v if sigmasBool else sigma_s
    if(ordersBool== False and lambdasBool == False):      #0
        orders = order_s
        loglambdas = lexp_s
    if(ordersBool==True and lambdasBool == False):        #1
        orders = order_v
        loglambdas = lexp_s
    if(ordersBool==False and lambdasBool == True):        #2
        orders = order_s
        loglambdas = lexp_v
    if(ordersBool==True and lambdasBool == True):         #3
        orders = order_v
        loglambdas = np.double(np.arange(-6,1+0.5,0.5))
        #loglambdas = no.logspace(-6,0,10)

    return sigmas,orders,loglambdas


def main():
    np.random.seed(420)

    terrainBool = False
    if(terrainBool):
        terrain = imread('SRTM_data_Norway_2.tif')

        n = 100 #Number of points along the x/y axis
        terrain = terrain[:n,:n]

        x = np.linspace(0,1, np.shape(terrain)[0])
        y = np.linspace(0,1, np.shape(terrain)[1])
        xx, yy = np.meshgrid(x,y)           # Creates mesh of image pixels
        Z = terrain
        z = np.ravel(Z)          #Original noisy
        #Target function(not known in this case). The bias will therefore be estimated
        #by using the original data in place of the target function. Since we won't add further
        #noise to z, f=z throughout the code.
        f = z
        xr = np.ravel(xx)
        yr = np.ravel(yy)

        # Show the terrain
        # plt.figure()
        # plt.title('Terrain over Norway 1')
        # plt.imshow(terrain, cmap='gray')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

    else:
        n = 20 #Number of ticks for x and y axes. We set Sigma to scale as 1/n^2 as discussed in the report

        x = np.linspace(0,1,n)
        y = np.linspace(0,1,n)
        xx, yy = np.meshgrid(x,y)
        Z = FrankeFunction(xx,yy)
        z = np.ravel(Z)         #Original z values of Franke's function without noise
        f = z                   #Target function. In this case it is know, and so we can calculate the true bias
        xr = np.ravel(xx)
        yr = np.ravel(yy)



    #===============================================================================
    #===============================================================================


    scaling = False
    skOLS = False  #Use sklearn in OLS (rather than pseudoinverse)?
    skCV = False   #Use sklearn's cross validation contra own code?

    nBoot = 10 #Number of bootstrap samples
    K = 5   #Number of folds in cross validation alg.
    shuffle = True  #Shuffle the data before performing folds?


#================================================================================================================================================
#=============== Perform surface plot of either chosen data (terrain or Franke function) ========================================================
    m_sigma = 0.1
    if(terrainBool):
        m_sigma = 0
    m_scaling = False
    m_skOLS = False
    m_lambda =  1e-3
    m_regMeth = 'OLS'
    m_savePlot = False
    m_order = 5

    alpha = 0.05

    X = desMat(xr,yr,m_order)
    #regMeth = regMethods[1]

    scorevalues,Z_orig,Z_tilde,beta_hat,var_hat\
     = linReg(m_regMeth,getScores(4,5),X,z,m_sigma,m_scaling,m_skOLS,m_lambda)

    #             (xx,yy,Z_orig,Z_tilde=0,savePlot=False,order=5,sigma=0,regMeth="OLS",lmd=0)
    surfacePlotter(xx,yy,Z_orig,Z_tilde,m_savePlot,m_order,m_sigma,m_regMeth,m_lambda)
    #beta_CI(beta_hat,var_hat,alpha,m_order,False)
#================================================================================================================================================


#================================================================================================================================================
#========================================  MODEL ASSESSMENT  =============================================================================================
#================================================================================================================================================
    #Here we set the parameters for the various resampling methods.
    no_resamp = ['no_resamp.', scaling,skOLS,xr,yr,z]
    bootstrap= ['bootstrap',   scaling,skOLS,xr,yr,z,nBoot]
    crossval = ['crossval',    scaling,skOLS,xr,yr,z,f,K,shuffle,skCV]

    # bootstrap = no_resamp.append([nBoot])
    # crossval = no_resamp.append([f,K,shuffle,skCV])

    #Nested list. Choose which of the elements of resampMethods to use as our parameter
    resampMethods = [no_resamp, bootstrap, crossval]       #List which again determines what resampling method to run.
    regMethods = ['OLS','ridge','lasso']                   #Choose between regression methods
    # graphs = ['bias','variance','MSEtest','MSEtrain','R2test','R2train'] #Which regression scores one can plot




#================================================================================================================================================
#========================================== CONTROL PANEL =======================================================================================

    sigmasBool = False  #If true, will produce a plot for each std. in sigma_v
    maxOrder = 20       #Will make order vector from 0 to maxOrder
    order_s  = [10]     #Default polynomial degree
    lambda_s  = [-100.] #Default log(lambda) value, i.e. lambda=0
    #lambda_s = [100000]

                  # [ordersBool lamdasBool] = plotTypeInt
    plotTypeInt =              1
                  # [  False   0   False  ]  Plots nothing, see returned list
                  # [  True    1   False  ]  Plots error vs. pol.deg.
                  # [  False   2   True   ]  Plots error vs. lambda
                  # [  True    3   True   ]  Produces heatmap
    indVars= setIndVars(n,maxOrder,order_s,lambda_s,sigmasBool, plotTypeInt)

    savePlot = False
    plotBool = True #Make plots?

    resampInt = 0   #0=no_resamp., 1=bootstrap, 2=crossval
    regInt    = 0   #0=OLS,        1=ridge,     2=lasso

    allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']] #Which regression scores one can plot
    #               0           1           2           3           4           5
    #Sets which scores to calculate by passing in the corresponding index from allScores.
    #Function def before main().
    scoresNames = getScores(4,5)

    #Produce plot(s) and return score results. They will be stored in the same order as the arguments of getScores()
    scoreResults = MasterFunction(savePlot,plotBool,resampMethods[resampInt],regMethods[regInt],scoresNames,indVars)

#================================================================================================================================================
#================================================================================================================================================

    # #Makes surface plot of the Franke function fit and plots the regression parameters
    # #with CI

# #================================================================================================================================================
    def surfAndCI():
        m_sigma = indVars[0][-1]
        print(m_sigma)
        m_scaling = scaling
        m_skOLS = skOLS
        m_lambda =  indVars[2][0]
        m_regMeth = 'OLS'
        m_savePlot = False
        m_order = order_s[0]

        alpha = 0.05

        X = desMat(xr,yr,m_order)
        #regMeth = regMethods[1]

        scorevalues,Z_orig,Z_tilde,beta_hat,var_hat\
         = linReg(m_regMeth,getScores(4,5),X,z,m_sigma,m_scaling,m_skOLS,m_lambda)

        surfacePlotter(xx,yy,Z_orig,Z_tilde,m_savePlot,m_order,m_sigma,m_regMeth,m_lambda)
        #beta_CI(beta_hat,var_hat,alpha,m_order,False)
    #surfAndCI()

#================================================================================================================================================




#------------------------------------------------------------------------------------------------------------------------------------------------
#                           Set plotTypeInt = 1 in the following section
#------------------------------------------------------------------------------------------------------------------------------------------------

    plotTypeInt =                             1
    indVars= setIndVars(n,maxOrder,order_s,lambda_s,sigmasBool, plotTypeInt)
    savePlot = True

#To reproduce a desired result, uncomment the corresponding function
#=====================================================================================================
#                                       Bias-variance
#=====================================================================================================
#For different resampling methods:
#no_resamp:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[0],getScores(0,1),indVars)
#bootstrap:
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[0],getScores(0,1),indVars)
#crossval:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[0],getScores(0,1),indVars)


#For differen regression methods (using Cross Validation)
#OLS:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[0],getScores(0,1),indVars)
#ridge:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[1],getScores(0,1),indVars)
#lasso:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[2],getScores(0,1),indVars)
#=====================================================================================================


#=====================================================================================================
#                                  MSEtest VS MSEtrain
#=====================================================================================================
    #For different resampling methods:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[0],getScores(2,3),indVars)
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[0],getScores(2,3),indVars)
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[0],getScores(2,3),indVars)

    #For differen regression methods (using Cross Validation)
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[0],getScores(2,3),indVars)
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[1],getScores(2,3),indVars)
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[2],getScores(2,3),indVars)
#=====================================================================================================


#=====================================================================================================
#                                  R2test VS R2train
#=====================================================================================================
#====================================================================================
#For different resampling methods:
#====================================================================================
#no_resamp:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[0],getScores(4,5),indVars)
#bootstrap:
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[0],getScores(2,3),indVars)
#crossval:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[0],getScores(2,3),indVars)
#====================================================================================
#For differen regression methods(using Cross Validation):
#====================================================================================
#OLS:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[0],getScores(2,3),indVars)
#ridge:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[1],getScores(2,3),indVars)
#lasso:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[2],getScores(2,3),indVars)
#=====================================================================================================
#=====================================================================================================


#------------------------------------------------------------------------------------------------------------------------------------------------
#                           Set plotTypeInt = 2 in the following section
#------------------------------------------------------------------------------------------------------------------------------------------------
    plotTypeInt =                             2
    indVars= setIndVars(n,maxOrder,order_s,lambda_s,sigmasBool, plotTypeInt)
    savePlot = False

#=====================================================================================================
#         MSEtest vs MSEtrain scores for Ridge and Lasso plotted against lambda
#=====================================================================================================
#====================================================================================
#Ridge for different resampling methods:
#====================================================================================
#no_resamp:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[1],getScores(2,3),indVars)
#bootstrap:
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[1],getScores(2,3),indVars)
#crossval:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[1],getScores(2,3),indVars)
#====================================================================================
#Lasso for different resampling methods:
#====================================================================================
#no_resamp:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[2],getScores(2,3),indVars)
#bootstrap:
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[2],getScores(2,3),indVars)
#crossval:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[2],getScores(2,3),indVars)
#=====================================================================================================
#=====================================================================================================



#------------------------------------------------------------------------------------------------------------------------------------------------
#                           Set plotTypeInt = 3 in the following section
#------------------------------------------------------------------------------------------------------------------------------------------------
    plotTypeInt =                             3
    indVars= setIndVars(n,maxOrder,order_s,lambda_s,sigmasBool, plotTypeInt)
    savePlot = False

#=====================================================================================================
#            Heat map of MSEtest vs MSEtrain scores for Ridge and Lasso
#=====================================================================================================
#====================================================================================
#Ridge for different resampling methods:
#====================================================================================
#no_resamp:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[1],getScores(2,3),indVars)
#bootstrap:
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[1],getScores(2,3),indVars)
#crossval:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[1],getScores(2,3),indVars)
#====================================================================================
#Lasso for different resampling methods:
#====================================================================================
#no_resamp:
    #MasterFunction(savePlot,True,resampMethods[0],regMethods[2],getScores(2,3),indVars)
#bootstrap:
    #MasterFunction(savePlot,True,resampMethods[1],regMethods[2],getScores(2,3),indVars)
#crossval:
    #MasterFunction(savePlot,True,resampMethods[2],regMethods[2],getScores(2,3),indVars)
#=====================================================================================================
#=====================================================================================================



if __name__ =="__main__":
    main()
