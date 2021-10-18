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

from calculate_errors import scoreCalculator
from plotters import scorePlotter,surfacePlotter,beta_CI
#from do_project import doProject




def generateDesMatPars(terrainBool,n):
    if(terrainBool):
        #Get terrain data:
        terrain = imread('SRTM_data_Norway_1.tif')
        terrain = terrain[:n,:n]
        x = np.linspace(0,1, np.shape(terrain)[0])
        y = np.linspace(0,1, np.shape(terrain)[1])
        xx, yy = np.meshgrid(x,y)           # Creates mesh of image pixels
        Z = terrain
        z = np.ravel(Z)#Has inherent noise
    else:
        x = np.linspace(0,1,n)
        y = np.linspace(0,1,n)
        xx, yy = np.meshgrid(x,y)
        Z = FrankeFunction(xx,yy)
        z= np.ravel(Z)  #Original z values of Franke's function without noise

    xr = np.ravel(xx)
    yr = np.ravel(yy)
    # f = z #Target function. In the case of Franke, it is know, and so we can calculate the true bias
    return (xr,yr,z)

def plotter():
    scorePlotter(terrainBool,savePlot,calcRes,calcAtts) #Plots the result
    if(sigmasBool == False and plotTypeInt == 0):
        savePlot = False
        z_temp = z
        sigma,order, lmd = hyperPars
        z = addNoise(z_temp,hyperPars[0][0])
        Z = z.reshape(n,n)
        surfacePlotter(xx,yy,Z,savePlot,Z_tilde,order[0],sigma[0],regMeth,lmd[0])

def getScores(*args):
    allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']]
    scores = allScores[args[0]]
    for i in range(1,len(args)):
        scores += allScores[args[i]]
    for s in scores:
        while(scores.count(s)>1):
            scores.pop(scores.index(s))
    return scores

# Sets sigma, the polynomial degree (order) and lambda to be either
# a range of values (hyperPar_v) or just one default value, hyperPar_s
def setHyperPars(n,setSigs,setOrds,setLmds,sigmasBool, plotTypeInt):
    #               sigmasBool, ordersBool , lambdasBool
    only_sigmas   =[sigmasBool,   False    ,   False   ]
    errorVSorder  =[sigmasBool,   True     ,   False   ]
    errorVSlambda =[sigmasBool,   False    ,   True    ]
    heatmap       =[sigmasBool,   True     ,   True    ]

    plotTypes = [only_sigmas,errorVSorder,errorVSlambda,heatmap]

    ordersBool  = plotTypes[plotTypeInt][1]
    lambdasBool = plotTypes[plotTypeInt][2]

    sigma_v, sigma_s    = setSigs

    order_s, minOrder, maxOrder = setOrds
    order_v  = np.arange(minOrder,maxOrder+1,1)

    lambda_s, minLoglmd, maxLoglmd = setLmds
    nLambdas= 2*(np.abs(minLoglmd-maxLoglmd))  #Number of elements for the lambda vector
    lambda_v = np.logspace(minLoglmd, maxLoglmd+1, nLambdas )
    # loglambda_v = np.arange(minLoglmd,maxLoglmd+0.2,0.2)
    # lambda_v = pow(10,loglambda_v)
    lmdInc = 0.5
    lambda_v = np.arange(minLoglmd,maxLoglmd+lmdInc,lmdInc)

    sigmas = sigma_v if sigmasBool else sigma_s
    if(ordersBool== False and lambdasBool == False):      #0
        orders =  order_s
        lambdas = lambda_s
    if(ordersBool==True and lambdasBool == False):        #1
        orders = order_v
        lambdas= lambda_s
    if(ordersBool==False and lambdasBool == True):        #2
        orders = order_s
        lambdas= lambda_v
    if(ordersBool==True and lambdasBool == True):         #3
        orders = order_v
        lambdas= lambda_v

    return sigmas,orders,lambdas

#==============================================================================================================================
#================================ MAIN FUNCTION ===============================================================================
#==============================================================================================================================
def main():
    # xr,yr,z,f=generateDesMatPars(False,1000)
    # surfacePlotter(False,False,xr,yr,z)
    np.random.seed(420)
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#========================================== CONTROL PANEL =======================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    ONbutton   = True    #Set to False to run main.py without performing regression with the parameters specified below
    tinkerBool = False    #If True, a folder named 'tinkerings' will be created if savePlot is True, in which the plots will be saved based on reg. and resamp. method.
    exercise = 6          #If tinkerBool=False, a folder named 'exercise_results' if savePlot is True. This int determines which subfolder to save to.

    terrainBool = True  #Use terrain data or Franke function?
    n_t = 200            #How many points on the x and y axes if using terrain data
    n_f = 20             #How many points on the x and y axes if using FrankeFunction
    n = n_t if terrainBool else n_f
    scaling = True     #Scale the design matrix, X?
    skOLS = False       #Use sklearn in OLS (rather than pseudoinverse)?
    skCV = False        #Use sklearn's cross validation contra own code?

    plotBetaCI  = False  #Plot confidence intervals for beta_hat? Only works for plotTypeInt=0
    alpha       = 0.05   #Significance level for confidence intervals

    nBoot = 400         #Number of bootstrap samples
    K = 5               #Number of folds in cross validation alg.
    shuffle = True      #Shuffle the data before performing crossval folds?

    #------------------------------------------
    S = 0 if terrainBool else 40/(n**2)#Is equal to 0.1 when n=20. Ensures sigma scales 'correctly?'
    # with n, discussed in report.
    #S = 40/(n**2)
    sigma_v  =       [0.1*S, 1*S, 2*S, 10*S]  #Make a separate plot for each of these sigmas
    #sigma_v  =       [0.1*S, 1*S, 2*S, 10*S]
    sigma_s  =               [2*S]        #Default standard deviation of epsilon
    #------------------------------------------
    minOrder,maxOrder =       1, 20         #Will make order vector from minOrder to maxOrder
    order_s =                 [10]           #Default pol.degree if we don't plot vs. degrees
    #------------------------------------------
    minLoglmd, maxLoglmd =   -15,15       #Will make log(lambda) vector from minLoglmd to maxLoglmd
    lambda_s  =               [-1.21]        #Default lambda value. Must be set
    #------------------------------------------
    sigmasBool = False  #If true, will produce a plot for each std. in sigma_v

                  # [ordersBool lamdasBool] = plotTypeInt
    plotTypeInt =              1
                  # [  False   0   False  ]  Generate surface plot(s) and print results if using no_resamp.
                  # [  True    1   False  ]  Plots error vs. pol.deg.
                  # [  False   2   True   ]  Plots error vs. lambda
                  # [  True    3   True   ]  Produces heatmap
    savePlot = True #Save plots?
    plotBool = True #Make error/score plots?

    resampInt = 0 #0=no_resamp., 1=bootstrap, 2=crossval
    regInt    = 0   #0=OLS,        1=ridge,     2=lasso

    allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']] #Which regression scores one can plot
    #               0           1           2           3           4           5
    scoresNames = getScores(0,1,2)
    #Sets which scores to calculate by passing in the corresponding index from allScores.
    #Function def before main().
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================

    # xr,yr,z,f=generateDesMatPars(False,1000)
    # surfacePlotter(False,False,xr,yr,z)

#================================================================================================================================================
#======================= USE CONTROL PANEL VALUES TO GENERATE/SET ALL NEEDED VARIABLES ==========================================================
    if(terrainBool):sigmasBool = False
    #This is done using lists.
    #We do this to avoid functions with too many arguments and to simplify the code


#---------------------------------------------------------------------------------------------------------------------------------------------
#--------------------- GENERATE xr,yr,z,f ----------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

    xr,yr,z = generateDesMatPars(terrainBool,n)

#---------------------------------------------------------------------------------------------------------------------------------------------
#--------- SET RESAMPLING VARIABLES ACCORDING TO resampInt(0,1 or 2)-------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
    no_resamp = ['no_resamp.', xr,yr,z,scaling,skOLS]
    bootstrap= ['bootstrap',   xr,yr,z,scaling,skOLS,nBoot]
    crossval = ['crossval',    xr,yr,z,scaling,skOLS,skCV,shuffle,K]
    resampMethods = [no_resamp, bootstrap, crossval]

    resampMeth = resampMethods[resampInt]
#---------------------------------------------------------------------------------------------------------------------------------------------
#------- SET REGRESSION STRING ACCORDING TO regInt(0,1 or 2)---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
    regMethods = ['OLS','ridge','lasso']

    regMeth = regMethods[regInt]
#---------------------------------------------------------------------------------------------------------------------------------------------
#--------SET HYPERPARAMETERS, i.e sigmas,orders and lambdas-----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
    setSigs= [sigma_v,sigma_s]
    setOrds= [order_s, minOrder, maxOrder]
    setLmds= [lambda_s,minLoglmd,maxLoglmd]

    hyperPars= setHyperPars(n,setSigs,setOrds,setLmds,sigmasBool, plotTypeInt)
#================================================================================================================================================
#================================================================================================================================================
#Finished generating/setting variables



#Run scoreCalculator to produce and (save?) plot(s) and return the score results
#as matrices. The results for a particular score will be stored in the same order
#as the arguments of the getScores() function above.

#================================================================================================================================================
#--------------------------  CALCULATE RESULTS  ------------------------------------------------------------------------------------------------
#================================================================================================================================================
#================================================================================================================================================
    if(ONbutton):
        dataStr  =  'Using TERRAIN data' if terrainBool else 'Using FRANKE function'
        tinkrStr = 'TINKER mode activated' if tinkerBool else 'Exercise mode active'
        savePlotStr = 'Plot saving ON' if savePlot else 'Plot saving OFF'
        print( dataStr, '\n')
        print(tinkrStr,'\n')
        print(savePlotStr,'\n')


        t_start = time.time()
        # dic       dict    list     list     list      list
        scoreRes,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=scoreCalculator(resampMeth,regMeth,scoresNames,hyperPars)
        t_end = time.time()

        print('scoreCalculator time: ', t_end-t_start,'\n')
#================================================================================================================================================
#--------------------------------  PLOT RESULTS  -------------------------------------------------------------------------------------------------------
#================================================================================================================================================
        sigmas,orders,lambdas = hyperPars
        regMeth = regMethods[regInt]     #Unpack chosen regression method name
        if(plotBool==True):
            if(plotTypeInt==0 and resampInt == 0):
                for s,sigma in enumerate(sigmas): #Loop through in reversed order to have the low sigma plots pop up first
                    surfacePlotter(tinkerBool,savePlot,xr,yr,z_noisyL[s],z_fittedL[s],sigmas[s],orders[0],lambdas[0],regMeth)
                                   #tinkerMode,savePlot,xr,yr,z_orig,z_tilde,sigma,order,lmd,regmeth)
                    if(plotBetaCI==True):
                        beta_CI(tinkerBool,savePlot,beta_hat,var_beta,alpha,orders[0])

                print('\nRegression scores for', 'sigma = ', sigmas, ': ')
                for scoreName in scoreRes:
                    print(f"{scoreName+':':<18}{scoreRes[scoreName]}")

            else:               #scoreRes,calcAtts,terrainBool,tinkerBool,exercise,savePlot
                scorePlotter(scoreRes,calcAtts,terrainBool,tinkerBool,exercise,savePlot)
#================================================================================================================================================
#--------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================

if __name__ =="__main__":
    main()
