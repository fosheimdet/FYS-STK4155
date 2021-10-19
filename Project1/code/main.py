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

from do_regression import doRegression
from plotters import scorePlotter,surfacePlotter,beta_CI
from reformat_variables import reformatVariables, generateDesMatPars


def getScores(*args):
    dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']

    emptyScoreScalars = {}
    for i in range(0,len(args)):
        scoreName = dummyList[args[i]]
        emptyScoreScalars[scoreName]=0
    print("Following scores have been requested:\n", emptyScoreScalars,'\n' )

    return emptyScoreScalars

# def getScores(*args):
#     allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']]
#     # scores = allScores[args[0]]
#     scores = []
#     for i in range(0,len(args)):
#         scores += allScores[args[i]]
#     #print(scores)
#     # for s in scores:
#     #     while(scores.count(s)>1):
#     #         scores.pop(scores.index(s))
#
#     print("Following scores requested:\n", scores,'\n' )
#
#     return scores


#==============================================================================================================================
#================================ MAIN FUNCTION ===============================================================================
#==============================================================================================================================
def main():
    # xr,yr,z,f=generateDesMatPars(False,1000)
    # surfacePlotter(False,False,xr,yr,z)
    np.random.seed(420)
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------  CONTROL PANEL ---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    #This is where all the variables are set
    ONbutton   = True    #Set to False to run main.py without performing regression with the parameters specified below
    tinkerBool = False    #If True, a folder named 'tinkerings' will be created if savePlot is True, in which the plots will be saved based on reg. and resamp. method.
    exercise = 1          #If tinkerBool=False, a folder named 'exercise_results' if savePlot is True. This int determines which subfolder to save to.

    terrainBool = False  #Use terrain data or Franke function?
    n_t = 1000            #How many points on the x and y axes if using terrain data
    n_f = 40             #How many points on the x and y axes if using FrankeFunction

    origSurfPlot = False #Plot original data w. no noise before doing regression? Will be set to False for terrain data
    showResults  = True #Plot results based on plotTypeInt?
    doublePlot = True   #Plot the data side by side with the fitted surface?
    plotOrig = False     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.

    savePlot = False #Save plots?


    scaling = True     #Scale the design matrix, X?
    skOLS = False       #Use sklearn in OLS (rather than pseudoinverse)?
    skCV = False        #Use sklearn's cross validation contra own code?

    plotBetaCI  = False  #Plot confidence intervals for beta_hat? Only works for plotTypeInt=0
    alpha       = 0.05   #Significance level for confidence intervals
    #---------------------------------------------------------------------------
    #-----------------------Choose hyperparamets -------------------------------
    #---------------------------------------------------------------------------
    S = 0 if terrainBool else 40/(n_f**2)     #Is equal to 0.1 when n=20. Ensures sigma scales 'correctly?'with n, discussed in report.
    S = 2*S
    sigma_v  =       [0.1*S, 1*S, 2*S, 10*S]  #Make a separate plot for each of these sigmas
    #sigma_v  =       [0.1*S, 1*S, 2*S, 10*S]
    sigma_s  =               [2*S]        #Default standard deviation of epsilon
    #------------------------------------------
    minOrder,maxOrder =       0, 20         #Will make order vector from minOrder to maxOrder
    order_s =                 [10]           #Default pol.degree if we don't plot vs. degrees
    #------------------------------------------
    minLoglmd, maxLoglmd =    -15,15      #Will make log(lambda) vector from minLoglmd to maxLoglmd
    lambda_s  =               [-3]        #Default lambda value. Must be set
    #---------------------------------------------------------------------------
    #------------------------ Choose type of plot-------------------------------
    #---------------------------------------------------------------------------
    sigmasBool = True #If true, will produce a plot for each element in sigma_v.
    #Will be set to False for terrain data during reformating

                  # [ordersBool lamdasBool] = plotTypeInt
    plotTypeInt =              1
                  # [  False   0   False  ]  Generate surface plot(s) and print results if using no_resamp.
                  # [  True    1   False  ]  Plots error vs. pol.deg.
                  # [  False   2   True   ]  Plots error vs. lambda
                  # [  True    3   True   ]  Produces heatmap(s) of error(s)
    # origSurfPlot = True #Plot original data w. no noise before doing regression? Will be set to False for terrain data
    # showResults  = False #Plot results based on plotTypeInt?
    # savePlot = True #Save plots?
    #---------------------------------------------------------------------------
    #Choose resampling technique, regression method and what scores to calculate
    #---------------------------------------------------------------------------
    resampInt = 0 #0=no_resamp., 1=bootstrap, 2=crossval
    regInt    = 1 #0=OLS,        1=ridge,     2=lasso

    allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']] #Which regression scores one can plot
    #               0           1           2           3           4           5
    scoreNames = getScores(0,1)
    #Sets which scores to calculate by passing in the corresponding indices from allScores.

    nBoot = 20         #Number of bootstrap samples
    K = 5               #Number of folds in cross validation alg.
    shuffle = True      #Shuffle the data before performing crossval folds?
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    dataStr  =  'Using TERRAIN data' if terrainBool else 'Using FRANKE function'
    tinkrStr = 'TINKER mode activated' if tinkerBool else 'Exercise mode active'
    savePlotStr = 'Plot saving ON' if savePlot else 'Plot saving OFF'
    print('\t\t\t',dataStr, '\n')
    print('\t\t\t',tinkrStr,'\n')
    print('\t\t\t',savePlotStr,'\n\n')
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------- TRANSFORM/REFORMAT ALL THE SET VARIABLES --------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    print('Reformatting variables\n')

    n = n_t if terrainBool else n_f
    if(terrainBool):sigmasBool = False

    resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(\
    \
     tinkerBool,exercise\
    ,terrainBool,n_t,n_f\
    ,scaling,skOLS,skCV\
    ,plotBetaCI,alpha\
    ,sigma_v,sigma_s\
    ,minOrder,maxOrder,order_s\
    ,minLoglmd, maxLoglmd, lambda_s\
    ,sigmasBool,plotTypeInt,origSurfPlot,savePlot\
    ,resampInt,regInt,allScores,scoreNames\
    ,nBoot,K,shuffle)

    #This is done using lists.
    #We do this to avoid functions with too many arguments and to simplify the code
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------  CALCULATE RESULTS  --------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    if(ONbutton):
        #print('Performing regression and calculating ', scoresNames,'\n')
        # print('scoreNames after reformatting, before regression: ', scoreNames,'\n')
        t_start = time.time()
        # dic       dict    list     list     list      list
        scoreRes,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=doRegression(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('doRegression time: ', t_end-t_start,'sec\n')

#================================================================================================================================================
#--------------------------------  SHOW RESULTS  ------------------------------------------------------------------------------------------------
#================================================================================================================================================
        sigmas,orders,lambdas = hyperPars
        xr,yr=generateDesMatPars(terrainBool,n)[0:2]
        if(showResults == True):
            #print("Plotting results")
            if(plotTypeInt == 0 and resampInt == 0):
                for s,sigma in enumerate(sigmas):
                    if(doublePlot == True or plotOrig == True):
                        surfacePlotter(scoreRes,calcAtts,doublePlot,plotOrig,terrainBool,tinkerBool\
                        ,exercise,savePlot,z_noisyL[s],z_fittedL[s],sigmas[s],s)

                    if(plotBetaCI==True):
                        beta_CI(tinkerBool,savePlot,beta_hat,var_beta,alpha,orders[0])

                print('\nRegression scores for', 'sigma = ', sigmas, ': ')
                for scoreName in scoreRes:
                    print(f"{scoreName+':':<18}{scoreRes[scoreName]}")

            else:
                print("Plotting scores vs hyperparameter(s)")
                scorePlotter(scoreRes,calcAtts,terrainBool,tinkerBool,exercise,savePlot)
                print('\nRegression scores for', 'sigma = ', sigmas, ': ')
                for scoreName in scoreRes:
                    print(f"{scoreName+':':<18}{scoreRes[scoreName]}")
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================



if __name__ =="__main__":
    main()
