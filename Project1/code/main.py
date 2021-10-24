import numpy as np
import time
import matplotlib.pyplot as plt

from perform_regressions import performRegressions
from plotters import scorePlotter,surfacePlotter,beta_CI
from reformat_variables import reformatVariables, generateDesMatPars


def getScores(*args):
    dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
    #dummyList = ['bias_test','bias_train','var_test','var_train','MSEtest','MSEtrain','R2test','R2train']
    emptyScoreScalars = {}
    for i in range(0,len(args)):
        scoreName = dummyList[args[i]]
        emptyScoreScalars[scoreName]=0
    print("The following scores are being calculated:\n", emptyScoreScalars,'\n' )
    return emptyScoreScalars


#==============================================================================================================================
#================================ MAIN FUNCTION ===============================================================================
#==============================================================================================================================
def main():
    # xr,yr,z,f=generateDesMatPars(False,1000)
    # surfacePlotter(False,False,xr,yr,z)
    #np.random.seed(420)
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------  CONTROL PANEL ---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    #This is where all the variables are set
    ONbutton   = True    #Set to False to run main.py without performing regression with the parameters specified below
    tinkerBool = False    #If True, a folder named 'tinkerings' will be created if savePlot is True, in which the plots will be saved based on reg. and resamp. method.

    terrainBool = False  #Use terrain data or Franke function?
    n_t = 40            #How many points on the x and y axes if using terrain data
    n_f = 40             #How many points on the x and y axes if using FrankeFunction

    origSurfPlot = False #Plot original data w. no noise before doing regression? Will be set to False for terrain data
    showResults  = True #Plot results based on plotTypeInt?
    doublePlot = True   #Plot the data side by side with the fitted surface?
    plotOrig = False     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.

    scaling = False     #Scale the design matrix, X?
    skOLS = False       #Use sklearn in OLS (rather than pseudoinverse)?
    skCV = False        #Use sklearn's cross validation contra own code?
    plotBetaCI  = False  #Plot confidence intervals for beta_hat? Only works for plotTypeInt=0
    alpha       = 0.05   #Significance level for confidence intervals
    #---------------------------------------------------------------------------
    #-----------------------Choose hyperparamets -------------------------------
    #---------------------------------------------------------------------------
    S = 0 if terrainBool else 40/(n_f**2)     #Is equal to 0.1 when n=20. Ensures sigma scales 'correctly?'with n, discussed in report.
    S = 4*S
    sigma_v  =       [0.1*S, 1*S, 2*S, 10*S]  #Make a separate plot for each of these sigmas
    #sigma_v  =       [0.01*S,0.1*S,0.5*S, 1*S, 2*S,5*S, 10*S,20*S]
    sigma_s  =               [2*S]        #Default standard deviation of epsilon
    #------------------------------------------
    minOrder,maxOrder =       2,20        #Will make order vector from minOrder to maxOrder
    order_s =                 [20]           #Default pol.degree if we don't plot vs. degrees
    #------------------------------------------
    minLoglmd, maxLoglmd =    -7,-2   #Will make log(lambda) vector from minLoglmd to maxLoglmd
    lambda_s  =               [-6]        #Default lambda value. Must be set
    #---------------------------------------------------------------------------
    #------------------------ Choose type of plot-------------------------------
    #---------------------------------------------------------------------------
    sigmasBool = False #If true, will produce a plot for each element in sigma_v.
    #Will be set to False for terrain data during reformating

                  # [ordersBool lamdasBool] = plotTypeInt
    plotTypeInt =              3
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
    resampInt = 1 #0=no_resamp., 1=bootstrap, 2=crossval
    regInt    = 2 #0=OLS,        1=ridge,     2=lasso

    dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
    #              0        1          2          3         4        5
    #Sets which scores to calculate by passing in the corresponding indices from allScores.
    emptyScoreScalars=getScores(2,4)
    nBoot =30         #Number of bootstrap samples
    K = 5               #Number of folds in cross validation alg.
    shuffle = True      #Shuffle the data before performing crossval folds?
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    dataStr  =  'Using TERRAIN data' if terrainBool else 'Using FRANKE function'
    tinkrStr = 'TINKER mode activated' if tinkerBool else 'Exercise mode active'
    print('\t\t\t',dataStr, '\n')
    print('\t\t\t',tinkrStr,'\n')
    saveInput = input("Save plot(s)?y/n: ")
    savePlot = True if saveInput == 'y' else False
    savePlotStr = 'Plot saving ON' if savePlot else 'Plot saving OFF'
    print('\t\t\t',savePlotStr,'\n\n')
    if(tinkerBool == False and savePlot ==True):
        exercise = input("Please select exercise number:")
    else:
        exercise = 0
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------- TRANSFORM/REFORMAT ALL THE SET VARIABLES --------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    print('\nReformatting variables\n')

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
    ,resampInt,regInt,emptyScoreScalars\
    ,nBoot,K,shuffle)

    #This is done using lists.
    #We do this to avoid functions with too many arguments and to simplify the code
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------  CALCULATE RESULTS  --------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
    if(ONbutton):
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        #      dic        dict    list     list     list     list
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time spent: ', t_end-t_start,'sec\n')
#================================================================================================================================================
#--------------------------------  SHOW RESULTS  ------------------------------------------------------------------------------------------------
#================================================================================================================================================
        sigmas,orders,lambdas = hyperPars
        xr,yr=generateDesMatPars(terrainBool,n)[0:2]
        if(showResults == True):
            #print("Plotting results")
            #if(plotTypeInt == 0 and resampInt == 0):
            if(plotTypeInt == 0 and resampInt ==0):
                # if(resampInt>0): print("Plotting fit from last resampling cycle")
                for s,sigma in enumerate(sigmas):
                    surfacePlotter(scoreMatrices,calcAtts,doublePlot,plotOrig,terrainBool,tinkerBool\
                    ,exercise,savePlot,z_noisyL[s],z_fittedL[s],sigmas[s],s)

                    if(plotBetaCI==True):
                        beta_CI(tinkerBool,savePlot,beta_hat,var_beta,alpha,orders[0])

                print('\nRegression scores for', 'sigma = ', sigmas, ': ')
                for scoreName in scoreMatrices:
                    print(f"{scoreName+':':<18}{scoreMatrices[scoreName]}")

            else:
                print("Plotting scores vs hyperparameter(s)")
                scorePlotter(scoreMatrices,calcAtts,terrainBool,tinkerBool,exercise,savePlot)
                # print('\nRegression scores for', 'sigma = ', sigmas, ': ')
                # for scoreName in scoreMatrices:
                #     print(f"{scoreName+':':<18}{scoreMatrices[scoreName]}")
        plt.show()

if __name__ =="__main__":
    main()
