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



def plotResults(scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,xr,yr):
        sigmas,orders,lambdas = hyperPars
        xr,yr=generateDesMatPars(terrainBool,n)[0:2]
        if(showResults == True):
            if(plotTypeInt == 0 and resampInt == 0):
                for s,sigma in enumerate(sigmas):
                    surfacePlotter(scoreMatrices,calcAtts,doublePlot,plotOrig,terrainBool,tinkerBool\
                    ,exercise,savePlot,z_noisyL[s],z_fittedL[s],sigmas[s],s)
                    # if(doublePlot == True or plotOrig == True):
                    #     surfacePlotter(scoreMatrices,calcAtts,doublePlot,plotOrig,terrainBool,tinkerBool\
                    #     ,exercise,savePlot,z_noisyL[s],z_fittedL[s],sigmas[s],s)

                    if(plotBetaCI==True):
                        beta_CI(tinkerBool,savePlot,beta_hat,var_beta,alpha,orders[0])

                print('\nRegression scores for', 'sigma = ', sigmas, ': ')
                for scoreName in scoreMatrices:
                    print(f"{scoreName+':':<18}{scoreMatrices[scoreName]}")

            else:
                print("Plotting scores vs hyperparameter(s)")
                scorePlotter(scoreMatrices,calcAtts,terrainBool,tinkerBool,exercise,savePlot)


#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------  CONTROL PANEL ---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
#This is where all the variables are set
ONbutton   = True    #Set to False to run main.py without performing regression with the parameters specified below
tinkerBool = False    #If True, a folder named 'tinkerings' will be created if savePlot is True, in which the plots will be saved based on reg. and resamp. method.
#exercise = 1          #If tinkerBool=False, a folder named 'exercise_results' if savePlot is True. This int determines which subfolder to save to.

terrainBool = False  #Use terrain data or Franke function?
n_t = 1000            #How many points on the x and y axes if using terrain data
n_f = 40             #How many points on the x and y axes if using FrankeFunction

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
S = 4*S
sigma_v  =       [0.1*S, 1*S, 2*S, 10*S]  #Make a separate plot for each of these sigmas
#sigma_v  =       [0.1*S, 1*S, 2*S, 10*S,20*S]
sigma_s  =               [2*S]        #Default standard deviation of epsilon
#------------------------------------------
minOrder,maxOrder =       0, 20         #Will make order vector from minOrder to maxOrder
order_s =                 [10]           #Default pol.degree if we don't plot vs. degrees
#------------------------------------------
minLoglmd, maxLoglmd =    -6,-1      #Will make log(lambda) vector from minLoglmd to maxLoglmd
lambda_s  =               [-3]        #Default lambda value. Must be set
#---------------------------------------------------------------------------
#------------------------ Choose type of plot-------------------------------
#---------------------------------------------------------------------------
sigmasBool = False #If true, will produce a plot for each element in sigma_v.
#Will be set to False for terrain data during reformating

              # [ordersBool lamdasBool] = plotTypeInt
plotTypeInt =              0
              # [  False   0   False  ]  Generate surface plot(s) and print results if using no_resamp.
              # [  True    1   False  ]  Plots error vs. pol.deg.
              # [  False   2   True   ]  Plots error vs. lambda
              # [  True    3   True   ]  Produces heatmap(s) of error(s)
# showResults  = False #Plot results based on plotTypeInt?
# savePlot = True #Save plots?
#---------------------------------------------------------------------------
#Choose resampling technique, regression method and what scores to calculate
#---------------------------------------------------------------------------
resampInt = 0 #0=no_resamp., 1=bootstrap, 2=crossval
regInt    = 0 #0=OLS,        1=ridge,     2=lasso

dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
#              0        1          2          3         4        5
emptyScoreScalars=getScores(0,1)
#Sets which scores to calculate by passing in the corresponding indices from allScores.
#emptyScoreScalars=getScores(0,1,2)
nBoot =50         #Number of bootstrap samples
K = 5               #Number of folds in cross validation alg.
shuffle = True      #Shuffle the data before performing crossval folds?



#=================================================================================
#======================= Select task to generate ===============================

saveInput = input("Save plot(s)?y/n: ")
savePlot = True if saveInput == 'y' else False
savePlotStr = 'Plot saving ON' if savePlot else 'Plot saving OFF'
exercise = input("Please select exercise: ")
print("\nTo select all tasks in a given exercise, type 0\n")

exercise = int(exercise)

displayPlots = True
#=================================================================================
#=================================================================================


savePlotStr = 'Plot saving ON' if savePlot else 'Plot saving OFF'
print('\t\t\t',savePlotStr,'\n\n')



if(exercise == 1):
    print("=====================================================================================")
    print("--------------------------   exercise 1: OLS on Franke  -----------------------------")
    print("=====================================================================================")
    print("Task 1:  Constant sigma, various n")
    print("Task 2:  n=40, various sigma")
    print("Task 3:  beta_CI")
    print("Task 4:  bias var tradeoff")
    print("Task 5:  bias var MSEtest")
    print("Task 6:  MSEtest vs MSEtrain")
    print("Task 7:  R2test vs R2train")
    print("Task 8:  MSEtest, R2test heatmap crossval, K=5")
    task = input("Please select task: ")
    task = int(task)


    if(task== 1 or task==0 ):
        print("-------------------------------   task 1  ---------------------------------------")
        dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
        #              0        1          2          3         4        5
        emptyScoreScalars=getScores(0,1,2,3,4,5)
        showResults  = True #Plot results based on plotTypeInt?
        doublePlot = True   #Plot the data side by side with the fitted surface?
        plotOrig = True     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.
        n_f = 40
        n_vec = [10,20,40,80,200]
        # n_vec = [20,50,100,200,500]
        sigma_s = [0.2]
        for n_f in n_vec:
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            if(ONbutton):
                print('Performing regression and calculating scores\n')
                t_start = time.time()
                scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
                t_end = time.time()
                print('performRegressions() time used: ', t_end-t_start,'sec\n')
                plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==2 or task == 0):
        print("-------------------------------   task 2  ---------------------------------------")
        showResults  = True #Plot results based on plotTypeInt?
        doublePlot = True   #Plot the data side by side with the fitted surface?
        plotOrig = False     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.
        sigmasBool = True
        dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
        #              0        1          2          3         4        5
        emptyScoreScalars=getScores(0,1,2,3,4,5)
        sigma_v  =       [0.1*S, 1*S, 2*S, 10*S, 50*S]
    if(task==3 or task == 0):
        print("-------------------------------   task 3, beta CI  ---------------------------------------")
        doublePlot = False
        plotOrig = False
        plotBetaCI  = True
        emptyScoreScalars=getScores(0,1,2,3,4,5)
        #minOrder,maxOrder =       0, 5
        for order in range(1,5+1):
            order_s     = [order]
            sigmasBool = False
            plotTypeInt = 0
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
    if(task==4 or task == 0):
        print("-------------------------------   task 4  ---------------------------------------")
        minOrder,maxOrder =       0, 10
        sigmasBool = True
        plotTypeInt = 1
    if(task==5 or task == 0):
        # doublePlot = False
        # plotOrig = False
        print("-------------------------------   task 5  ---------------------------------------")
        plotBetaCI  = False
        dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
        #              0        1          2          3         4        5
        emptyScoreScalars=getScores(0,1,2)
        minOrder,maxOrder =       0, 20
        sigmasBool = True
        plotTypeInt = 1
    if(task==6 or task==0):
        print("-------------------------------   task 6  ---------------------------------------")
        plotBetaCI  = False
        dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
        #              0        1          2          3         4        5
        emptyScoreScalars=getScores(2,3)
        minOrder,maxOrder =       0, 20
        sigmasBool = True
        plotTypeInt = 1
    if(task==7 or task==0):
        print("-------------------------------   task 7  ---------------------------------------")
        plotBetaCI  = False
        dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
        #              0        1          2          3         4        5
        emptyScoreScalars=getScores(4,5)
        minOrder,maxOrder =       0, 20
        sigmasBool = False
        plotTypeInt = 1
    if(task==8 or task==0):
        print("-------------------------------- task 8: heatmap OLS  -----------------------------")
        minOrder,maxOrder =       1, 20
        emptyScoreScalars=getScores(2,4)
        plotTypeInt = 3
        resampInt = 2
        regInt = 0
        K=5
        minLoglmd, maxLoglmd =    -6,-1

#===============================================================================
    excludeTasks = [1,3]
    if task not in excludeTasks:
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)

        if(displayPlots):plt.show()
#===============================================================================
#-----------------------------------  CONTROL PANEL ---------------------------------------------------------------------------------------------


terrainBool = False  #Use terrain data or Franke function?
n_t = 1000            #How many points on the x and y axes if using terrain data
n_f = 40             #How many points on the x and y axes if using FrankeFunction

showResults  = True #Plot results based on plotTypeInt?
doublePlot = True   #Plot the data side by side with the fitted surface?
plotOrig = True     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.

savePlot = True #Save plots?

scaling = True     #Scale the design matrix, X?
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
minOrder,maxOrder =       0, 20         #Will make order vector from minOrder to maxOrder
order_s =                 [10]           #Default pol.degree if we don't plot vs. degrees
#------------------------------------------
minLoglmd, maxLoglmd =    -6,-1      #Will make log(lambda) vector from minLoglmd to maxLoglmd
lambda_s  =               [-3]        #Default lambda value. Must be set
sigmasBool = False #If true, will produce a plot for each element in sigma_v.
plotTypeInt =              1

resampInt = 1 #0=no_resamp., 1=bootstrap, 2=crossval
regInt    = 0 #0=OLS,        1=ridge,     2=lasso

dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
#              0        1          2          3         4        5
#Sets which scores to calculate by passing in the corresponding indices from allScores.
emptyScoreScalars=getScores(3,2)
nBoot =20         #Number of bootstrap samples
K = 5               #Number of folds in cross validation alg.
shuffle = True      #Shuffle the data before performing crossval folds?

if(exercise==2):
    print("=====================================================================================")
    print("--------------------   exercise 2: Bootstrap, bias var  -----------------------------")
    print("=====================================================================================")
    print("Task 1:  MSEtest VS MSEtrain for various n")
    print("Task 2:  Bootstrap calc of bias,var,MSEtest")
    task = input("Please select task: ")
    task = int(task)
    if(task==1 or task == 0):
        print("-------------------------------   task 1  ---------------------------------------")
        emptyScoreScalars=getScores(3,2)
        n_v = [10,20,40,80]
        for n_f in n_v:
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
    if(task == 2 or task==0):
        print("-------------------------------   task 2  ---------------------------------------")
        emptyScoreScalars=getScores(0,1,2)
        nBoot = 20
#===============================================================================
    excludeTasks = [1,0]
    regInt = 0
    if task not in excludeTasks:
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)

        if(displayPlots):plt.show()
#===============================================================================


if(exercise==3 or exercise==42):
    print("=====================================================================================")
    print("----------------------------- exercise 3: Cross Validation  -------------------------")
    print("=====================================================================================")
    print("Task 1:  K=5, MSEtest VS MSEtrain for various n")
    print("Task 2:  K=10, MSEtest VS MSEtrain for various n")
    task = input("Please select task: ")
    task = int(task)

    if(task==1 or task == 0):
        print("-------------------------------   task 1  ---------------------------------------")
        emptyScoreScalars=getScores(2,3)
        resampInt = 2
        K =5
        n_v = [10,20,40,80]

        for n_f in n_v:
            #===============================================================================
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
            #===============================================================================
    if(task==2 or task == 0):
        print("-------------------------------   task 2  ---------------------------------------")
        emptyScoreScalars=getScores(2,3)
        resampInt = 2
        K =10
        n_v = [10,20,40,80]

        for n_f in n_v:
            #===============================================================================
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
            #===============================================================================

#-----------------------------------  CONTROL PANEL ---------------------------------------------------------------------------------------------

n_f = 40             #How many points on the x and y axes if using FrankeFunction
showResults  = True #Plot results based on plotTypeInt?
doublePlot = True   #Plot the data side by side with the fitted surface?
plotOrig = True     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.
savePlot = True #Save plots?
scaling = True     #Scale the design matrix, X?
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
minOrder,maxOrder =       0, 20         #Will make order vector from minOrder to maxOrder
order_s =                 [10]           #Default pol.degree if we don't plot vs. degrees
#------------------------------------------
minLoglmd, maxLoglmd =    -6,-1      #Will make log(lambda) vector from minLoglmd to maxLoglmd
lambda_s  =               [-3]        #Default lambda value. Must be set
#---------------------------------------------------------------------------
#------------------------ Choose type of plot-------------------------------
#---------------------------------------------------------------------------
sigmasBool = False #If true, will produce a plot for each element in sigma_v.
#Will be set to False for terrain data during reformating

              # [ordersBool lamdasBool] = plotTypeInt
plotTypeInt =              1
resampInt = 0 #0=no_resamp., 1=bootstrap, 2=crossval
regInt    = 1 #0=OLS,        1=ridge,     2=lasso

dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
#              0        1          2          3         4        5
#Sets which scores to calculate by passing in the corresponding indices from allScores.
emptyScoreScalars=getScores(3,2)
nBoot =20         #Number of bootstrap samples
K = 5               #Number of folds in cross validation alg.
shuffle = True      #Shuffle the data before performing crossval folds?

if(exercise==4 or exercise==42):
    print("=====================================================================================")
    print("--------------------------   exercise 4: Ridge  -------------------------------------")
    print("=====================================================================================")
    print("Task 1:  MSEtest,train vs complexity, nBoot = 30 for various lmd")
    print("Task 2:  bias, variance vs complexity, nBoot = 30 for various lmd")
    print("Task 3:  MSEtest,train vs lambda, for no resamp., crossval and boot")
    print("Task 4:  MSEtest and R2test heatmap for ridge using bootstrap")
    task = input("Please select task: ")
    task = int(task)

    if(task==1 or task==0):
        print("-------------------------------   task 1  ---------------------------------------")
        emptyScoreScalars=getScores(3,2)
        lambdaVec = [[5],[2],[0],[-2],[-5],[-10]]
        resampInt = 1
        nBoot = 30
        for lambda_s in lambdaVec:
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==2 or task==0):
        print("-------------------------------   task 2  ---------------------------------------")

        lambdaVec = [[5],[2],[0],[-2],[-5],[-10]]
        resampInt = 1
        nBoot = 30
        emptyScoreScalars=getScores(0,1)
        for lambda_s in lambdaVec:
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
    if(task==3 or task==0):
        print("-------------------------------   task 3  ---------------------------------------")
        emptyScoreScalars=getScores(4,5)
        print("-------------------------------   no resampling  ---------------------------------------")
        plotTypeInt = 2
        resampInt = 0
        regInt = 2
        minLoglmd, maxLoglmd =    -15,15

        #===============================================================================
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
        #===============================================================================
        print("-------------------------------   crossval  ---------------------------------------")
        plotTypeInt = 2
        resampInt = 2
        K=5
        minLoglmd, maxLoglmd =    -15,15

        #===============================================================================
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
        #===============================================================================

        print("-------------------------------   bootstrap  ---------------------------------------")
        plotTypeInt = 2
        resampInt = 1
        nBoot = 50
        minLoglmd, maxLoglmd =    -15,15

        #===============================================================================
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
        #===============================================================================

    if(task ==4 or task == 0):
        print("-------------------------------- task4: heatmap bootstrap  ----------------------------------------------")
        minOrder,maxOrder =       1, 20
        emptyScoreScalars=getScores(2,4)
        plotTypeInt = 3
        resampInt = 1
        regInt = 1
        nBoot = 20
        minLoglmd, maxLoglmd =    -6,-1

        #===============================================================================
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
        #===============================================================================

if(exercise==5 or exercise==42):
    print("=====================================================================================")
    print("------------------------  exercise 5: Lasso regression  -----------------------------")
    print("=====================================================================================")
    print("Task 1:  bias variance, for various lmd with nBoot=30")
    print("Task 2:  MSEtest,train vs complexity, crossval K =5 for various lmd")
    print("Task 3:  MSEtest,train vs lambda, for no resamp., and crossval")
    print("Task 4:  MSEtest,train vs lambda, with bootstrap")
    print("Task 5:  Lasso heatmap for MSEtest and R2test, using bootstrap")
    task = input("Please select task: ")
    task = int(task)
    regInt = 2

    if(task==1 or task==0):
        print("-------------------------------   task 1  ---------------------------------------")
        lambdaVec = [[10],[5],[2],[0],[-2],[-5],[-10]]
        resampInt = 1
        nBoot = 30
        emptyScoreScalars=getScores(0,1)

        for lambda_s in lambdaVec:
            #===============================================================================
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
            #===============================================================================
        if(displayPlots):plt.show()

    if(task==2 or task==0):
        print("-------------------------------   task 2  ---------------------------------------")
        lambdaVec = [[10],[5],[2],[0],[-2],[-5],[-10]]
        resampInt = 2
        K = 5
        emptyScoreScalars=getScores(2,3)
        for lambda_s in lambdaVec:
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==3 or task==0):
        print("-------------------------------   task 3  ---------------------------------------")
        print("-------------------------------   no resamp  ---------------------------------------")
        plotTypeInt = 2
        resampInt = 0
        minLoglmd, maxLoglmd =    -15,15

        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        #if(displayPlots):plt.show()

        print("-------------------------------   crossval  ---------------------------------------")
        resampInt = 2
        K=5

        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==4 or task==0):
        print("-------------------------------  bootstrap  ---------------------------------------")
        plotTypeInt = 2
        emptyScoreScalars=getScores(2,3)
        #emptyScoreScalars=getScores(4,5)
        resampInt = 1
        nBoot = 50
        minLoglmd, maxLoglmd =    -15,15

    if(task ==5 or task == 0):
        print("-------------------------------- task 5: heatmap bootstrap  ----------------------------------------------")
        minOrder,maxOrder =       1, 20
        emptyScoreScalars=getScores(2,4)
        plotTypeInt = 3
        resampInt = 1
        regInt = 2
        nBoot = 20
        minLoglmd, maxLoglmd =    -6,-1



    excludeTasks = [1,2,3]
    if task not in excludeTasks:
        #=======================================================================
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
        #=======================================================================

#-----------------------------------  CONTROL PANEL ---------------------------------------------------------------------------------------------
terrainBool = True  #Use terrain data or Franke function?
n_t = 50            #How many points on the x and y axes if using terrain data
n_f = 40             #How many points on the x and y axes if using FrankeFunction

showResults  = True #Plot results based on plotTypeInt?
doublePlot = True   #Plot the data side by side with the fitted surface?
plotOrig = True     #Plot the data or the fit if doublePlot is False? I.e. terrain or noisy Franke.

savePlot = True #Save plots?

scaling = True     #Scale the design matrix, X?
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
minOrder,maxOrder =       0, 20         #Will make order vector from minOrder to maxOrder
order_s =                 [10]           #Default pol.degree if we don't plot vs. degrees
#------------------------------------------
minLoglmd, maxLoglmd =    -6,-1      #Will make log(lambda) vector from minLoglmd to maxLoglmd
lambda_s  =               [-3]        #Default lambda value. Must be set

sigmasBool = False #If true, will produce a plot for each element in sigma_v.

plotTypeInt =              0
resampInt = 0 #0=no_resamp., 1=bootstrap, 2=crossval
regInt    = 0 #0=OLS,        1=ridge,     2=lasso

dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
#              0        1          2          3         4        5
#Sets which scores to calculate by passing in the corresponding indices from allScores.
emptyScoreScalars=getScores(3,2)
nBoot =20         #Number of bootstrap samples
K = 5               #Number of folds in cross validation alg.
shuffle = True      #Shuffle the data before performing crossval folds?
#================================================================================================================================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================


if(exercise==6 or exercise==42):
    print("Task 1:  Surface plot of p=10 fits for all regression models")
    print("Task 2:  MSE vs lambda and R2 vs lambda for all the models")
    print("Task 3:  Heatmaps in polynomial ranges 4 to 25 for all regression models")
    print("Task 4:  Heatmaps in polynomial ranges 10 to 35 for specified regression model")
    # print("Task 5:  Heatmap with n=400. Do not try.")
    print("Task 5:  Surface plot of p=50 Ridge OLS comparison")
    task = input("Please select task: ")
    task = int(task)
    terrainBool = True
    n_t = 50
    print("=====================================================================================")
    print("-------------------------------   exercise 6  ---------------------------------------")
    print("=====================================================================================")

    if(task==1 or task==0):
        print("-------------------------------   task 1  ---------------------------------------")
        resampInt = 0

        #Plot only terrain data
        doublePlot = False
        plotOrig = True
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)

        plotOrig = False
        #Plot fits of all three reg methods
        emptyScoreScalars=getScores(0,1,2,3,4,5)
        for regInt in range(0,3):

            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==2 or task==0):
        print("-------------------------------   task 2  ---------------------------------------")
        minOrder,maxOrder =       0, 35
        plotTypeInt = 1
        resampInt = 2
        K = 5
        regInt = 0
        dummyList = ['bias','variance','MSEtest','MSEtrain','R2test','R2train']
        #              0        1          2          3         4        5
        emptyScoreScalars=getScores(2,3)


        for regInt in range(0,3):
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        #if(displayPlots):plt.show()

        emptyScoreScalars=getScores(4,5)

        for regInt in range(0,3):
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==3 or task==0):
        print("-------------------------------   task 3  ---------------------------------------")
        minOrder,maxOrder =       4,25
        minLoglmd, maxLoglmd =    -6,0
        plotTypeInt = 3
        resampInt = 2
        emptyScoreScalars=getScores(2,4)
        K = 5
        for regInt in range(0,3):
            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
            if(displayPlots):plt.show()


    if(task==4 or task==0):
        print("-------------------------------   task 4  ---------------------------------------")
        minOrder,maxOrder =       10,35
        minLoglmd, maxLoglmd =    -10,-2.5
        n_t = 40
        plotTypeInt = 3
        resampInt = 2
        shuffle = True
        emptyScoreScalars=getScores(2,4)
        K = 5
        regInt = input("Choose regression method (OLS=0,ridge=1,lasso=2)")
        regInt = int(regInt)
        dummyRegList = ["OLS","ridge","lasso"]
        print("Performing ",dummyRegList[regInt], " regression")
        print('Reformatting variables\n')
        n = n_t if terrainBool else n_f
        if(terrainBool):sigmasBool = False
        resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
        print('Performing regression and calculating scores\n')
        t_start = time.time()
        scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
        t_end = time.time()
        print('performRegressions() time used: ', t_end-t_start,'sec\n')
        plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()

    if(task==5 or task==0):
        print("-------------------------------   task 5  ---------------------------------------")
        n_t = 40
        order_s = [50]
        lambda_s = [-10]
        doublePlot = False
        plotOrig = False
        #Plot fits of all three reg methods
        emptyScoreScalars=getScores(0,1,2,3,4,5)
        for regInt in range(0,2):

            print('Reformatting variables\n')
            n = n_t if terrainBool else n_f
            if(terrainBool):sigmasBool = False
            resampMeth,regMeth,emptyScoreScalars,hyperPars=reformatVariables(tinkerBool,exercise,terrainBool,n_t,n_f,scaling,skOLS,skCV,plotBetaCI,alpha,sigma_v,sigma_s,minOrder,maxOrder,order_s,minLoglmd, maxLoglmd, lambda_s,sigmasBool,plotTypeInt,savePlot,resampInt,regInt,emptyScoreScalars,nBoot,K,shuffle)
            print('Performing regression and calculating scores\n')
            t_start = time.time()
            scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta=performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars)
            t_end = time.time()
            print('performRegressions() time used: ', t_end-t_start,'sec\n')
            plotResults (scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta,hyperPars,resampMeth,showResults)
        if(displayPlots):plt.show()
