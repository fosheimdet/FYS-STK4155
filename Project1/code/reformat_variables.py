from imageio import imread
import numpy as np

from functions import FrankeFunction

#==================================================================================================================================================
#-------------------------------------------------------------------------------------------------------------------------------------------------
#                Reformatting of all the variables defined in the control panel.
#                  all t
#-------------------------------------------------------------------------------------------------------------------------------------------------
#==================================================================================================================================================
def reformatVariables(\
     tinkerBool,exercise\
    ,terrainBool,n_t,n_f\
    ,scaling,skOLS,skCV\
    ,plotBetaCI,alpha\
    ,sigma_v,sigma_s\
    ,minOrder,maxOrder,order_s\
    ,minLoglmd, maxLoglmd, lambda_s\
    ,sigmasBool,plotTypeInt,origSurfPlot,savePlot\
    ,resampInt,regInt,allScores,scoresNames\
    ,nBoot,K,shuffle):

    allScores = [['bias'],['variance'],['MSEtest'],['MSEtrain'],['R2test'],['R2train']] #Which regression scores one can plot
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------- GENERATE xr,yr,z,f ----------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------------------------------------
    n = n_t if terrainBool else n_f
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

    return resampMeth,regMeth,scoresNames,hyperPars


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
