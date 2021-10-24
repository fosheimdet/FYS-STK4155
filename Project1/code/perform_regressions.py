import numpy as np
import os

from functions import desMat
from resampling_methods import noResamp,bootstrap,crossValidation


#===============================================================================
#-------------------------------------------------------------------------------
#                   Calculates the specified scores
#-------------------------------------------------------------------------------
#===============================================================================
def performRegressions(resampMeth,regMeth,emptyScoreScalars,hyperPars):
    #===========================================================================
    #Setting common parameters across resampling methods
    #===========================================================================

    sigmas,orders,lambdas = hyperPars

    #Parameters common across resampling methods:
    xr,yr,z,scaling,skOLS=resampMeth[1:6]
    n = int(np.sqrt(len(z)))               #Number of ticks on x/y axes
    #===========================================================================
    #Setting parameters specific to given resampling method
    #===========================================================================
    nBoot = 0
    K = 0
    if(resampMeth[0]=='bootstrap'):
        nBoot = resampMeth[6]
    elif(resampMeth[0]=='crossval'):
        skCV,shuffle,K = resampMeth[6:9]
    #===========================================================================
    nSigmas = len(sigmas)
    nOrders = len(orders)
    nLambdas = len(lambdas)
    #===========================================================================
    # Dict of containing the requested model assessment scores.
    # The results for each score is stored as matrix with dimensionality
    #(nLambdas,nOrders,nSigmas)
    scoreMatrices = {}
    for scoreName in emptyScoreScalars:
        scoreMatrices[scoreName] = np.zeros((nLambdas,nOrders,nSigmas))
    #e.g.
    # calcRes =\
    # {'bias':        np.zeros((nLambdas,nOrders,nSigmas))\
    # ,'MSEtest':     np.zeros((nLambdas,nOrders,nSigmas))\
    # ,'R2train':     np.zeros((nLambdas,nOrders,nSigmas))}
    #===========================================================================
    #===========================================================================
    # Dict of calculation attributes. Here we store the lambdas,orders and sigmas
    # (which are needed for plotting) in hyperPars. We also store title and filename
    # strings tailored to the requested calculation(s) in 'tfNames'.
    calcAtts = {'hyperPars': hyperPars,
                'plotTitleAtts':[regMeth,resampMeth,nBoot,K,n]}
    #===========================================================================
    #===========================================================================
    beta_hat,var_beta = [],[]  #Needed for beta_hat confidence intervals
    z_noisyL,z_fittedL  =[],[] #Needed for surface plotting
    #===========================================================================
    #================================================================================================
    #======== Find model-assessment scores specified in 'scoreNames' ================================
    #================================================================================================
    #scoreValues={"bias":0,"variance":0, "MSEtest": 0, "MSEtrain": 0, "R2test": 0, "R2train":0}
    for s,sigma in enumerate(sigmas):
        for o,order in enumerate(orders):
            X = desMat(xr,yr,order)
            for l,lmd in enumerate(lambdas):
                if(resampMeth[0]=='no_resamp.'):
                    scoreScalars,z_noisy,z_fitted,beta_hat,var_beta=noResamp(regMeth,emptyScoreScalars,X,sigma,lmd,z,scaling,skOLS)
                    if(nOrders==1 and nLambdas ==1):
                        z_noisyL.append(z_noisy), z_fittedL.append(z_fitted)
                elif(resampMeth[0]=='bootstrap'):
                    scoreScalars = bootstrap(regMeth,emptyScoreScalars,X,sigma,lmd,z,scaling,skOLS,nBoot)[0]
                elif(resampMeth[0]=='crossval'):
                    scoreScalars = crossValidation(regMeth,emptyScoreScalars,X,sigma,lmd,z,scaling,skOLS,skCV,shuffle,K)[0]
                for scoreName in scoreScalars:
                    scoreMatrices[scoreName][l,o,s] = scoreScalars[scoreName]    #Fill one element each requested score matrix

    return scoreMatrices,calcAtts,z_noisyL,z_fittedL,beta_hat,var_beta
