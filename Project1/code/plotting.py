import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import seaborn as sns

from functions import FrankeFunction, desMat
from regression_methods import linReg
from resampling_methods import bootstrap,crossValidation


my_path = "/home/hakon/Documents/FYS-STK4155/project1/code_and_plots"

#Font for plots
fonts= ['Comic Sans MS','cursive', 'fantasy', 'monospace', 'sans', 'sans serif', 'sans-serif', 'serif']
fontInd = 6



#===================================================================
#================= Find title&filename str =========================
def titleMaker(regMeth,resampMeth,nBoot,K,lambdas,sigma,n,orders):
    resampPar,  regressionPar    ='', ''
    resampPar_f,regressionPar_f  ='', ''

    if(resampMeth[0]=='bootstrap'):
        resampTypeStr ='Bootstrap'
        resampPar, resampPar_f = f'nBoot={nBoot},  ', f'nBoot={nBoot}_'
    if(resampMeth[0]=='crossval'):
        resampPar, resampPar_f = f'K={K},  ', f'K={K}_'
    if(regMeth=='ridge' or regMeth=='lasso'):
        if(len(lambdas)==1): #If lambdas is not a vector
            regressionPar, regressionPar_f = f'$\lambda$={lambdas[0]:.2f},  ', f'lmd={lambdas[0]:.2f}'
            # regressionPar_filename = f'lmd={lambdas[0]:.2f}'
        elif(len(orders)==1):#If orders is not a vector
            regressionPar, regressionPar_f = f'pol.deg.={orders[0]},  ', f'pol.deg.={orders[0]}_'

    heatMapTitle =" score using " + regMeth + " w. "+resampMeth[0]\
    +f"\n "+ resampPar+ r"$\sigma_{\epsilon}$=" + f"{sigma},  n={n}"

    normPlotTitle = regMeth+" regression with "+resampMeth[0]+'\n '\
    +regressionPar+ resampPar+ r"$\sigma_{\epsilon}$=" + f"{sigma},  n={n}"

    heatMapFilename ="score_"+regMeth+"_"+resampMeth[0]+"_"\
    +resampPar_f+ "sigma=" + f"{sigma}_n={n}"

    normPlotFilename =regMeth+"_"+resampMeth[0]+"_"+regressionPar_f\
    +resampPar_f+ "sigma=" + f"{sigma}_n={n}"

    return heatMapTitle,normPlotTitle, heatMapFilename, normPlotFilename
#===================================================================


def MasterFunction(savePlot,plotBool,resampMeth,regMeth,scoreNames,indVars):
    #===========================================================================
    #Setting common parameters across resampling methods
    #===========================================================================
    sigmas = indVars[0]
    orders = indVars[1]
    loglambdas = indVars[2]

    scaling=resampMeth[1]
    skOLS = resampMeth[2] #Use sklearn for OLS?
    xr,yr = resampMeth[3],resampMeth[4]
    z = resampMeth[5] #z here is without noise
    #===========================================================================

    #===========================================================================
    #Setting parameters specific to given resampling method
    #===========================================================================
    nBoot = 0
    K = 0
    if(resampMeth[0]=='bootstrap'):
        nBoot = resampMeth[6]
    elif(resampMeth[0]=='crossval'):
        f,K,shuffle = resampMeth[6:9]
        skCV = resampMeth[9] #Use sklearn for cross validation?
    #===========================================================================

    #Make the subdirectory for saving plot(s) if it doesn't exist
    subfolder = "results/"+regMeth+"/"+resampMeth[0]
    if(savePlot):
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    n = int(np.sqrt(len(z)))
    lambdas =pow(10,np.array(loglambdas))

    nSigmas = len(sigmas)
    nOrders = len(orders)
    nLambdas = len(lambdas)

    #=================================================================================
    #Dictionary containing the model assessment scores for various orders and lambdas.
    #For each iteration of the sigma for-loop, we produce a plot with the scores
    #and then overwrite the previous SD in the next sigma iteration
    #=================================================================================
    #                        matrix of scores    index      plot formatting
    SD =\
    {'bias':        [np.zeros((nLambdas,nOrders,nSigmas)),0,['firebrick', 'solid'   ,(2,1)]]\
    ,'variance':    [np.zeros((nLambdas,nOrders,nSigmas)),1,['steelblue', 'solid'   ,(2,1)]]\
    ,'cov':         [np.zeros((nLambdas,nOrders,nSigmas)),2,['purple',    'solid'   ,(2,1)]]\
    ,'MSEtest':     [np.zeros((nLambdas,nOrders,nSigmas)),3,['seagreen',  'solid'   ,(2,1)]]\
    ,'MSEtrain':    [np.zeros((nLambdas,nOrders,nSigmas)),4,['seagreen',    '--'    ,(2,1)]]\
    ,'R2test':      [np.zeros((nLambdas,nOrders,nSigmas)),5,['blueviolet', 'solid'  ,(2,1)]]\
    ,'R2train':     [np.zeros((nLambdas,nOrders,nSigmas)),6,['blueviolet',  '--'    ,(2,1)]]}
    #=================================================================================
    #================================================================================================

    #================================================================================================
    #======== Find model-assessment scores specified in 'scoreNames' ================================
    #================================================================================================
    #Loop through the various sigma's and make a plot for each
    for s in range(0,nSigmas):
        for order in range(0,nOrders):
            X = desMat(xr,yr,order)
            for l in range(0,nLambdas):
                if(resampMeth[0]=='no_resamp.'):
                    scoreValues=linReg(regMeth,scoreNames,X,z,sigmas[s],scaling,skOLS,lambdas[l])[0]
                    for i,scoreName in enumerate(scoreNames):
                        SD[scoreName][0][l,order,s]=scoreValues[i]
                elif(resampMeth[0]=='bootstrap'):
                    scoreValues = bootstrap(regMeth,scoreNames,X,z,nBoot,sigmas[s],scaling,skOLS,lambdas[l])[0]
                    for i,scoreName in enumerate(scoreNames):
                        SD[scoreName][0][l,order,s]=scoreValues[i]
                elif(resampMeth[0]=='crossval'):
                    scoreValues = crossValidation(regMeth,scoreNames,X,z,K,sigmas[s],shuffle,skOLS,skCV,lambdas[l])[0]
                    for i,scoreName in enumerate(scoreNames):
                        SD[scoreName][0][l,order,s]=scoreValues[i]

        #====================================================================================
        #====== Use the found scores to generate a plot for this iteration of sigma =========
        #====================================================================================
        heatMapTitle,normPlotTitle, heatMapFilename,normPlotFilename = \
        titleMaker(regMeth,resampMeth,nBoot,K,lambdas,sigmas[s],n,orders)
        if(nLambdas>1 and nOrders>1 and plotBool): #Run only if both vectors are greater than one
            for scoreStr in scoreNames:
                #========================================================================
                #================ Make heatmap using seaborn ============================
                #========================================================================
                fig0 = plt.figure()
                scoreMat = SD[scoreStr][0][:,:,s]
                M = pd.DataFrame(scoreMat,loglambdas,orders)
                ax = sns.heatmap(M,cmap="YlGnBu",linewidths=.0,annot = not savePlot)
                #ax = sns.heatmap(M,linewidths=.0,annot = not savePlot)
                ax.invert_yaxis()
                ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
                ax.set_xlabel('Polynomial degree, p',size='12',**{'fontname':fonts[fontInd]})
                ax.set_ylabel(r'log$_{10}(\lambda)$', size = '12',**{'fontname':fonts[fontInd]})
                plt.title(scoreStr + heatMapTitle,size=12, **{'fontname':fonts[fontInd]})
                if(savePlot):
                    plt.savefig(subfolder+"/"+scoreStr+heatMapFilename+'.png')
                plt.show()

                # Load the example flights dataset and convert to long-form
                #flights_long = sns.load_dataset("flights")
                #flights = flights_long.pivot("month", "year", "passengers")

                # Draw a heatmap with the numeric values in each cell
                f, ax = plt.subplots(figsize=(9, 6))
                ax = sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

                ax.add_patch(Rectangle((10,6),2,2, fill=False, edgecolor='blue', lw=3))

        elif(nLambdas==nOrders and plotBool):
            print('No plots implemented for loops over only sigma.\
            \nSee return matrices for results.')

        elif(plotBool): #Only run in case only one of orders or lambas is a 'vector'
            #========================================================================
            #=== Plot score(s) as a function of either order or lambda ==============
            #========================================================================

            #====================================================================
            #Draw the curves of the desired scores. Determines whether to
            #set xaxis to orders or lambdas.
            #====================================================================
            fig=plt.figure()
            for score in scoreNames:
                score_color = SD[score][2][0]
                score_linestyle = SD[score][2][1]
                score_dashes = SD[score][2][2]
                if (nLambdas==1):
                    xaxis = orders
                    xaxis_title= 'Polynomial degree, p'
                    score_vector=SD[score][0][0,:,s]
                if (nOrders==1):
                    xaxis = loglambdas
                    xaxis_title = r'Regularization parameter, $\lambda$'
                    score_vector=SD[score][0][:,0,s]
                plt.plot(xaxis,score_vector, score_color,\
                label=score,linestyle=score_linestyle)
            #====================================================================

            # if('R2test' in scoreNames and 'R2train' in scoreNames):
            #     plt.plot(xaxis,np.ones(len(xaxis)),'black',linestyle='--',dashes=(2,1))
            #     plt.plot(xaxis,np.zeros(len(xaxis)),'black',linestyle='--',dashes=(2,1))

            plt.ylabel('Error',size=11,**{'fontname':fonts[fontInd]})
            plt.xlabel(xaxis_title,size=11,**{'fontname':fonts[fontInd]})
            plt.title(normPlotTitle,size=12,**{'fontname':fonts[fontInd]})

            # Show the minor grid lines with very faint and almost transparent grey lines
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.tight_layout()
            plt.legend()
            axes = plt.gca()
            axes.set_xlim([xaxis[0], xaxis[len(xaxis)-1]])
            plt.legend()
            joinedScoreStr = ''
            for i in range(0,len(scoreNames)): joinedScoreStr+=scoreNames[i]+'_'
            if(savePlot):
                plt.savefig(subfolder+"/"+joinedScoreStr+normPlotFilename + '.png')
                # f'_scorelen={len(scoreNames)}'
            plt.show()

            #end of sigma for-loop
            #================================================================================================
    scoreResults = []
    for score in scoreNames:
        scoreResults.append(SD[score][0])
    #Return list of score matrices. Their order is the same as in "scoreNames"
    return scoreResults
