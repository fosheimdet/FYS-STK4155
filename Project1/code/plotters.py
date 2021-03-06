import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import os


from functions import addNoise,desMat


#Font for plots
fonts= ['Comic Sans MS','cursive', 'fantasy', 'monospace', 'sans', 'sans serif', 'sans-serif', 'serif']
fontInd = 6 #Index of font we want for plot axes/titles

#Sets line attributes for the different scores
scoreLineSpec =\
{'bias':        ['firebrick', 'solid'   ,(2,1)]\
,'variance':    ['steelblue', 'solid'   ,(2,1)]\
,'cov':         ['purple',    'solid'   ,(2,1)]\
,'MSEtest':     ['seagreen',  'solid'   ,(2,1)]\
,'MSEtrain':    ['seagreen',    '--'    ,(2,1)]\
,'R2test':      ['blueviolet', 'solid'  ,(2,1)]\
,'R2train':     ['blueviolet',  '--'    ,(2,1)]}

#Title and filename maker for plotting scores
def titleMaker(terrainBool,sigma,regMeth,resampMeth,nBoot,K,n,lambdas,orders):
    dataTypeStr = ''
    if(terrainBool):
        dataTypeStr, dataTypeStr_f = 'terrain data', 'terrain'
    else:
        dataTypeStr, dataTypeStr_f = 'Franke function', 'Franke'

    resampPar,  regressionPar    ='', ''
    resampPar_f,regressionPar_f  ='', ''

    if(resampMeth[0]=='bootstrap'):
        resampTypeStr ='Bootstrap'
        resampPar, resampPar_f = f'nBoot={nBoot},  ', f'nBoot={nBoot}_'
    if(resampMeth[0]=='crossval'):
        resampPar, resampPar_f = f'K={K},  ', f'K={K}_'
    if(regMeth=='ridge' or regMeth=='lasso'):
        if(len(lambdas)==1): #If lambdas is not a vector
            #regressionPar, regressionPar_f = f'$\lambda$={lambdas[0]:.2f},  ', f'lmd={lambdas[0]:.2f}'
            regressionPar  = r'$\log{\lambda}$=' + f'{lambdas[0]:.2f},  '
            regressionPar_f =  f'logLmd={lambdas[0]:.2f}'
            # regressionPar_filename = f'lmd={lambdas[0]:.2f}'
        elif(len(orders)==1):#If orders is not a vector
            regressionPar, regressionPar_f = f'pol.deg.={orders[0]},  ', f'pol.deg.={orders[0]}_'

    heatMapTitle =" score using " + regMeth + " w. "+resampMeth[0]+" on " +dataTypeStr+"\n "\
    + resampPar+ r"$\sigma_{\epsilon}$=" + f"{sigma:.2f},  n={n}"

    normPlotTitle = regMeth+" regression" " on "+dataTypeStr+ " with " +resampMeth[0]+'\n '\
    +regressionPar+ resampPar+ f"n={n},  "+r"$\sigma_{\epsilon}$=" + f"{sigma:.3f}"

    heatMapFilename =dataTypeStr+"_score_"+regMeth+"_"+resampMeth[0]+"_"\
    +resampPar_f+ f"n={n}_" + f"sigma={sigma:.2f}"

    normPlotFilename ="_"+regMeth+"_"+resampMeth[0]+"_"+regressionPar_f\
    +resampPar_f+ f"n={n}_" + f"sigma={sigma:.2f}"

    return heatMapTitle,normPlotTitle, heatMapFilename, normPlotFilename
#==================================================================================================


#====================================================================================================================
#--------------------------------------------------------------------------------------------------------------------
#                            Plots the score results
#--------------------------------------------------------------------------------------------------------------------
#====================================================================================================================
def scorePlotter(calcRes,calcAtts,terrainBool,tinkerBool,exercise,savePlot):
    #Dict containing score results in matrix form
    sigmas, orders, lambdas = calcAtts['hyperPars']


    nSigmas = len(sigmas)
    nOrders = len(orders)
    nLambdas = len(lambdas)

    scoreNames = []
    for s,score in enumerate(calcRes):
        scoreNames.append(score)


    #Get attributes needed for making plot titles/filenames
    regMeth,resampMeth,nBoot,K,n = calcAtts['plotTitleAtts']
    #---------------------------------------------------------------------------
    #   Make the subdirectory for saving plot(s) if it doesn't exist
    #   tinkerStr   = 'tinker'  if (tinkerBool) else  'exercises'
    #---------------------------------------------------------------------------
    dataTypeStr = 'terrain' if (terrainBool) else 'Franke'
    if(tinkerBool):
        subfolder = "tinkerings/"+dataTypeStr+"/"+regMeth+"/"+resampMeth[0]
        # subfolder_surfplot = "tinkerings/surface_plots/"
    elif(tinkerBool==False):
        subfolder = "exercise_results/"+f'exercise{exercise}/'+f'n={n}'
        # subfolder_surfplot = "exercise_results/surface_plots/"
    if(savePlot):
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # annotStr = input("Annotate heatmap plot(s)?y/n: ")
    # annotBool = True if annotStr=='y' else False
    # annotBool = True
    annotDict ={"annot":True, "":False}
    #====================================================================================
    #====== Use the found scores to generate a plot for this iteration of sigma =========
    #====================================================================================

    for s,sigma in enumerate(sigmas):
        heatMapTitle,normPlotTitle, heatMapFilename,normPlotFilename = \
        titleMaker(terrainBool,sigma,regMeth,resampMeth,nBoot,K,n,lambdas,orders)
        #titleMaker(regMeth,resampMeth,nBoot,K,lambdas,sigmas[s],n,orders)
        if(nLambdas>1 and nOrders>1): #Run only if both vectors are greater than one
            for scoreStr in scoreNames:
                for annot in annotDict:
                    #========================================================================
                    #================ Make heatmap using seaborn ============================
                    #========================================================================
                    fig0 = plt.figure()
                    scoreMat = calcRes[scoreStr][:,:,s]
                    lmds_formatted=lambdas
                    annotSize = {"size": 10}
                    for i in range(0,nLambdas):
                        lmds_formatted[i] = float("{:.2f}".format(lambdas[i]))
                    M = pd.DataFrame(scoreMat,lmds_formatted,orders)
                    if(scoreStr == 'R2test' or scoreStr == 'R2train'):
                        ax = sns.heatmap(M,cmap ="YlGnBu",linewidths=.0,annot = annotDict[annot],annot_kws=annotSize)
                        #ax = sns.heatmap(M,cmap="BuPu_r",linewidths=.0,annot = not savePlot)
                    if(scoreStr == 'MSEtest' or scoreStr == 'MSEtrain'):
                        ax = sns.heatmap(M,linewidths=.0,annot = annotDict[annot],annot_kws=annotSize)
                        #ax = sns.heatmap(M,cmap ="Greens",linewidths=.0,annot = not savePlot)
                    if(scoreStr == 'variance'):
                        ax = sns.heatmap(M,cmap ="Blues",linewidths=.0,annot = annotDict[annot],annot_kws=annotSize)
                    if(scoreStr == 'bias'):
                        ax = sns.heatmap(M,cmap ="Reds",linewidths=.0,annot =annotDict[annot],annot_kws=annotSize)
                        # ax = sns.heatmap(M,cmap ="YlGnBu",linewidths=.0,annot = not savePlot)
                    ax.invert_yaxis()
                    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
                    ax.set_xlabel('Polynomial degree, p',size='12',**{'fontname':fonts[fontInd]})
                    ax.set_ylabel(r'log$_{10}(\lambda)$', size = '12',**{'fontname':fonts[fontInd]})
                    plt.title(scoreStr + heatMapTitle,size=12, **{'fontname':fonts[fontInd]})
                    # figManager = plt.get_current_fig_manager()
                    # figManager.window.showMaximized()
                    if(savePlot):
                        # mng = plt.get_current_fig_manager()
                        # mng.resize(*mng.window.maxsize())

                        plt.savefig(subfolder+"/"+scoreStr+heatMapFilename+"_"+annot+'.png', bbox_inches='tight')
                        print("Saving to:\n"+ subfolder+"/"+scoreStr+heatMapFilename+'.png')
                    #plt.show()

        elif(nLambdas>1 or nOrders>1 ): #Only run in case only one of orders or lambas is a 'vector'
            #========================================================================
            #    Plot score(s) as a function of either order or lambda
            #========================================================================

            #-----------------------------------------------------------------------
            #   Draw the curves of the desired scores. Determines whether to
            #   set xaxis to orders or lambdas.
            #-----------------------------------------------------------------------
            fig=plt.figure()
            for score in scoreNames:
                score_color = scoreLineSpec[score][0]
                score_linestyle = scoreLineSpec[score][1]
                score_dashes = scoreLineSpec[score][2]
                if (nLambdas==1):
                    xaxis = orders
                    xaxis_title= 'Polynomial degree, p'
                    # score_vector=calcRes[score][0,:,s]
                    score_vector=calcRes[score][0,:,s]
                if (nOrders==1):
                    xaxis = lambdas
                    # for i in range(0,nLambdas):
                    #     lmds_formatted[i] = float("{:.2f}".format(lambdas[i]))
                    xaxis_title = r'Regularization parameter, $\log{\lambda}$'
                    #score_vector=calcRes[score][:,0,s]
                    score_vector=calcRes[score][:,0,s]
                scoreLabel = score
                if(scoreLabel == 'bias'): scoreLabel = r'bias$^2$'

                plt.plot(xaxis,score_vector, score_color,\
                label=scoreLabel,linestyle=score_linestyle)

            #-----------------------------------------------------------------------
            #-----------------------------------------------------------------------

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
            joinedScoreStr = dataTypeStr+'_'
            for i in range(0,len(scoreNames)): joinedScoreStr+=scoreNames[i]+'_'
            if(savePlot):
                print("Saving score plot as: \n",subfolder+"/"+joinedScoreStr+normPlotFilename + '.png' )
                plt.savefig(subfolder+"/"+joinedScoreStr+normPlotFilename + '.png')
                # f'_scorelen={len(scoreNames)}'
            #plt.show()
            #========================================================================
            #========================================================================


#====================================================================================================================
#--------------------------------------------------------------------------------------------------------------------
#       Makes surface plots of both the original data and the fitted function
#--------------------------------------------------------------------------------------------------------------------
#====================================================================================================================

#==================================================================================================

#Title and filename maker for surfaceplots
def surfPlotStringMaker(doublePlot,plotOrig,terrainBool,tinkerBool,exercise,regMeth,n,order,lmd,sigma):
        #Plot titles for single plots
        SPtitle_fitted = ''
        SPtitle_noisy = ''
        #filename for singleplots
        SPfilename_fitted = ''
        SPfilename_noisy = ''
        #Plot titles for double plot
        DPfilename_fitted = ''
        DPfilename_noisy = ''
        #filename for double plots
        DPfilename = ''


        orderStr = f'pol.deg={order}'
        orderStr_f = f'p={order}'
        sigmaStr ='' if(terrainBool) else r'$\sigma_{\epsilon}=$'+f'{sigma:.3f}'
        sigmaStr_f ='' if(terrainBool) else 'std=' + f'{sigma:.3f}'
        lmdStr = '' if (regMeth=='OLS') else r'$\log{\lambda}=$'+f'{lmd:0.3f}'
        lmdStr_f='' if (regMeth=='OLS') else 'lmd='+f'{lmd:0.3f}'

        tinkerStr = 'tinkering' if tinkerBool else 'exercises'
        terrainStr = 'terrain' if terrainBool else 'Franke'
        if(terrainBool==True):
            terrainStr = 'terrain'
        else:
            terrainStr = 'Franke'
        surfaceType = 'data' if plotOrig else 'fit'
        subfolder = tinkerStr+'/surface_plots/'+terrainStr+f'/{regMeth}/'
        if(tinkerBool == False):
            subfolder = "exercise_results/exercise"+f"{exercise}/"
        #-----------------------------------------------------------------------
        #                           Filenames
        #-----------------------------------------------------------------------
        #Double plot:-----------------------------------------------------------
        DPfilename = subfolder+terrainStr+f'_{regMeth}_'\
        +sigmaStr_f+'_'+orderStr_f+'_'+'lmdStr_f_'+f'n={n}'

        #Single plots:----------------------------------------------------------
        SPfilename_fitted  = subfolder+terrainStr+f'_{regMeth}_'\
        +sigmaStr_f+'_'+orderStr_f+'_'+lmdStr_f+'_SP'

        SPfilename_noisy = subfolder+terrainStr+'_'+surfaceType+'_'+f'n={n}'+'_'+sigmaStr_f
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        #                           Titles
        #-----------------------------------------------------------------------
        #Double plot:-----------------------------------------------------------
        DPtitle_fitted = f'{regMeth} fit, '+orderStr+ '  '+ lmdStr

        DPtitle_noisy = terrainStr+' data'+'\n'+f'n={n}' + ',  '+sigmaStr
        #Single plots:----------------------------------------------------------
        SPtitle_fitted = terrainStr+' data, '+DPtitle_fitted+'  '+sigmaStr
        SPtitle_noisy= terrainStr + ' data\n'+f'n={n}' + '  ' +sigmaStr
        #-----------------------------------------------------------------------
        DPfilename+='.png'
        SPfilename = SPfilename_noisy if(plotOrig) else SPfilename_fitted
        SPfilename +='.png'
        SPtitle = SPtitle_noisy if(plotOrig) else SPtitle_fitted
        return subfolder,SPfilename,SPtitle,DPfilename,DPtitle_noisy,DPtitle_fitted


def surfacePlotter(scoreMatrices,calcAtts,doublePlot,plotOrig,terrainBool,tinkerBool,exercise,savePlot,z_noisy,z_fitted,sigma,iterationInd):
    # calcAtts = {'hyperPars': hyperPars,
    #         'plotTitleAtts':[regMeth,resampMeth,nBoot,K,n]}

    #---------------------------------------------------------------------------
    #Unpacking variables needed for plotting:
    #---------------------------------------------------------------------------
    regMeth = calcAtts['plotTitleAtts'][0] #Gets name of regression method
    #regMeth = regMethList[0]
    resampMeth = calcAtts['plotTitleAtts'][1] #Gets the list of variables corresponding to the resampling method
    xr,yr,z,scaling,skOLS=resampMeth[1:6]
    #xr,yr,z,scaling,skOLS=resampMeth[1:6]
    n = int(np.sqrt(len(xr)))
    xx,yy = xr.reshape(n,n), yr.reshape(n,n)

    hyperPars = calcAtts['hyperPars']
    sigmas,orders,lambdas = hyperPars
    order = orders[0]
    lmd = lambdas[0]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    cmapStr = 'ocean'
    cmapStr = 'viridis'
    azimuth = 20
    theta = 30
    Lwidth = 0.3
    alpha_value = 0.9
    if(terrainBool):
        cmapStr = 'terrain'
        Lwidth = 0.1
        azimuth = 45+5+1
        theta = 90+45+45-10-5-10-17

    Z_noisy, Z_fitted = z_noisy.reshape(n,n), z_fitted.reshape(n,n)

    subfolder,SPfilename,SPtitle,DPfilename,DPtitle_noisy,DPtitle_fitted=\
    surfPlotStringMaker(doublePlot,plotOrig,terrainBool,tinkerBool,exercise,regMeth,n,order,lmd,sigma)
    # if(tinkerBool == False):
    #     subfolder = "exercise_results/exercise"+f"{exercise}/"
    if(savePlot):
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    #plt.show()


    if(doublePlot==False):
        #----------------------------------------------------------------------
        #                          single plot
        #----------------------------------------------------------------------

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        #ax.scatter(xs = heights, ys = weights, zs = ages)
        #m_Z = Z_noisy if plotOrig else Z_fitted
        print("plotOrig: ", plotOrig)
        if(plotOrig):
            surf = ax.plot_surface(xx,yy,Z_noisy, cmap = cmapStr,linewidth=Lwidth, alpha = alpha_value, edgecolor = 'k')
        else:
            surf = ax.plot_surface(xx,yy,Z_fitted, cmap = cmapStr,linewidth=Lwidth, alpha = alpha_value, edgecolor = 'k')
        #surf = ax.plot_surface(xx,yy,m_Z, cmap = cmapStr,linewidth=Lwidth, alpha = alpha_value, edgecolor = 'k')


        counter = 0
        scoreResStr = ''
        for scoreName in scoreMatrices:
            if(counter%2==0 and counter>0):scoreResStr+='\n'
            scoreResStr += scoreName+": "+\
            f'{scoreMatrices[scoreName][0][0][iterationInd]:.3f}'+"  "
            counter+=1
        if(plotOrig):
            ax.set_title(SPtitle)
        else:
            ax.set_title(SPtitle+"\n"+scoreResStr)
        # title = SPtitle_noisy if plotOrig else SPtitle
        # ax.set_title(title+"\n"+scoreResStr)
        #ax.set_title(SPtitle)
        #ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.view_init(azimuth,theta)
        #fig.colorbar(surf, shrink=0.5, aspect = 5)
        if(savePlot):
            print("Saving as: \n", SPfilename)
            plt.savefig(SPfilename)
        #plt.show()
        #----------------------------------------------------------------------
    else:
        #----------------------------------------------------------------------
        #                          double plot
        #----------------------------------------------------------------------
        fig= plt.figure(figsize=(8,4))
        #First plot
        ax = fig.add_subplot(121, projection='3d')
        surf = ax.plot_surface(xx,yy,Z_noisy, cmap=cmapStr, linewidth=Lwidth, alpha =alpha_value, edgecolor = 'k')
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.view_init(azimuth, theta)
        ax.set_title(DPtitle_noisy)
        # Add a color bar which maps values to colors
        #fig.colorbar(surf, shrink=0.5, aspect = 5)
        #Second plot
        ax = fig.add_subplot(122, projection='3d')
        #ax.scatter(xs = heights, ys = weights, zs = ages)
        surf = ax.plot_surface(xx,yy,Z_fitted, cmap = cmapStr,linewidth=Lwidth, alpha = alpha_value, edgecolor = 'k')

        counter = 0
        scoreResStr = ''
        for scoreName in scoreMatrices:
            if(counter%2==0 and counter>0):scoreResStr+='\n'
            scoreResStr += scoreName+": "+\
            f'{scoreMatrices[scoreName][0][0][iterationInd]:.3f}'+"  "
            counter+=1
        ax.set_title(DPtitle_fitted+"\n"+scoreResStr)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.view_init(azimuth,theta)
        if(savePlot):
            print("saving surfaceplot in:\n", DPfilename)
            plt.savefig(DPfilename)
        #plt.show()
        #----------------------------------------------------------------------
#====================================================================================================================
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#====================================================================================================================


#===============================================================================
#====Calculates the confidence intervals of beta_hat and plots them=============
#===============================================================================
def beta_CI(tinkerBool,savePlot,beta_hat,var_beta,alpha,order):
    Z_CI = norm.ppf(1-alpha/2)
    #95% confidence intervals of beta
    sigBeta=np.sqrt(var_beta)
    beta_lower = beta_hat - Z_CI*sigBeta
    beta_upper = beta_hat + Z_CI*sigBeta
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
    plt.title(f"Regression coefficients with their {(1-alpha)*100}% CI for {order}"+orderStr+ " order pol.")
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.tight_layout()
    #plt.savefig(datafile_path+ofile+"_"+solv+"_"+dist+'_nh_'+str(nh)+"_RBM_"+str(RBM)+'_.pdf')
    #plt.legend()
    if(savePlot == True):
        print("saving confidence interval plot in: ", f'exercise_results/exercise1/CI_beta_order{order}.png')
        plt.savefig( f'exercise_results/exercise1/CI_beta_order{order}.png')
    #plt.show()
#===============================================================================
