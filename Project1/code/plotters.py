import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import os
from imageio import imread

from functions import addNoise,desMat
from regression_methods import linReg
from resampling_methods import crossValidation, bootstrap
from sklearn.model_selection import train_test_split
#from main import generateDesMatPars


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

    heatMapFilename ="score_"+regMeth+"_"+resampMeth[0]+"_"\
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

    #====================================================================================
    #====== Use the found scores to generate a plot for this iteration of sigma =========
    #====================================================================================
    for s,sigma in enumerate(sigmas):
        heatMapTitle,normPlotTitle, heatMapFilename,normPlotFilename = \
        titleMaker(terrainBool,sigma,regMeth,resampMeth,nBoot,K,n,lambdas,orders)
        #titleMaker(regMeth,resampMeth,nBoot,K,lambdas,sigmas[s],n,orders)
        if(nLambdas>1 and nOrders>1): #Run only if both vectors are greater than one
            for scoreStr in scoreNames:
                #========================================================================
                #================ Make heatmap using seaborn ============================
                #========================================================================
                fig0 = plt.figure()
                scoreMat = calcRes[scoreStr][:,:,s]
                lmds_formatted=lambdas
                for i in range(0,nLambdas):
                    lmds_formatted[i] = float("{:.2f}".format(lambdas[i]))
                M = pd.DataFrame(scoreMat,lmds_formatted,orders)
                if(scoreStr == 'R2test' or scoreStr == 'R2train'):
                    ax = sns.heatmap(M,cmap ="YlGnBu",linewidths=.0,annot = not savePlot)
                    #ax = sns.heatmap(M,cmap="BuPu_r",linewidths=.0,annot = not savePlot)
                if(scoreStr == 'MSEtest' or scoreStr == 'MSEtrain'):
                    ax = sns.heatmap(M,linewidths=.0,annot = not savePlot)
                    #ax = sns.heatmap(M,cmap ="Greens",linewidths=.0,annot = not savePlot)
                if(scoreStr == 'variance'):
                    ax = sns.heatmap(M,cmap ="Blues",linewidths=.0,annot = not savePlot)
                if(scoreStr == 'bias'):
                    ax = sns.heatmap(M,cmap ="Reds",linewidths=.0,annot = not savePlot)
                    # ax = sns.heatmap(M,cmap ="YlGnBu",linewidths=.0,annot = not savePlot)
                ax.invert_yaxis()
                ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
                ax.set_xlabel('Polynomial degree, p',size='12',**{'fontname':fonts[fontInd]})
                ax.set_ylabel(r'log$_{10}(\lambda)$', size = '12',**{'fontname':fonts[fontInd]})
                plt.title(scoreStr + heatMapTitle,size=12, **{'fontname':fonts[fontInd]})
                if(savePlot):
                    plt.savefig(subfolder+"/"+scoreStr+heatMapFilename+'.png')
                plt.show()

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
                plt.savefig(subfolder+"/"+joinedScoreStr+normPlotFilename + '.png')
                # f'_scorelen={len(scoreNames)}'
            plt.show()
            #========================================================================
            #========================================================================


#====================================================================================================================
#--------------------------------------------------------------------------------------------------------------------
#       Makes surface plots of both the original data and the fitted function
#--------------------------------------------------------------------------------------------------------------------
#====================================================================================================================


#def surfacePlotter(tinkerMode,savePlot,xr,yr,z_orig,z_tilde,sigma,order,lmd,regmeth)
def surfacePlotter(terrainBool,tinkerBool,savePlot,xr,yr,z_orig,*args):
    n = int(np.sqrt(len(xr)))
    xx = xr.reshape(n,n)
    yy = yr.reshape(n,n)

    #Setting default optional argument values:
    #---------------------------------------------------------------------------
    z_fitted,         sigma, order,  lmd, regMeth =\
    np.zeros(n**2+1),    0 ,     5, 1e-5, 'OLS'

    # z_fitted,order,sigma,regMeth,lmd =Z_orig, 5, 0.1, 'OLS', 0

    paramList = [z_fitted, sigma, order, lmd, regMeth]

    for i in range(0,len(args)):
        paramList[i] = args[i]

    z_fitted, sigma, order, lmd, regMeth = paramList
    #---------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    #    If Z_tilde is not included in the function's argument list, plot only Z_orig.
    #    Otherwise plot both
    #-------------------------------------------------------------------------------


    #surf = ax.plot_surface(xx, yy, Z_orig, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #surf = ax.plot_surface(xx, yy, Z_orig, cmap=cm.coolwarm,linewidth=0.3, edgecolor = 'k', alpha= 0.8,antialiased=False)
    # surf = ax.plot_surface(xx,yy,Z_orig, cmap='viridis_r', linewidth=0.3, alpha = 0.8, edgecolor = 'k')
    # surf = ax.plot_surface(xx,yy,Z_orig, cmap='coolwarm_r', linewidth=0.3, alpha = 0.7, edgecolor = 'k')

    if(len(z_fitted)!=len(z_orig)):
    #if(len(z_fitted)==len(z_orig)):
        Z_orig = z_orig.reshape(n,n)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #surf = ax.plot_surface(xx, yy, Z_orig, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #surf = ax.plot_surface(xx, yy, Z_orig, cmap=cm.coolwarm,linewidth=0.3, edgecolor = 'k', alpha= 0.8,antialiased=False)
        #surf = ax.plot_surface(xx,yy,Z_orig, cmap='coolwarm_r', linewidth=0.3, alpha = 0.7, edgecolor = 'k')
        surf = ax.plot_surface(xx,yy,Z_orig, cmap='viridis', linewidth=0.3, alpha = 0.8, edgecolor = 'k')
        # Plot the surface.

        # Customize the z axis.
        #ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(20, 30)
        #m_dataTypeStr = "Terrain data" if(not terrainBool) else 'Franke Function, no noise'
        if(terrainBool):
            plt.title(r"Terrain data"+"\n"+ f"n={n}, " +"$\sigma_{\epsilon}$" + f"= {sigma}")
        else:
            plt.title(r"The Franke function"+"\n"+ f"n={n}")
        plt.show()
    else:
        Z_orig, Z_fitted = z_orig.reshape(n,n), z_fitted.reshape(n,n)
        fig = plt.figure(figsize=(8,4))
        #First plot
        ax = fig.add_subplot(121, projection='3d')
        #surf = ax.plot_surface(xx,yy,Z_orig, cmap = cm.coolwarm, linewidth=0, antialiased=False)
        surf = ax.plot_surface(xx,yy,Z_orig, cmap='viridis', linewidth=0.3, alpha = 0.8, edgecolor = 'k')
        #m_dataTypeStr = "Terrain data" if(terrainBool) else 'Franke Function, no noise'
        if(terrainBool):
            ax.set_title("Terrain data"+"\n"+ f"n={n}")
        else:
            ax.set_title("Franke function"+"\n"+ f"n={n}" + r", $\sigma_{\epsilon}$" + f"= {sigma:.4f}")
        #Customize the z axis.
        #ax.set_zlim(-0.10, 1.40)
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
        # Z_tilde = np.zeros(Z_orig.shape)
        #surf = ax.plot_surface(xx,yy,Z_fitted, cmap = cm.coolwarm, alpha=1,linewidth=0, antialiased=False)
        surf = ax.plot_surface(xx,yy,Z_fitted, cmap='viridis', linewidth=0.3, alpha = 0.8, edgecolor = 'k')
        titleStr = ''
        if(regMeth=='ridge' or regMeth=='lasso'):
            titleStr = f', '+r'$\log{\lambda}=$'+f'{lmd:0.2f}'# f'{np.log10(lambdas[0]):.2f}'
        ax.set_title(regMeth+f' fit\n pol.deg={order}'+titleStr)
        #ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.view_init(20, 30)
        if(tinkerBool==True):
            subfolder_surfplot = "tinkerings/surface_plots/"
        elif(tinkerBool==False):
            subfolder_surfplot = "exercise_results/surface_plots/"
        #if(savePlot):
        if not os.path.exists(subfolder_surfplot):
            os.makedirs(subfolder_surfplot)
        #tinkerStr = 'tinkering/' if tinkerBool else 'exercises/'
        if(savePlot):
            plt.savefig(subfolder_surfplot+f'linReg_surface_order={order}__n={n}__'\
            +r'\sigma_{\epsilon}$=' + f'{sigma}.png')
        #print('From end of surfacePlotter')
        plt.show()
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
    plt.title(f"Regression coefficients with their 95% CI for {order}"+orderStr+ " order pol.")
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.tight_layout()
    #plt.savefig(datafile_path+ofile+"_"+solv+"_"+dist+'_nh_'+str(nh)+"_RBM_"+str(RBM)+'_.pdf')
    #plt.legend()
    if(savePlot == True):
        plt.savefig(f'CI_beta_order{order}.png')
    plt.show()
#===============================================================================
