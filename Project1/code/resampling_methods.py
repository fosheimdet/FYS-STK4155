import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from functions import OLS,ridge,lasso,desMat,getScores,addNoise,getMSE,getR2,StandardPandascaler
from regression_methods import linReg

#===============================================================================
#===============================================================================
def crossValidation(regMeth,emptyScoreScalars,X,sigma,lmd,z,scaling,skOLS,skCV,shuffle,K):

    #Scores from each fold will be stored as row vectors in crossValScores
    #nScores = len(scoreNames)
    CVscoreVectors={} #Dict of vectors, corresponding to the different scores,
    #the elements of which are the score values obtained in a given fold
    for scoreName in emptyScoreScalars:
        CVscoreVectors[scoreName] = np.zeros(K)
    X_temp = X
    if(scaling):
        X = StandardPandascaler(X_temp)

    f = z   #Target function. Used for calculating bias
    z_temp = addNoise(z,sigma)
    z=z_temp

    n = int(np.sqrt(len(z)))
    Z_orig = z.reshape(n,n)

    if(z.shape[0]==1):
        z.reshape(-1,1)
        f.reshape(-1,1)
    n =len(z) #Total number of datapoints
    if(shuffle==True):
        indices = np.arange(n)
        np.random.shuffle(indices) #Shuffle the indixes and use this to shuffle X and z
        X = X[indices,:]
        z = z[indices]
        f = f[indices]

    beta_hat = np.zeros((X.shape[1],1))#Initialize the optimal reg. param. vector

    if(skCV==False):
        for i in range(0,K):
            n_samp = int(n/K) #number of datapoints in each of the K samples
            z_train = np.concatenate((z[:(i)*n_samp], z[(i+1)*n_samp:]),axis=0) #Concatenate vertically
            z_test = z[(i)*n_samp:(i+1)*n_samp]
            f_train = np.concatenate((f[:(i)*n_samp], f[(i+1)*n_samp:]),axis=0) #Concatenate vertically
            f_test = f[(i)*n_samp:(i+1)*n_samp]
            X_train = np.concatenate((X[:i*n_samp,:], X[(i+1)*n_samp:,:]),axis=0)
            X_test = X[i*n_samp:(i+1)*n_samp,:]

            if(regMeth=='OLS'):
                beta_hat= OLS(X_train,z_train,skOLS)
            if(regMeth=='ridge'):
                beta_hat= ridge(X_train,z_train,lmd)
            if(regMeth=='lasso'):
                beta_hat= lasso(X_train,z_train,lmd)

            z_tilde = X_train@beta_hat
            z_predict = X_test@beta_hat


            scoreScalars = getScores(emptyScoreScalars,z_test,f_test,z_train,z_predict,z_tilde)
            for scoreName in scoreScalars:
                CVscoreVectors[scoreName][i] =scoreScalars[scoreName] #i'th fold result

    elif(skCV==True):
        kf = KFold(n_splits=K)
        i = 0
        for train_index, test_index in kf.split(z):
            z_train = z[train_index]
            z_test = z[test_index]
            f_train = f[train_index]
            f_test = f[test_index]
            X_train = X[train_index,:]
            X_test = X[test_index,:]

            LR = LinearRegression()
            LR.fit(X_train,z_train)
            z_tilde = LR.predict(X_train)
            z_predict = LR.predict(X_test)


            scoreScalars = getScores(emptyScoreScalars,z_test,f_test,z_train,z_predict,z_tilde)
            for scoreName in scoreScalars:
                CVscoreVectors[scoreName][i] = scoreScalars[scoreName]

            i+=1

    scoreMeans = {}
    scoreVars = {}
    for scoreName in CVscoreVectors:
        scoreMeans[scoreName] = np.mean(CVscoreVectors[scoreName])
        scoreVars[scoreName] = np.var(CVscoreVectors[scoreName])

    # print("scoreMeans: \n", scoreMeans)
    # print("scoreVars: \n", scoreVars)
    return scoreMeans,scoreVars

#===============================================================================
#===============================================================================


#===============================================================================
#===============================================================================

def bootstrap(regMeth,emptyScoreScalars,X,sigma,lmd,z,scaling,skOLS,nBoot): #z is the original data sample and B is the number of bootstrap samples
    n = len(z)
    #Scores from each bootstrap cycle will be stored as row vectors in bootScores
    #bootScores = np.zeros((nBoot,len(scoreNames)))
    bootScoreVectors = {}
    for scoreName in emptyScoreScalars:
        bootScoreVectors[scoreName] = np.zeros(nBoot)


    for b in range(0,nBoot):  #Loop through bootstrap cycles
        z_star = np.zeros(n)
        X_star = np.zeros((n,X.shape[1])) #X.shape[1] = (p+2)(p+1)/2 = len(beta)
        #Form a bootstrap sample,z_star, by drawing w. replacement from the original sample
        # zStarIndeces = np.random.randint(0,n,z.shape)
        # z_star = z[zStarIndeces]
        for i in range(0,n):
            zInd = np.random.randint(0,n)
            z_star[i] = z[zInd]
            X_star[i,:] = X[zInd,:]

        scoreScalars = linReg(regMeth,emptyScoreScalars,X_star,sigma,lmd,z_star,scaling,skOLS)[0]
        for scoreName in scoreScalars:
            bootScoreVectors[scoreName][b] = scoreScalars[scoreName]

    scoreMeans = {}
    scoreVars = {}

    for scoreName in scoreScalars:
        scoreMeans[scoreName] = np.mean(bootScoreVectors[scoreName])
        scoreVars[scoreName] = np.var(bootScoreVectors[scoreName])


    return scoreMeans,scoreVars
#===============================================================================
#===============================================================================
