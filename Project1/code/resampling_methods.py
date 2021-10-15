import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from functions import OLS,ridge,lasso,desMat,getScores,addNoise,getMSE,getR2,StandardPandascaler
from regression_methods import linReg





#===============================================================================
#===============================================================================
def crossValidation(regMeth,scoreNames,X,z,K,sigma,shuffle,sklearnOLS,sklearnCV,lam):

    #Scores from each fold will be stored as row vectors in crossValScores
    crossValScores = np.zeros((K,len(scoreNames)))


    X = StandardPandascaler(X)
    f = z #Target function
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
    #beta_hat = np.zeros(X.shape[1])#Initialize the optimal reg. param. vector

    if(sklearnCV==False):
        for i in range(0,K):
            n_samp = int(n/K) #number of datapoints in each of the K samples
            z_train = np.concatenate((z[:(i)*n_samp], z[(i+1)*n_samp:]),axis=0) #Concatenate vertically
            z_test = z[(i)*n_samp:(i+1)*n_samp]
            f_train = np.concatenate((f[:(i)*n_samp], f[(i+1)*n_samp:]),axis=0) #Concatenate vertically
            f_test = f[(i)*n_samp:(i+1)*n_samp]
            X_train = np.concatenate((X[:i*n_samp,:], X[(i+1)*n_samp:,:]),axis=0)
            X_test = X[i*n_samp:(i+1)*n_samp,:]

            if(regMeth=='OLS'):
                beta_hat= OLS(X_train,z_train,sklearnOLS)
            if(regMeth=='ridge'):
                beta_hat= ridge(X_train,z_train,lam)
            if(regMeth=='lasso'):
                beta_hat= lasso(X_train,z_train,lam)

            z_tilde = X_train@beta_hat
            z_predict = X_test@beta_hat

            scoreValues = getScores(scoreNames,z_test,f_test,z_train,z_predict,z_tilde)
            for ind,score in enumerate(scoreNames):
                crossValScores[i,ind] = scoreValues[ind]


    elif(sklearnCV==True):
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

            scoreValues = getScores(scoreNames,z_test,f_test,z_train,z_predict,z_tilde)
            for ind,score in enumerate(scoreNames):
                crossValScores[i,ind] = scoreValues[ind]

            i+=1

    scoreMeans = np.mean(crossValScores,0)
    scoreVars = np.var(crossValScores,0)

    return scoreMeans,scoreVars

#===============================================================================
#===============================================================================


#===============================================================================
#===============================================================================
def bootstrap(regMeth,scoreNames,X,z,B,sigma,scaling,sklearn,lam): #z is the original data sample and B is the number of bootstrap samples
    n = len(z)
    #Scores from each bootstrap cycle will be stored as row vectors in bootScores
    bootScores = np.zeros((B,len(scoreNames)))

    for b in range(0,B):  #Loop through bootstrap cycles
        z_star = np.zeros(n)
        X_star = np.zeros((n,X.shape[1]))
        #Form a bootstrap sample,z_star, by drawing w. replacement from the original sample
        # zStarIndeces = np.random.randint(0,n,z.shape)
        # z_star = z[zStarIndeces]
        for i in range(0,n):
            zInd = np.random.randint(0,n)
            z_star[i] = z[zInd]
            X_star[i,:] = X[zInd,:]

        scoreValues = linReg(regMeth,scoreNames,X_star,z_star,sigma,scaling,sklearn,lam)[0]
        for ind,score in enumerate(scoreNames):
            bootScores[b,ind] = scoreValues[ind]

    scoreMeans = np.mean(bootScores,0)
    scoreVars = np.var(bootScores,0)



    return scoreMeans,scoreVars
#===============================================================================
#===============================================================================




# #===============================================================================
# #
# #===============================================================================
# def crossValidation(X,z,K,sigma,shuffle,sklearnOLS,sklearnCV,regMeth,lam):
#     #Scale the data:
#     #X = desMat(xr,yr,order)
#
#     X = StandardPandascaler(X)
#     f = z #Target function
#     z = addNoise(z,sigma)
#     if(z.shape[0]==1):
#         z.reshape(-1,1)
#         f.reshape(-1,1)
#     n =len(z) #Total number of datapoints
#     if(shuffle==True):
#         indices = np.arange(n)
#         np.random.shuffle(indices) #Shuffle the indixes and use this to shuffle X and z
#         X = X[indices,:]
#         z = z[indices]
#         f = f[indices]
#
#     bias_vec = np.zeros(K)
#     variance_vec = np.zeros(K)
#     cov_vec = np.zeros(K)     #Covariance of z_test and z_pred, an approx of cov(f_test,z_pred)
#     MSEtest_vec =np.zeros(K)
#     MSEtrain_vec = np.zeros(K)
#     R2test_vec = np.zeros(K)
#     R2train_vec = np.zeros(K)
#
#     beta_hat = np.zeros((X.shape[1],1))#Initialize the optimal reg. param. vector
#
#     if(sklearnCV==False):
#         for i in range(0,K):
#             n_samp = int(n/K) #number of datapoints in each of the K samples
#             z_train = np.concatenate((z[:(i)*n_samp], z[(i+1)*n_samp:]),axis=0) #Concatenate vertically
#             z_test = z[(i)*n_samp:(i+1)*n_samp]
#
#             f_train = np.concatenate((f[:(i)*n_samp], f[(i+1)*n_samp:]),axis=0) #Concatenate vertically
#             f_test = f[(i)*n_samp:(i+1)*n_samp]
#
#             X_train = np.concatenate((X[:i*n_samp,:], X[(i+1)*n_samp:,:]),axis=0)
#             X_test = X[i*n_samp:(i+1)*n_samp,:]
#
#             if(regMeth=='OLS'):
#                 beta_hat= OLS(X_train,z_train,sklearnOLS)
#             if(regMeth=='ridge'):
#                 beta_hat= ridge(X_train,z_train,lam)
#             if(regMeth=='lasso'):
#                 beta_hat= lasso(X_train,z_train)
#             #beta_hat = ridge(X_train,z_train,lambda)
#             #beta_hat = lasso(X_train,z_train,lambda)
#             z_tilde = X_train@beta_hat
#             z_predict = X_test@beta_hat
#
#             bias_vec[i],variance_vec[i],cov_vec[i],MSEtest_vec[i],MSEtrain_vec[i]\
#             ,R2test_vec[i],R2train_vec[i] = getScores(z_test,f_test,z_train,z_predict,z_tilde)
#
#     elif(sklearnCV==True):
#         kf = KFold(n_splits=K)
#         i = 0
#         for train_index, test_index in kf.split(z):
#             #print(train_index,test_index)
#             z_train = z[train_index]
#             z_test = z[test_index]
#
#             f_train = f[train_index]
#             f_test = f[test_index]
#
#             X_train = X[train_index,:]
#             X_test = X[test_index,:]
#
#             LR = LinearRegression()
#             LR.fit(X_train,z_train)
#             z_tilde = LR.predict(X_train)
#             z_predict = LR.predict(X_test)
#
#             bias_vec[i],variance_vec[i],cov_vec[i],MSEtest_vec[i],MSEtrain_vec[i]\
#             ,R2test_vec[i],R2train_vec[i] = getScores(z_test,f_test,z_train,z_predict,z_tilde)
#             i+=1
#
#     bias = np.mean(bias_vec)
#     var = np.mean(variance_vec)
#     cov = np.mean(cov_vec)
#     MSEtest = np.mean(MSEtest_vec)
#     MSEtrain = np.mean(MSEtrain_vec)
#     R2test = np.mean(R2test_vec)
#     R2train = np.mean(R2train_vec)
#
#
#     R2test_vec = np.array(R2test_vec)
#     MSEtest_vec = np.array(MSEtest_vec)
#
#     return bias,var,cov,MSEtest,MSEtrain,R2test,R2train
#
# #===============================================================================
# #===============================================================================
