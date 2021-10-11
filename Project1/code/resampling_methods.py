import numpy as np
from sklearn.utils import resample

from functions import getMSE,getR2,StandardPandascaler
from regression_methods import OLS



#===============================================================================
#
#===============================================================================
def crossValidation(X,z,f,K,shuffle):
    #Scale the data:
    X = StandardPandascaler(X)
    #Reshape to column vector if z is given as a row vector
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

    n_test = int(n/K) #Number of data points in the test set
    remainder = np.mod(n,K)

    MSEtest_vec =np.zeros(K)
    bias_vec = np.zeros(K)
    variance_vec = np.zeros(K)
    cov_vec = np.zeros(K)     #Covariance of z_test and z_pred, an approx of cov(f_test,z_pred)
    R2test_vec = np.zeros(K)

    for i in range(0,K):
        n_samp = int(n/K) #number of datapoints in each of the K samples
        z_train = np.concatenate((z[:(i)*n_samp], z[(i+1)*n_samp:]),axis=0) #Concatenate vertically
        z_test = z[(i)*n_samp:(i+1)*n_samp]

        f_train = np.concatenate((f[:(i)*n_samp], f[(i+1)*n_samp:]),axis=0) #Concatenate vertically
        f_test = f[(i)*n_samp:(i+1)*n_samp]

        X_train = np.concatenate((X[:i*n_samp,:], X[(i+1)*n_samp:,:]),axis=0)
        X_test = X[i*n_samp:(i+1)*n_samp,:]

        beta = np.linalg.pinv(X_train.T@X_train)@X_train.T@z_train
        z_tilde = X_train@beta
        z_predict = X_test@beta

        EZ_tilde = np.mean(z_predict)
        MSE_test = getMSE(z_test,z_predict)
        bias = getMSE(f_test,EZ_tilde)
        variance = getMSE(z_predict,EZ_tilde)
        cov = np.cov(f_test,z_predict)[0,1] #We use z_test as an approximation for f_test
        R2_test = getR2(z_test,z_predict)


    #     EZ_tilde = np.mean(z_predict)
    #     MSE_test = getMSE(z_test,z_predict)
    #     bias =getMSE(f_test,EZ_tilde)
    #     #bias =getMSE(z_test,EZ_tilde)
    #     variance =getMSE(z_predict,EZ_tilde)
    # cov_f_ytilde= np.cov(f_test.reshape(1,-1),z_predict.reshape(1,-1))[0,1]

        MSEtest_vec[i]=MSE_test
        bias_vec[i] = bias
        variance_vec[i] = variance
        cov_vec[i] = cov
        R2test_vec[i] = R2_test



        # bias[i],variance[i],X,Z_orig,Z_tilde,beta_hat,MSE[i],R2_test, \
        # MSE_train, R2_train, var_beta = OLS(xr,yr,Z,order,True,0)


    R2test_vec = np.array(R2test_vec)
    MSEtest_vec = np.array(MSEtest_vec)
    # print("R2-Score_OLS: %0.3f (+/- %0.3f)" % (R2test_vec.mean(), R2test_vec.std() * 2))
    # print("MSE-Score_OLS: %0.5f (+/- %0.5f)" % (MSEtest_vec.mean(), MSEtest_vec.std() * 2))
    return np.mean(MSEtest_vec), np.mean(bias_vec), np.mean(variance_vec), \
    np.mean(R2test_vec),np.mean(cov_vec)


#===============================================================================
#===============================================================================


#===============================================================================
#===============================================================================
def bootstrap(xr,yr,z,B,order,sigma,scaled,sklearn): #z is the original data sample and B is the number of bootstrap samples
    n = len(z)
    bias_vec =np.zeros(B)
    variance_vec = np.zeros(B)
    MSEtest_vec = np.zeros(B)
    cov_f_ytilde_vec = np.zeros(B)

    for bCycle in range(0,B):
        z_star = np.zeros(n)
        for j in range(0,n):
            z_star[j] = z[np.random.randint(0,n-1)]
        #z_star = resample(z)
        Z = z_star.reshape(int(np.sqrt(n)),int(np.sqrt(n)))

        cov_f_ytilde,bias,variance,X,Z_orig,Z_tilde,beta_hat,MSE_test,R2_test, \
        MSE_train, R2_train, var_beta = OLS(xr,yr,Z,order,sigma,scaled,sklearn)
        # We use sigma=0 since the noise is alrdy included in z
        bias_vec[bCycle] = bias
        variance_vec[bCycle] = variance
        MSEtest_vec[bCycle] = MSE_test
        cov_f_ytilde_vec[bCycle] = cov_f_ytilde



    MSEtest_mean = np.mean(MSEtest_vec)
    MSEtest_var = np.var(MSEtest_vec)
    bias_mean = np.mean(bias_vec)
    bias_var = np.var(bias_vec)
    variance_mean = np.mean(variance_vec)
    variance_var = np.var(variance_vec)
    covariance_mean = np.mean(cov_f_ytilde)
    return MSEtest_mean,MSEtest_var,bias_mean,bias_var,variance_mean,variance_var,covariance_mean
#===============================================================================
#===============================================================================
