import numpy as np
from sklearn.preprocessing import StandardScaler



from neural_network import FFNN


def gridsearch(itIndices, hyperparams, regression=True):

    epoch_vals= [20,50,80,100,150,200,250,300]
    M_vals = [5,10,20,30,50,80,100,150]
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)


    valList = [epoch_vals,  M_vals, eta_vals,  lmbd_vals] #List containing values we want to loop over

    iterableStr = ['epochs','batch size','eta','lambda']
                #       0        1         2       3

    #itIndices=[2,3] #Pick which variables to iterate over

    iterable1 = valList[itIndices[0]]
    iterable2 = valList[itIndices[1]]

    score_train = np.zeros((len(iterable1), len(iterable2)))
    score_test = np.zeros((len(iterable1), len(iterable2)))

    # score_train_sk = np.zeros((len(iterable1), len(iterable2)))
    # score_test_sk = np.zeros((len(iterable1), len(iterable2)))


    for i, it1 in enumerate(iterable1):
        for j, it2 in enumerate(iterable2):
            hyperparams[itIndices[0]] = it1
            hyperparams[itIndices[1]] = it2
            #Using our own NN
            MLP = FFNN(X_train, y_train, nhidden, lengths,sigmoidL, softmaxL,derCrossEntropy,True, hyperparams)
            MLP.initializeNetwork()
            MLP.train()

            output_train, output_test = MLP.feedForward(X_train), MLP.feedForward(X_test)
            if(regression==True):
                score_train[i,j]= getMSE(y_train, output_train)
                score_test[i,j] = getMSE(y_test, output_test)
            else:
                score_train[i,j] = accuracy(output_train, y_train, True)
                score_test[i,j] = accuracy(output_test, y_test, True)

            #Using sklearn's NN
            # MLP_sklearn = MLPClassifier(hidden_layer_sizes = lengths[1:-1], activation = 'logistic', solver = 'sgd',
            # alpha = lmbd, batch_size = M, learning_rate_init = eta, max_iter = epochs, momentum = 0)
            # MLP_sklearn.fit(X_train,y_train)
            # output_train_sk, output_test_sk = MLP_sklearn.predict(X_train), MLP_sklearn.predict(X_test)
            # acc_train_sk, acc_test_sk = accuracy(output_train_sk, y_train, True), accuracy(output_test_sk, y_test, True)
            # train_accuracies_sk[i,j] = acc_train_sk
            # test_accuracies_sk[i,j] = acc_test_sk
    if(regression):
        it1_opt = np.where(score_test == np.min(score_test))[0][0]
        it2_opt = np.where(score_test == np.min(score_test))[1][0]
    else:
        it1_opt = np.where(score_test == np.max(score_test))[0][0]
        it2_opt = np.where(score_test == np.max(score_test))[1][0]

    return score_train, score_test, it1_opt, it2_opt


def scale_data(X_train,X_test):

    #Scale X_train using the mean and std of its own columns
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    #Use mean and std of the columns of X_train to scale X_test
    # n_features = X_train.shape[1]
    # X_test_scaled = X_test
    # for i in range(n_features):
    #     mu = np.mean(X_train[:,i])
    #     sigma = np.std(X_train[:,i])
    #     X_test_scaled[:,i] = (X_test[:,i]-mu)/sigma

    return X_train_scaled, X_test_scaled


def accuracy(y, pred):
    """
    Accuracy score for classification.
    """
    y = np.argmax(y, axis = 1)
    pred = np.argmax(pred, axis = 1)

    return np.sum(y==pred)/len(y)

def shuffle_in_unison(X,z):
    assert len(X) == len(z)
    p = np.random.permutation(len(z))
    return X[p], z[p]

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def addNoise(z,sigma):
    n = len(z)
    z_noise = np.zeros(n)
    for i in range(0,n):
        epsilon = np.random.normal(0,sigma)
        z_noise[i]=z[i] + epsilon
    return z_noise

def desMat(xr,yr,p):
    N = len(xr)
    numEl = int((p+2)*(p+1)/2)#Number of elements in beta
    X = np.ones((N,numEl))
    colInd=0#Column index
    for l in range(1,p+1):
        for k in range(0,l+1):
            X[:,colInd+1] = (xr**(l-k))*yr**k
            colInd = colInd+1
    return X

def desMat1D(x,p):
    N = len(x)
    X = np.ones(N,p+1)
    for i in range(p+1):
        X[:,p] = x**i
    return X


def getR2(z,z_tilde):
    n = len(z)
    num = np.sum((z-z_tilde)**2)
    denom = np.sum((z-np.mean(z))**2)
    return 1-num/denom

def getMSE(z,z_tilde):
    return np.sum((z-z_tilde)**2)/len(z)
