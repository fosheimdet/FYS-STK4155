import numpy as np
from scipy import signal
def cross_entropy(AL,y):
    return -np.sum(y*np.log(AL))


def finite_diff(model,layer_ind,X, y):
    K = model.layers[layer_ind].K
    #output = model.predict(test)
    # cost_before = np.sum(output[0])
    print(X[0:2,:,:].shape)

    output = model.predict(X[0:2,:,:])
    # cost_before = -np.sum(y[sample_ind,:]*np.log(output[sample_ind,:]))
    cost_before = -np.sum(y*np.log(output))

    model.backpropagate(y)

    Atilde = np.pad(X,((0,0),(1,1),(1,1)))
    delCdelK_backprop = signal.correlate2d(Atilde[0],
                model.layers[layer_ind].delta[0], mode="valid")
    #Adding contributions from all samples, as is done in DNN
    for n in range(1,X.shape[0]):
        delCdelK_backprop+=signal.correlate2d(Atilde[n],model.layers[layer_ind].delta[n], mode="valid")

    print("delCdelK_backprop: \n ", delCdelK_backprop)


    dw = 1e-5
    delCdelK_num = np.zeros(delCdelK_backprop.shape)
    for u in range(delCdelK_backprop.shape[0]):
        for v in range(delCdelK_backprop.shape[1]):
            K[u,v]+=dw
            output2 = model.predict(X)
            # cost_after = np.sum(output2[0])
            # cost_after =-np.sum(y[sample_ind,:]*np.log(output2[sample_ind,:]))
            cost_after = -np.sum(y*np.log(output2))
            delCdelK_num[u,v] = (cost_after-cost_before)/dw
            K[u,v]-=dw

    print("delCdelK_num: \n ", delCdelK_num)
    # print(delCdelK_num)
