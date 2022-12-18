import numpy as np
from scipy import signal
def cross_entropy(AL,y):
    return -np.sum(y*np.log(AL))


#Use finite differences to compare kernel gradients found in backpropagation
#with the numerical ones
def fd_kernel(model,layer_ind,X, y):
    K = model.layers[layer_ind].F[0,:,:,0]

    #=========Numerical=========
    output = model.predict(X)
    cost_before = -np.sum(y*np.log(output))

    dw = 1e-6
    grad_num = np.zeros(K.shape)
    for u in range(K.shape[0]):
        for v in range(K.shape[1]):
            K[u,v]+=dw #Actually alters the layer weights. K=... doesnt work
            output2 = model.predict(X)
            cost_after = -np.sum(y*np.log(output2))
            grad_num[u,v] = (cost_after-cost_before)/dw
            K[u,v]-=dw
    print("===== Kernel gradients =====")
    print("grad_num: \n ", grad_num)
    #====Backprop/analytical====
    model.backpropagate(y)
    Delta_F,delta_b = model.layers[layer_ind].update(X,0,0)
    print("grad_analytical: \n", Delta_F[0,:,:,0])

#Test bias gradients
def fd_biases(model,layer_ind,X, y):
    K = model.layers[layer_ind].F[0,:,:,0]
    b = model.layers[layer_ind].b
    #=========Numerical=========
    output = model.predict(X)
    cost_before = -np.sum(y*np.log(output))

    db = 1e-10
    bgrad_num = np.zeros(b.shape)
    for i in range(b.shape[0]):
        model.layers[layer_ind].b[i]+=db
        output2_b = model.predict(X)
        cost_after = -np.sum(y*np.log(output2_b))
        bgrad_num[i] = (cost_after-cost_before)/db
        model.layers[layer_ind].b[i]-=db
    print("===== bias gradients =====")
    print("bgrad_num: \n", bgrad_num)
    #====Backprop/analytical====
    model.backpropagate(y)
    delta_F,delta_b = model.layers[layer_ind].update(X,0,0)
    print("bgrad_analytical: \n", delta_b)
