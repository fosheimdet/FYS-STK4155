'''
PROJECT 2
Linus Hoetzel
Erasmus Exchange Student
'''
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

def r2(y_data, y_model):
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def mse(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y, noise):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise*np.random.normal(0,1,x.shape)
    #noise added here with function call FrankeFunction()

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X

# activation functions:
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def relu(x):
    if x.all() > 0:
	    return x
    else:
	    return 0
def leakyRelu(x):
    a = 0.01
    if x.all() > 0:
        return x
    else:
        return a * x

# feed-forward pass
# How wrong is the current set of parameters?
def feed_forward(X):
    # weighted sum of inputs to the HIDDEN layer --> x*w + b
    z_h = np.matmul(X, hidden_weights) + hidden_bias

    # activation in the hidden layer --> a*(w*x +b)
    a_h = sigmoid(z_h)

    # with different activation functions
    #a_h = relu(z_h)
    #a_h = leakyRelu(z_h)
    
    # weighted sum of inputs to the OUTPUT layer --> w*x + b
    z_o = np.matmul(a_h, output_weights) + output_bias

    # softmax output
    # axis 0 holds each input and axis 1 the predictedOutput of each category
    exp_term = np.exp(z_o)
    predictedOutput = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    return predictedOutput

# new feed_forward loop to get activation functions of hidden layers a_h
# a_h needed for backpropagation but not for the prediction
def feed_forward_train(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias

    # activation in the hidden layer
    a_h = sigmoid(z_h)
    # with different activation functions
    #a_h = relu(z_h)
    #a_h = leakyRelu(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias

    # softmax output
    # axis 0 holds each input and axis 1 the predictedOutput of each category
    exp_term = np.exp(z_o)
    predictedOutput = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    # for backpropagation need activations in hidden and output layers
    return a_h, predictedOutput

def feed_forward_optimal(X, optimal_hidden_weights, optimal_hidden_bias, optimal_output_weights, optimal_output_bias):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, optimal_hidden_weights) + optimal_hidden_bias

    # activation in the hidden layer
    a_h = sigmoid(z_h)
    # with different activation functions
    #a_h = relu(z_h)
    #a_h = leakyRelu(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, optimal_output_weights) + optimal_output_bias

    # softmax output
    # axis 0 holds each input and axis 1 the predictedOutput of each category
    exp_term = np.exp(z_o)
    predictedOutput = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    return predictedOutput

# set up of backpropagation algorithm  
# how should we change the set of parameters (w,b) to achieve better prediction
def backpropagation(X, Y):
    a_h, predictedOutput = feed_forward_train(X)

    # error in the output layer
    error_output = predictedOutput - Y
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

##### Set up of data #####
np.random.seed(1214) 

#data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x,y)
Z = FrankeFunction(X, Y, 0.15) # increased to 0.5 for higher bias
x = X.ravel()
y = Y.ravel()
z = Z.ravel()

#degree of polynomial
p = 5
#  The design matrix now as function of a given polynomial
X = create_X(x, y, p)

# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, z, test_size = 0.2)

##### Exercise b) and c) -  Back propagation of neural network #####
# building neural network
# z: input
# a: activation

# setting up weights and biases based on random variables
n_inputs, n_features = X_train.shape # --> (320 inputs, 21 features)
n_hidden_neurons = 50
n_categories = len(y_train) # neurons in the output layer

# weights and bias set up
# wights normally distributed
hidden_weights = np.random.randn(n_features, n_hidden_neurons) # dimension (21, 50)
output_weights = np.random.randn(n_hidden_neurons, n_categories) # dimension (50, 10)

# bias initialized with zero and 0.01 added to ensure not zero for ouput
output_bias = np.zeros(n_categories) + 0.01  # dimension (10,)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01  # dimension (50,)

##### setting up feed forward and calculate predictions #####
#apply feed forward algorithm
predictedOutput = feed_forward(X_train)

# cost function with respect to mse
# model evaluation for training set
print("MSE of training data without backpropagation: " + str(mse(y_train, predictedOutput)))

##### ridge regression, back propagation and one-hot-encoding of y #####
# one hot encoding with keras library
Y_train_onehot  = to_categorical(y_train, num_classes=n_categories)
Y_test_onehot   = to_categorical(y_test, num_classes=n_categories)

# set up of ridge regression
#  test the learning rates µ=10−6,10−5,...,10−1 with different regularization parameters lmbd=10−6,...,10−0
µ     = 0.1      # learning rate
lmbd  = 0.01     # ridge parameter

# set up of batches and epochs
n = n_inputs    #number datapoints 
M = 15          #size of each mini-batch
m = int(n/M)    #number of minibatches
n_epochs = 10   #number of epochs

for epoch in range(1, n_epochs+1):
    for i in range(m):
        #Pick the k-th minibatch at random
        k = np.random.randint(m) 

        xi = X_train[k:k+1]
        yi = Y_train_onehot[k:k+1]
        
        # calculate gradients through backpropagation
        gradientWeightsOutput, gradientBiasOutput, gradientWeightsHidden, gradientBiasHidden = backpropagation(xi, yi)
        
        # gradients with ridge regression --> dw = dw + ridgelmbd*w
        gradientWeightsOutput += lmbd * output_weights
        gradientWeightsHidden += lmbd * hidden_weights
        
        # update weights and biases with method of gradient descent tetha_i+1 = theta_i - µ*dtheta_i
        output_weights  -=   µ * gradientWeightsOutput
        output_bias     -=   µ * gradientBiasOutput
        hidden_weights  -=   µ * gradientWeightsHidden
        hidden_bias     -=   µ * gradientBiasHidden

# do the feedforward prediction with optimal weights and biases from the backpropagation
predictedOutputOptimal = feed_forward_optimal(X_train, hidden_weights, hidden_bias, output_weights, output_bias)
predictedOutputOptimalTest = feed_forward_optimal(X_test, hidden_weights, hidden_bias, output_weights, output_bias)

# cost function with respect to mse after backpropagation
# model evaluation for training set
print("MSE of training data with backpropagation: " + str(mse(y_train, predictedOutputOptimal)))
print("MSE difference before and after backpropagation: " + str(mse(y_train, predictedOutputOptimal) - mse(y_train, predictedOutput)))
# model evaluation on testing set
#print("MSE of testing data with backpropagation: " + str(mse(y_test, predictedOutputOptimalTest)))

'''
Hyperparameters to tune:
---------------------------------
- number of hidden neurons
- learning rate
- ridge parameter
- size and number of mini-batches
- number of epochs
'''