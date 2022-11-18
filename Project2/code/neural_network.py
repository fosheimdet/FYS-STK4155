import numpy as np
from activation_functions import sigmoidL,reluL,leakyReluL, tanhL




#Classical way. X.shape = [n_inputs, n_features]
#Here the weights are indexed with rows corresponding to the neurons of the previous layer and columns corresponding to nodes of the current layer
class FFNN:
    def __init__(self, X, y, act_out_list, derCost, hyperparams, softmaxBool):

        act_h_dict = {"sigmoid": sigmoidL, "relu": reluL, "leaky relu": leakyReluL, "tanh": tanhL}

        self.epochs,self.batch_size,self.eta,self.lmbd, self.lengths_hidden, self.act_h_str = hyperparams
        self.act_h_list = act_h_dict[self.act_h_str]

        self.X_data_full = X
        self.y_data_full = y
        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_categories = y.shape[1]

        self.nhidden = len(self.lengths_hidden)
        self.lengths = [self.n_features] + self.lengths_hidden + [self.n_categories] #a list containing the number of neurons in each layer

        self.act_h, self.derAct_h = self.act_h_list
        self.act_out, self.derAct_out = act_out_list
        self.derCost = derCost
        self.softmaxBool = softmaxBool


    def initializeNetwork(self):
        self.W = [] #Weights
        self.b = [] #biases
        self.errors_initial = []

        #We add a dummy element to the lists to make the indexing more intuitive
        self.W.append(np.array([0]))
        self.b.append(np.array([0]))
        self.errors_initial.append(np.array([0]))

        for i in range(self.nhidden+1):
            self.W.append(np.random.normal(0,1,(self.lengths[i],self.lengths[i+1]))) #Draw from standard normal distributin
            #self.W.append(np.random.rand(self.lengths[i],self.lengths[i+1])) #Draw from a uniform dist. over [0,1)
            self.b.append(np.repeat(0.01,self.lengths[i+1]).reshape(1,-1))
            self.errors_initial.append(np.zeros((self.n_inputs,self.lengths[i+1])))

    def feedForward(self, X):
        self.A = [] #Activations
        #First layer
        self.A.append(X)
        #Hidden layers
        for i in range(self.nhidden):
            self.A.append(self.act_h(self.A[i]@self.W[i+1] + self.b[i+1] ))

        #Output layer
        AL = self.act_out(self.A[-1]@self.W[-1] + self.b[-1])
        self.A.append(AL)

        self.output = self.A[-1]

        return AL

    def predict(self,X):
        self.feedForward(X)

        return self.A[-1]


    def backpropagate(self,y):

        self.errors = self.errors_initial #Reset errors


        if(self.softmaxBool):
            self.errors[-1] = (self.output-y)
        else:
            #self.errors[-1] = 2*(self.output-y)/y.shape[0]
            self.errors[-1] = self.derAct_out(self.output)*self.derCost(self.output, y)

        # self.errors[-1] = deltaL(self.output,y,self.derAct_out,self.derCost)


        #Finding the remaining errors:
        for l in range(self.nhidden, 0, -1):
            self.errors[l] = (self.errors[l+1]@self.W[l+1].T)*self.derAct_h(self.A[l])


        for l in range(1,self.nhidden+2):
            # self.W[l] = self.W[l] - self.eta*(self.A[l-1].T@self.errors[l] + self.lmbd*self.W[l])
            self.W[l] = self.W[l] - self.eta*(self.A[l-1].T@self.errors[l] + self.lmbd*self.W[l])

            # self.b[l] = self.b[l] - self.eta*( np.ones(self.errors[l].shape[0]).reshape(1,-1)@self.errors[l] + self.lmbd*self.b[l])
            self.b[l] = self.b[l] - self.eta*( np.sum(self.errors[l],axis=0,keepdims=True)  + self.lmbd*self.b[l])

    def train(self):
        indices =np.arange(self.n_inputs)
        m = int(self.n_inputs/self.batch_size)

        for epoch in range(self.epochs):
            for i in range(m):
                batch_indices = np.random.choice(indices, self.batch_size, replace=False)
                X_batch = self.X_data_full[batch_indices]
                y_batch = self.y_data_full[batch_indices]
                self.feedForward(X_batch)
                self.backpropagate(y_batch)

        print(f"Completed training with {self.epochs} epochs")





    def displayNetwork(self,errors=False):

        print("A: " )
        print("-------------------------------")
        for i in range(len(self.A)):
            print(self.A[i].shape)
        print("-------------------------------")
        if(errors == True) :
            print("Errors: " )
            print("-------------------------------")
            for i in range(len(self.errors)):
                print(self.errors[i].shape)
            print("-------------------------------")
            print("batch_size: ", self.batch_size)
