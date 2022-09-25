import numpy as np




class FFNN:
    def __init__(self, X, y, nhidden, lengths, act_h_list, act_out_list, derCost, softmaxBool,  hyperparams):


        self.X_data_full = X
        self.y_data_full = y
        self.n_inputs = X.shape[1]
        self.nhidden = nhidden
        self.lengths = lengths #a list containing the number of neurons in each layer
        self.epochs,self.batch_size,self.eta,self.lmbd = hyperparams
        self.act_h, self.derAct_h = act_h_list
        self.act_out, self.derAct_out = act_out_list
        self.derCost = derCost
        self.softmaxBool = softmaxBool



    def initializeNetwork(self):
        self.W = [] #Weights
        self.b = [] #biases
        self.errors_initial = []

        self.W.append(np.array([0]))
        self.b.append(np.array([0]))
        self.errors_initial.append(np.array([0]))

        for i in range(self.nhidden+1):
            self.W.append(np.random.normal(0,1,(self.lengths[i+1],self.lengths[i])))
            self.b.append(np.repeat(0.01,self.lengths[i+1]).reshape(-1,1))
            self.errors_initial.append(np.zeros((self.lengths[i+1], self.n_inputs)))

    def feedForward(self, X):
        self.A = [] #Activations
        #First layer
        self.A.append(X)
        #Hidden layers
        for i in range(self.nhidden):
            self.A.append(self.act_h(self.W[i+1]@self.A[i] + self.b[i+1]))    # def predict(self, X, y):

        #Output layer
        AL = self.act_out(self.W[-1]@self.A[-1] + self.b[-1])
        self.A.append(AL)

        self.output = self.A[-1]

        return AL


    def backpropagate(self,y):

        self.errors = self.errors_initial #Reset errors


        if(self.softmaxBool):
            self.errors[-1] = (self.output-y)
        else:
            # self.errors[-1] = self.output-y
            self.errors[-1] = self.derAct_out(self.output)*self.derCost(self.output, y)

        # self.errors[-1] = deltaL(self.output,y,self.derAct_out,self.derCost)


        #Finding the remaining errors:

        for l in range(self.nhidden, 0, -1):
            self.errors[l] = self.derAct_h(self.A[l])*(self.W[l+1].T@self.errors[l+1])

        for l in range(1,self.nhidden+2):
            self.W[l] = self.W[l] - self.eta*(self.errors[l]@self.A[l-1].T + self.lmbd*self.W[l])


            self.b[l] = self.b[l] - self.eta*self.errors[l]@np.ones(self.errors[l].shape[1]).reshape(-1,1)

    def train(self):
        indices =np.arange(self.n_inputs)

        for epoch in range(self.epochs):
            for i in range(int(self.n_inputs/self.batch_size)):
                batch_indices = np.random.choice(indices, self.batch_size, replace=True)
                X_batch = self.X_data_full[:,batch_indices]
                y_batch = self.y_data_full[:,batch_indices]
                self.feedForward(X_batch)
                self.backpropagate(y_batch)




    def displayNetwork(self,errors=False):
        # for i in range(len(self.A)):
        #     print("{}  {}".format(self.A[i].shape, self.errors[i].shape))

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

# def deltaL(AL,y, derAct_out, derCost):
#     nL = AL.shape[0]
#     n_input = AL.shape[1]
#     deltaL = np.zeros((nL,n_input))
#     for index in range(n_input):
#         M = np.zeros((nL,nL))
#         for c in range(len(y)):
#             for i in range(len(y)):
#                 M[c,i] = derAct_out(i,c,AL[:,index])
#         derC_vec = derCost(AL[:,index], y[:,index])
#         deltaL[:,index] = M@derC_vec
#
#     return deltaL

class FFNN2:
    def __init__(self, X, y, nhidden, lengths, act_h_list, act_out_list, derCost, softmaxBool,  hyperparams):


        self.X_data_full = X
        self.y_data_full = y
        self.n_inputs = X.shape[0]
        self.nhidden = nhidden
        self.lengths = lengths #a list containing the number of neurons in each layer
        self.epochs,self.batch_size,self.eta,self.lmbd = hyperparams
        self.act_h, self.derAct_h = act_h_list
        self.act_out, self.derAct_out = act_out_list
        self.derCost = derCost
        self.softmaxBool = softmaxBool


    def initializeNetwork(self):
        self.W = [] #Weights
        self.b = [] #biases
        self.errors_initial = []

        self.W.append(np.array([0]))
        self.b.append(np.array([0]))
        self.errors_initial.append(np.array([0]))

        for i in range(self.nhidden+1):
            self.W.append(np.random.normal(0,1,(self.lengths[i],self.lengths[i+1])))
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


    def backpropagate(self,y):

        self.errors = self.errors_initial #Reset errors


        if(self.softmaxBool):
            self.errors[-1] = (self.output-y)
        else:
            self.errors[-1] = self.derAct_out(self.output)*self.derCost(self.output, y)

        # self.errors[-1] = deltaL(self.output,y,self.derAct_out,self.derCost)


        #Finding the remaining errors:

        for l in range(self.nhidden, 0, -1):
            self.errors[l] = self.errors[l+1]@self.W[l+1].T*self.derAct_h(self.A[l])


        for l in range(1,self.nhidden+2):
            self.W[l] = self.W[l] - self.eta*self.A[l-1].T@self.errors[l] + self.lmbd*self.W[l]


            self.b[l] = self.b[l] - self.eta*np.ones(self.errors[l].shape[0]).reshape(1,-1)@self.errors[l]

    def train(self):
        indices =np.arange(self.n_inputs)

        for epoch in range(self.epochs):
            for i in range(int(self.n_inputs/self.batch_size)):
                batch_indices = np.random.choice(indices, self.batch_size, replace=True)
                X_batch = self.X_data_full[batch_indices,:]
                y_batch = self.y_data_full[batch_indices,:]
                self.feedForward(X_batch)
                self.backpropagate(y_batch)







    def displayNetwork(self,errors=False):
        # for i in range(len(self.A)):
        #     print("{}  {}".format(self.A[i].shape, self.errors[i].shape))

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




#
# class NeuralNetworkMatrix:
#     def __init__(self, X, y, nhidden, lengths, epochs = 10, batch_size = 20, eta = 0.1):
#
#
#         self.X = X
#         self.y = y
#         self.nhidden = nhidden
#         self.lengths = lengths #a list containing the number of neurons in each layer
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.eta = eta
#
#
#     def initializeNetwork(self):
#         self.W = [] #Weights
#         self.b = [] #biases
#         self.errors_initial = []
#
#         self.W.append(np.array([0]))
#         self.b.append(np.array([0]))
#         self.errors_initial.append(np.array([0]))
#
#         for i in range(self.nhidden+1):
#             self.W.append(np.random.normal(0,1,(self.lengths[i+1],self.lengths[i])))
#             self.b.append(np.repeat(0.01,self.lengths[i+1]).reshape(-1,1))
#             # self.errors_initial.append(np.repeat(0.0,self.lengths[i+1]).reshape(-1,1))
#             self.errors_initial.append(np.zeros((self.lengths[i+1], self.X.shape[0])))
#
#     def feedForward(self, X):
#         self.A = [] #Activations
#         #First layer
#         self.A.append(X)
#         #Hidden layers
#         for i in range(self.nhidden):
#             self.A.append(sigm(self.W[i+1]@self.A[i] + self.b[i+1]))
#         #Output layer
#         AL = self.W[-1]@self.A[-1] + self.b[-1]
#         self.A.append(AL)
#
#         self.output = self.A[-1]
#
#
#     def backpropagate(self, y):
#
#         self.errors = self.errors_initial
#
#         self.errors[-1] = self.output-y
#
#         #Finding the remaining errors:
#
#         for l in range(self.nhidden, 0, -1):
#
#             self.errors[l] = ((self.W[l+1].T)@self.errors[l+1])*self.A[l]*(1-self.A[l])
#
#         for l in range(1,self.nhidden+2):
#             self.W[l] = self.W[l] - self.eta*self.errors[l]@self.A[l-1].T
#             # self.W[l] = self.W[l] - self.eta*np.outer(self.errors[l],self.A[l-1].T)
#
#             self.b[l] = self.b[l] - self.eta*self.errors[l]@np.ones(self.X.shape[1]).reshape(-1,1)
#
#     def displayNetwork(self):
#         # for i in range(len(self.A)):
#         #     print("{}  {}".format(self.A[i].shape, self.errors[i].shape))
#         print("A: " )
#         print("-------------------------------")
#         for i in range(len(self.A)):
#             print(self.A[i].shape)
#         print("-------------------------------")
#         print("Errors: " )
#         print("-------------------------------")
#         for i in range(len(self.errors)):
#             print(self.errors[i].shape)
#         print("-------------------------------")
#
#
# class MLPxor:
#     def __init__(self, X, y, nhidden, lengths, epochs = 10, batch_size = 20, eta = 0.1):
#
#
#         self.X = X
#         self.y = y
#         self.nhidden = nhidden
#         self.lengths = lengths #a list containing the number of neurons in each layer
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.eta = eta
#
#
#     def initializeNetwork(self):
#         self.W = [] #Weights
#         self.b = [] #biases
#         self.errors_initial = []
#
#         self.W.append(0)
#         self.b.append(0)
#         self.errors_initial.append(0)
#
#         for i in range(self.nhidden+1):
#             self.W.append(np.random.normal(0,1,(self.lengths[i+1],self.lengths[i])))
#             self.b.append(np.repeat(0.01,self.lengths[i+1]).reshape(-1,1))
#             # self.errors_initial.append(np.repeat(0.0,self.lengths[i+1]).reshape(-1,1))
#             self.errors_initial.append(np.zeros((self.lengths[i+1], self.X.shape[0])))
#
#     def feedForward(self, X):
#         self.A = [] #Activations
#         #First layer
#         self.A.append(X)
#         #Hidden layers
#         for i in range(self.nhidden+1):
#             self.A.append(sigm(self.W[i+1]@self.A[i] + self.b[i+1]))
#         #Output layer
#         # AL = self.W[-1]@self.A[-1] + self.b[-1]
#         # self.A.append(AL)
#
#         self.output = self.A[-1]
#
#
#     def backpropagate(self):
#
#         self.errors = self.errors_initial
#
#         self.errors[-1] = (self.output-self.y)*self.A[-1]*(1-self.A[-1])
#
#         #Finding the remaining errors:
#
#         for l in range(self.nhidden, 0, -1):
#
#             self.errors[l] = ((self.W[l+1].T)@self.errors[l+1])*self.A[l]*(1-self.A[l])
#
#         for l in range(1,self.nhidden+2):
#             self.W[l] = self.W[l] - self.eta*self.errors[l]@self.A[l-1].T
#
#             self.b[l] = self.b[l] - self.eta*self.errors[l]@np.ones(self.X.shape[1]).reshape(-1,1)
