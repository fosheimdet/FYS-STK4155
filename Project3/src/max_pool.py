import numpy as np



class MaxPool():
    def __init__(self, pool_shape, stride, padding="same"):
        self.pool_size = pool_shape[0]
        self.stride = stride
        self.padding = padding #"Same": pad so that every element gets pooled. "valid": discards edges if these can't be included without padding

        self.shape = None
        self.shape_prev = None
        self.pad_y = None #Padding to be used on input.
        self.pad_x = None
        self.A = None
        self.delta = None
        self.max_coord = None  #Matrix containing location of the max values. Same shape as pooled matrix.


    def initialize(self,shape_prev):
        self.shape_prev = shape_prev
        Height_p = shape_prev[0]
        Width_p = shape_prev[1]

        self.pad_y = 0
        self.pad_x = 0
        Height = 1+int((Height_p-self.pool_size+self.pad_y)/self.stride)
        Width=  1+int((Width_p-self.pool_size+self.pad_x)/self.stride)
        ##=======================================================================
        R_y = Height_p - self.pool_size
        R_x = Width_p - self.pool_size

        #Number of times the stride fits completely in R
        nS_y = np.floor(R_y/self.stride)
        nS_x = np.floor(R_x/self.stride)

        remainder_y = R_y - nS_y*self.stride
        remainder_x = R_x - nS_x*self.stride
        if(self.padding=="same"):
            p_y = np.ceil(remainder_y/self.stride)*self.stride - remainder_y
            p_x = np.ceil(remainder_x/self.stride)*self.stride - remainder_x
            #Only add padding if filter/pool only partially sticks out
            if(p_y<self.pool_size):
                self.pad_y = int(p_y)
            if(p_x<self.pool_size):
                self.pad_x = int(p_x)

            #If the filter size is greater than the image, the above calculations don't hold
            if(self.pool_size>Height_p):
                self.pad_y = self.pool_size-Height_p
            if(self.pool_size>Width_p):
                self.pad_x = self.pool_size-Width_p
            Height = 1+int((Height_p-self.pool_size+self.pad_y)/self.stride)
            Width=  1+int((Width_p-self.pool_size+self.pad_x)/self.stride)

        elif(self.padding=="valid"):
            Height = 1+int((Height_p-self.pool_size-remainder_y)/self.stride)
            Width=  1+int((Width_p-self.pool_size-remainder_x)/self.stride)
            if(self.pool_size > Height_p):
                print("Warning: Pool size is greater than height of input. Not supported in 'valid' mode.")
            if(self.pool_size > Width_p):
                print("Warning: Pool siplt.show()ze greater than width of input. Not supported in 'valid mode.")

        ##=======================================================================


        self.shape = (Height,Width)

    def feedForward(self,input):


        n_samples = input.shape[0]

        self.max_coord = np.zeros((n_samples,)+input[0].shape)  #Ensures we remove coordinates from previous forward pass
        input_padded = np.pad(input,((0,0),(0,self.pad_y),(0,self.pad_x)))


        self.A =np.zeros((n_samples,)+self.shape)
        S = self.stride
        p = self.pool_size
        for n in range(n_samples):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    pool = input_padded[n,(i*S):(i*S+p),(j*S):(j*S+p)]
                    self.A[n,i,j] = np.max(pool)

                    pool_coord_flat= np.argmax(pool)
                    pool_coords = np.unravel_index(pool_coord_flat,pool.shape)
                    absolute_coords = (i*S+pool_coords[0],j*S+pool_coords[1])
                    absolute_coord_flat = absolute_coords[0]*input[n,:,:].shape[1]+absolute_coords[1]

                    self.max_coord[n,i,j] = absolute_coord_flat

        return self.A

    def backpropagate(self,input):
        n_samples = input.shape[0]
        self.delta = np.zeros((n_samples,)+self.shape_prev)

        #=========Needs adjustment for multiple channels====================
        for n in range(n_samples):
            for i in range(self.A[0,:,:].shape[0]):
                for j in range(self.A[0,:,:].shape[1]):

                    coord_flat = int(self.max_coord[n,i,j])
                    delta_coords = np.unravel_index(coord_flat,self.delta[n,:,:].shape)

                    #Multiple elements in self.A (pooled layer) may come from the same node in
                    #the input (layer to be pooled) in case the steps result in filter overlap.
                    #We therefore sum errors that stem from the same node in the imput.
                    self.delta[n][delta_coords]+= input[n,i,j]


        return self.delta

    def update(self,*args):
        return None


a = np.arange(0,20)
A = a.reshape(5,4)
A = np.random.randint(0,20,(3,4))
A = A[np.newaxis,:]


pool_shape = (3,3)
stride = 1
print("pool_size: ", pool_shape[0])
print("stride: ", stride)
max = MaxPool(pool_shape,stride,"valid")
max.initialize(A.shape[1:])
output = max.feedForward(A)

max.backpropagate(output)
