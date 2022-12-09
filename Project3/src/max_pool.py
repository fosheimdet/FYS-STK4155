import numpy as np



class Max_pool():
    def __init__(self, pool_shape, stride, padding="same"):
        self.pool_size = pool_shape[0]
        self.stride = stride
        self.padding = padding #"Same": pad so that every element gets pooled. "valid": discards edges if these can't be included without padding

        self.shape = None
        self.pad_y = None #Padding to be used on input.
        self.pad_x = None
        self.A = None

    def initialize(self,shape_prev):
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
        ##=======================================================================
        # #Finally calculate the shape of the output from the applied padding
        # Height = 1+int((Height_p-self.pool_size+self.pad_y)/self.stride)
        # Width=  1+int((Width_p-self.pool_size+self.pad_x)/self.stride)

        self.shape = (Height,Width)

    def feedForward(self,input):
        n_samples = input.shape[0]
        input_padded = np.pad(input,((0,0),(0,self.pad_y),(0,self.pad_x)))
        print("input_padded: ")
        print(input_padded)

        self.A =np.zeros((n_samples,)+self.shape)
        S = self.stride
        p = self.pool_size
        print("shape output: ", self.shape)
        for n in range(n_samples):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    pool = input_padded[n,(i*S):(i*S+p),(j*S):(j*S+p)]
                    #print(pool)
                    self.A[n,i,j] = np.max(pool)

                    #print(pool)
                    # self.A[n,i,j] = np.max(input[n,i*self.stride:(i+1)*self.stride,
                    #                                j*self.stride:(j+1)*self.stride])


        return self.A


a = np.arange(0,20)
A = a.reshape(5,4)
A = A[np.newaxis,:]
print("A.shape:", A.shape[1:])
print(A)

pool_shape = (3,3)
stride = 2
print("pool_size: ", pool_shape[0])
print("stride: ", stride)
max = Max_pool(pool_shape,stride,"same")
max.initialize(A.shape[1:])
output = max.feedForward(A)


print("---------------")
print(output)

# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
#
# def plot_image(img: np.array):
#     plt.figure(figsize=(6,6))
#     plt.imshow(img,cmap='gray')
#     plt.show()
#
# def plot_two_images(img1: np.array, img2: np.array):
#     ax = plt.subplots(1,2, figsize(12,6))
#     ax[0].show(img1,cmap='gray')
#     ax[1].show(img2,cmap='gray')
#
# img = Image.open('doggo.jpeg')
# print(img)
# img= ImageOps.grayscale(img)
# print(img)
# #img = img.resize(size=(224,224))
# plot_image(img)




# conv_layer = A
#
# input_height,input_width = conv_layer.shape[0], conv_layer.shape[1]
#
# pool_size = 4
# stride = 2
#
# R = input_height - pool_size
#
# p=0
# if(R%stride !=0):
#     p = pool_size-R%stride
# print("R: ", R)
# print("stride: ", stride)
# print("pool_size: ", pool_size)
# print("p: ",p)
#
# conv_padded = np.pad(conv_layer, ((0,p),(0,p)) )
# print(conv_padded)
#
# R_out = 1+conv_padded.shape[0] -pool_size
# width_out = 1+int(R_out/stride)
#
# max_layer = np.zeros((width_out,width_out))
#
#
# # output_width = np.ceil()
#
# for i in range(width_out):
#     for j in range(width_out):
#         max_layer[i,j] = np.max(conv_padded[i*stride:(i+1)*stride,j*stride:(j+1)*stride])
# print("max_layer: ")
# print(max_layer)
#
# # k,l = 0,0
# # for i in np.arange(input_height,step=stride):
# #     for j in np.arange(input_width,step=stride):
# #         pool = conv_padded[i:i+pool_size,j:j+pool_size]
# #         max_layer[k,l] = np.max(pool)
# #         print(np.max(pool))
# #         l+=1
# #     l=0
# #     k+=1
# # print(max_layer)
