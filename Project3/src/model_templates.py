from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD

from neural_network import CNN
from dense import Dense
from convolutional import Conv
from flatten import Flatten
from max_pool import MaxPool

from tensorflow.keras import layers, models
from tensorflow.keras import datasets as tf_datasets

from activation_functions import*


##==============================================================================
##------------------------------------------------------------------------------
##============================ Our models ======================================
##------------------------------------------------------------------------------
##==============================================================================

##================shallow_dense====================
# shallow_dense = CNN(name="shallow_dense")
# shallow_dense.addLayer( Flatten() )
# shallow_dense.addLayer( Dense(20,sigmoidL) )
# shallow_dense.addLayer( Dense(20,sigmoidL) )
# shallow_dense.addLayer( Dense(10,softmaxL) )
# hyperparams = [100,10,0.01,1e-4]
# # hyperparams = [100,10,0.01,1e-4]
# shallow_dense.set_hyperparams(hyperparams)
# shallow_dense.scheme=""
##================denseNet1_10 ====================
# denseNet1_10 = CNN(name="denseNet1_10")
# denseNet1_10.addLayer( Flatten() )
# denseNet1_10.addLayer( Dense(20,reluL) )
# denseNet1_10.addLayer( Dense(10,softmaxL) )
# hyperparams = [80,10,0.01,1e-4]
# # hyperparams = [100,10,0.01,1e-4]
# denseNet1_10.set_hyperparams(hyperparams)
# denseNet1_10.scheme="Xavier"
##================denseNet1_20 ====================
# denseNet1_20 = CNN(name="denseNet1_20")
# denseNet1_20.addLayer( Flatten() )
# denseNet1_20.addLayer( Dense(20,sigmoidL) )
# denseNet1_20.addLayer( Dense(10,softmaxL) )
# hyperparams = [80,10,0.01,0.1]
# # hyperparams = [100,10,0.01,1e-4]
# denseNet1_20.set_hyperparams(hyperparams)
# denseNet1_20.scheme="Xavier"
##============================================================================
##===================== denseNet2_20(MNIST8x8) ===============================
##============================================================================
denseNet2_20 = CNN(name="denseNet2_20")
denseNet2_20.addLayer( Flatten() )
denseNet2_20.addLayer( Dense(20,sigmoidL) )
denseNet2_20.addLayer( Dense(20,sigmoidL) )
denseNet2_20.addLayer( Dense(10,softmaxL) )
hyperparams = [100,10,0.01,1e-4]
denseNet2_20.set_hyperparams(hyperparams)
denseNet2_20.scheme=""
##============================================================================
##============================================================================
# ##=============Dense Neural Network==============
# act = tanhL
# denseNet3_128 = CNN(name="denseNet3_128")
# denseNet3_128.addLayer( Flatten() )
# denseNet3_128.addLayer( Dense(128,act) )
# denseNet3_128.addLayer( Dense(128,act) )
# denseNet3_128.addLayer( Dense(64,act) )
# denseNet3_128.addLayer( Dense(10,softmaxL) )
# hyperparams = [10,128,0.01,1e-4]
# denseNet3_128.set_hyperparams(hyperparams)
# denseNet3_128.scheme=""
##============================================================================
##===================== denseNet4_300(MNIST) =================================
##============================================================================
act = tanhL
denseNet4_300 = CNN(name="denseNet4_300")
denseNet4_300.addLayer( Flatten() )
denseNet4_300.addLayer( Dense(300,act) )
denseNet4_300.addLayer( Dense(128,act) ) #128
denseNet4_300.addLayer( Dense(128,act) )
denseNet4_300.addLayer( Dense(64,act) )
denseNet4_300.addLayer( Dense(10,softmaxL) )
hyperparams = [15,128,0.01,1e-4]
denseNet4_300.set_hyperparams(hyperparams)
denseNet4_300.scheme=""
##============================================================================
##============================================================================
# ##===============================================
# ##=================conv1=========================
# convNet1_3 = CNN(X_train.shape,name="conv")
# conv.addLayer (Conv(3,(3,3),sigmoidL,"same"))
# # conv1.addLayer( MaxPool((2,2),stride=2, padding="valid") )
# # conv1.addLayer (Conv(6,(3,3),sigmoidL,"same"))
# conv.addLayer( Flatten() )
# conv.addLayer( Dense(30,sigmoidL) )
# conv.addLayer( Dense(128,sigmoidL) )
# conv.addLayer( Dense(10,softmaxL) )
# conv.initialize_weights()
##============================================================================
##===================== convNet1_3(MNIST8x8) =================================
##============================================================================
convNet1_3 = CNN(name="convNet1_3")
convNet1_3.addLayer (Conv(3,(3,3),sigmoidL,padding="same"))
convNet1_3.addLayer( Flatten() )
convNet1_3.addLayer( Dense(20,sigmoidL) )
# convNet1_3.addLayer( Dense(20,sigmoidL) )
convNet1_3.addLayer( Dense(10,softmaxL) )
hyperparams = [200,15,0.001,1e-4]
convNet1_3.set_hyperparams(hyperparams)
convNet1_3.scheme = ""
##============================================================================
##============================================================================


##==============================================================================
##------------------------------------------------------------------------------
##============================ Keras models ====================================
##------------------------------------------------------------------------------
##==============================================================================


##=================================================================
##====================== MNIST8x8 =================================
# denseKeras1_10 = models.Sequential(name='denseKeras1_10')
# denseKeras1_10.add(layers.Flatten())
# denseKeras1_10.add(layers.Dense(10,activation="sigmoid"))
# denseKeras1_10.add(layers.Dense(10, activation='softmax'))
# # epochs,batch_size,eta,lmbd = 100,128,0.01,1e-4 #denseNet1_10
# hyperparams= [epochs,batch_size,eta,lmbd] = 100,128,0.01,1e-4    #denseNet1_20
# LdenseKeras1_10=[denseKeras1_10,hyperparams]
##=================================================================
##======================== MNIST ==================================
# denseKeras1_20 = models.Sequential(name='denseKeras1_20')
# denseKeras1_20.add(layers.Flatten())
# denseKeras1_20.add(layers.Dense(20,activation="sigmoid"))
# denseKeras1_20.add(layers.Dense(10, activation='softmax'))
# hyperparams= [epochs,batch_size,eta,lmbd] = 80,10,0.01,0.1    #denseNet1_20
# LdenseKeras1_20=[denseKeras1_20,hyperparams]
##=================================================================
# denseKeras3_128 = models.Sequential(name='denseKeras3_128')
# # denseNet1_10_keras.add(layers.Conv2D(1,(3,3),activation='sigmoid',input_shape=X_train.shape[1:]))
# # denseNet1_10_keras.add(layers.)
# denseKeras3_128.add(layers.Flatten())
# denseKeras3_128.add(layers.Dense(128,activation="sigmoid"))
# denseKeras3_128.add(layers.Dense(128,activation="sigmoid"))
# denseKeras3_128.add(layers.Dense(64,activation="sigmoid"))
# denseKeras3_128.add(layers.Dense(10, activation='softmax'))
# hyperparams= [epochs,batch_size,eta,lmbd] = 10,128,0.01,"?"    #denseNet1_20
# LdenseKeras3_128=[denseKeras1_20,hyperparams]
##=================================================================
##=================================================================
# convKeras1_3= models.Sequential(name='convKeras1_3')
# convKeras1_3.add(layers.Conv2D(3,(3,3),activation='sigmoid'))
# convKeras1_3.add(layers.Flatten())
# convKeras1_3.add(layers.Dense(20,activation="sigmoid"))
# convKeras1_3.add(layers.Dense(10, activation='softmax'))
# hyperparams=[epochs,batch_size,eta,lmbd]= 10,30,0.01,"?"
# LconvKeras1_3 = [convKeras1_3,hyperparams]

##============================================================================
##===================== convKeras4_20(MNIST) =================================
##============================================================================
convKeras4_20= models.Sequential(name='convKeras4_20')
convKeras4_20.add(layers.Conv2D(20,(3,3),activation='sigmoid'))
convKeras4_20.add(layers.MaxPooling2D((2, 2)))
convKeras4_20.add(layers.Conv2D(20, (3, 3), activation='sigmoid'))
convKeras4_20.add(layers.MaxPooling2D((2, 2),padding="same"))
convKeras4_20.add(layers.Flatten())
convKeras4_20.add(layers.Dense(128,activation="sigmoid"))
convKeras4_20.add(layers.Dense(10, activation='softmax'))
hyperparams=[epochs,batch_size,eta,lmbd]= 10,30,0.01,"?"
LconvKeras4_20 = [convKeras4_20,hyperparams]
##============================================================================
##============================================================================

# convKeras4_32= models.Sequential(name='convKeras4_32')
# convKeras4_32.add(layers.Conv2D(20,(3,3),activation='sigmoid'))
# convKeras4_32.add(layers.MaxPooling2D((2, 2)))
# convKeras4_32.add(layers.Conv2D(20, (3, 3), activation='sigmoid'))
# convKeras4_32.add(layers.MaxPooling2D((2, 2)))
# convKeras4_32.add(layers.Flatten())
# convKeras4_32.add(layers.Dense(128,activation="sigmoid"))
# convKeras4_32.add(layers.Dense(10, activation='softmax'))
# hyperparams=[epochs,batch_size,eta,lmbd]= 10,30,0.01,"?"
# LconvKeras4_32 = [convKeras4_32,hyperparams]
##=================================================================
# convKeras6_32= models.Sequential(name='convKeras6_32')
# convKeras6_32.add(layers.Conv2D(32,(3,3),activation='sigmoid'))
# convKeras6_32.add(layers.MaxPooling2D((2, 2)))
# convKeras6_32.add(layers.Conv2D(32, (3, 3), activation='sigmoid'))
# convKeras6_32.add(layers.MaxPooling2D((2, 2)))
# convKeras6_32.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
# convKeras6_32.add(layers.MaxPooling2D((2, 2)))
# convKeras6_32.add(layers.Flatten())
# convKeras6_32.add(layers.Dense(128,activation="sigmoid"))
# # convKeras4_20.add(layers.Dense(64,activation="sigmoid"))
# convKeras6_32.add(layers.Dense(10, activation='softmax'))
# hyperparams=[epochs,batch_size,eta,lmbd]= 20,128,0.01,"?"
# LconvKeras6_32 = [convKeras6_32,hyperparams]

##=================================================================
##======================== SVHN models  ===========================
##=================================================================


##============================================================================
##===================== convKerasC6_32(SVHN) =================================
##============================================================================
pad_mode = "same"
act = "relu"
convKerasC6_32= models.Sequential(name='convKerasC6_32')
convKerasC6_32.add(layers.Conv2D(32,(3,3),activation=act,padding = pad_mode))
convKerasC6_32.add(layers.MaxPooling2D((2, 2)))
convKerasC6_32.add(layers.Conv2D(32, (3, 3), activation=act,padding = pad_mode))
convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
convKerasC6_32.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
convKerasC6_32.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
convKerasC6_32.add(layers.Conv2D(128, (3, 3), activation=act,padding = pad_mode))
convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="same"))
convKerasC6_32.add(layers.Conv2D(128, (3, 3), activation=act,padding = pad_mode))
convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="same"))
convKerasC6_32.add(layers.Flatten())
convKerasC6_32.add(layers.Dense(128,activation=act))
# convKeras4_20.add(layers.Dense(64,activation="sigmoid"))
convKerasC6_32.add(layers.Dense(10, activation='softmax'))
hyperparams=[epochs,batch_size,eta,lmbd]= 20,128,0.01,"?"
LconvKerasC6_32 = [convKerasC6_32,hyperparams]
##============================================================================
##============================================================================

##=======================LeNet======================================
# pad_mode="same"
# act = "tanh"
# leNet= models.Sequential(name='LeNet')
# leNet.add(layers.Conv2D(6,(5,5),activation=act, padding = pad_mode))
# leNet.add(layers.AveragePooling2D((2, 2)))
# leNet.add(layers.Conv2D(16, (5,5), activation=act,padding = pad_mode))
# leNet.add(layers.AveragePooling2D((2, 2)))
# leNet.add(layers.Conv2D(120, (5,5), activation=act,padding = pad_mode))
#
# leNet.add(layers.Flatten())
# leNet.add(layers.Dense(128, activation = act))
# leNet.add(layers.Dense(10, activation = 'softmax'))
# hyperparams=[epochs,batch_size,eta,lmbd]= 15,128,0.01,"?"
# LleNet = [leNet,hyperparams]
##=================================================================

##============================================================================
##===================== LeNet2(SVHN) ===============================================
##============================================================================
pad_mode="same"
act = "tanh"
leNet2= models.Sequential(name='LeNet2')
leNet2.add(layers.Conv2D(32,(3,3),activation=act, padding = pad_mode))
leNet2.add(layers.MaxPooling2D((2, 2)))
leNet2.add(layers.Conv2D(32, (3, 3), activation=act,padding = pad_mode))
leNet2.add(layers.MaxPooling2D((2, 2)))
leNet2.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
leNet2.add(layers.MaxPooling2D((2, 2),padding="same"))
leNet2.add(layers.Conv2D(120, (3, 3), activation=act,padding = pad_mode))
leNet2.add(layers.MaxPooling2D((2, 2),padding="same"))
leNet2.add(layers.Flatten())
leNet2.add(layers.Dense(128, activation = act))
leNet2.add(layers.Dense(10, activation = 'softmax'))
hyperparams=[epochs,batch_size,eta,lmbd]= 20,32,0.01,"?"
LleNet2 = [leNet2,hyperparams]
###=================================================================
##=================================================================

# act = "tanh"
# denseKeras3_20 = models.Sequential(name='denseKeras3_20')
# denseKeras3_20.add(layers.Flatten())
# denseKeras3_20.add(layers.Dense(20,activation=act))
# denseKeras3_20.add(layers.Dense(65,activation=act))
# denseKeras3_20.add(layers.Dense(25,activation=act))
# denseKeras3_20.add(layers.Dense(10, activation='softmax'))
# hyperparams= [epochs,batch_size,eta,lmbd] = 10,128,0.01,"?"    #denseNet1_20
# LdenseKeras3_20=[denseKeras3_20,hyperparams]
# ##=================================================================
# act = "sigmoid"
# denseKeras4 = models.Sequential(name='denseKeras4')
# denseKeras4.add(layers.Flatten())
# denseKeras4.add(layers.Dense(1000,activation=act))
# denseKeras4.add(layers.Dense(1000,activation=act))
# denseKeras4.add(layers.Dense(128,activation=act))
# denseKeras4.add(layers.Dense(128,activation=act))
# denseKeras4.add(layers.Dense(64,activation=act))
# denseKeras4.add(layers.Dense(10, activation='softmax'))
# hyperparams= [epochs,batch_size,eta,lmbd] = 20,32,0.01,"?"    #denseNet1_20
# LdenseKeras4=[denseKeras4,hyperparams]



# convKeras = models.Sequential(name='convKeras')
# # convKeras.add(layers.Conv2D(3,(3,3),activation='sigmoid',input_shape=X_train.shape[1:]))
# convKeras.add(layers.Conv2D(3,(3,3),activation='sigmoid'))
# # convNet1_3.add(layers.)
# convKeras.add(layers.Flatten())
# convKeras.add(layers.Dense(100,activation="sigmoid"))
# # denseNet1_10_keras.add(layers.Dense(128,activation="sigmoid"))
# # denseNet1_10_keras.add(layers.Dense(64,activation="sigmoid"))
# convKeras.add(layers.Dense(10, activation='softmax'))
