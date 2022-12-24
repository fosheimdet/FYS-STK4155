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



##=================================================================
##======================== SVHN models  ===========================
##=================================================================


##============================================================================
##===================== convKerasC6_32(SVHN) =================================
##============================================================================
pad_mode = "same"
act = "relu"
convKeras12_32= models.Sequential(name='convKeras12_32')
convKeras12_32.add(layers.Conv2D(32,(3,3),activation=act,padding = pad_mode))
convKeras12_32.add(layers.MaxPooling2D((2, 2)))
convKeras12_32.add(layers.Conv2D(32, (3, 3), activation=act,padding = pad_mode))
convKeras12_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
convKeras12_32.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
convKeras12_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
convKeras12_32.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
convKeras12_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
convKeras12_32.add(layers.Conv2D(128, (3, 3), activation=act,padding = pad_mode))
convKeras12_32.add(layers.MaxPooling2D((2, 2),padding="same"))
convKeras12_32.add(layers.Conv2D(128, (3, 3), activation=act,padding = pad_mode))
convKeras12_32.add(layers.MaxPooling2D((2, 2),padding="same"))
convKeras12_32.add(layers.Flatten())
convKeras12_32.add(layers.Dense(128,activation=act))
# convKeras4_20.add(layers.Dense(64,activation="sigmoid"))
convKeras12_32.add(layers.Dense(10, activation='softmax'))
hyperparams=[epochs,batch_size,eta,lmbd]= 20,128,0.01,"?"
LconvKeras12_32 = [convKeras12_32,hyperparams] 
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






# ##============================================================================
# ##===================== convKerasC6_32(SVHN) =================================
# ##============================================================================
# pad_mode = "same"
# act = "relu"
# convKerasC6_32= models.Sequential(name='convKerasC6_32')
# convKerasC6_32.add(layers.Conv2D(32,(3,3),activation=act,padding = pad_mode))
# convKerasC6_32.add(layers.MaxPooling2D((2, 2)))
# convKerasC6_32.add(layers.Conv2D(32, (3, 3), activation=act,padding = pad_mode))
# convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
# convKerasC6_32.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
# convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
# convKerasC6_32.add(layers.Conv2D(64, (3, 3), activation=act,padding = pad_mode))
# convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="valid"))
# convKerasC6_32.add(layers.Conv2D(128, (3, 3), activation=act,padding = pad_mode))
# convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="same"))
# convKerasC6_32.add(layers.Conv2D(128, (3, 3), activation=act,padding = pad_mode))
# convKerasC6_32.add(layers.MaxPooling2D((2, 2),padding="same"))
# convKerasC6_32.add(layers.Flatten())
# convKerasC6_32.add(layers.Dense(128,activation=act))
# # convKeras4_20.add(layers.Dense(64,activation="sigmoid"))
# convKerasC6_32.add(layers.Dense(10, activation='softmax'))
# hyperparams=[epochs,batch_size,eta,lmbd]= 20,128,0.01,"?"
# LconvKerasC6_32 = [convKerasC6_32,hyperparams]
# ##============================================================================
# ##============================================================================
