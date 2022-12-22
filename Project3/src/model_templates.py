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



##========================Keras models=============================
##=================================================================
denseKeras1_20 = models.Sequential(name='denseKeras1_20')
# denseNet1_10_keras.add(layers.Conv2D(1,(3,3),activation='sigmoid',input_shape=X_train.shape[1:]))
# denseNet1_10_keras.add(layers.)
denseKeras1_20.add(layers.Flatten())
denseKeras1_20.add(layers.Dense(128,activation="sigmoid"))
denseKeras1_20.add(layers.Dense(128,activation="sigmoid"))
denseKeras1_20.add(layers.Dense(64,activation="sigmoid"))
denseKeras1_20.add(layers.Dense(10, activation='softmax'))

denseKeras3_128 = models.Sequential(name='denseKeras3_128')
# denseNet1_10_keras.add(layers.Conv2D(1,(3,3),activation='sigmoid',input_shape=X_train.shape[1:]))
# denseNet1_10_keras.add(layers.)
denseKeras3_128.add(layers.Flatten())
denseKeras3_128.add(layers.Dense(128,activation="sigmoid"))
denseKeras3_128.add(layers.Dense(128,activation="sigmoid"))
denseKeras3_128.add(layers.Dense(64,activation="sigmoid"))
denseKeras3_128.add(layers.Dense(10, activation='softmax'))


convKeras = models.Sequential(name='convKeras')
# convKeras.add(layers.Conv2D(3,(3,3),activation='sigmoid',input_shape=X_train.shape[1:]))
convKeras.add(layers.Conv2D(3,(3,3),activation='sigmoid'))
# convNet1_3.add(layers.)
convKeras.add(layers.Flatten())
convKeras.add(layers.Dense(100,activation="sigmoid"))
# denseNet1_10_keras.add(layers.Dense(128,activation="sigmoid"))
# denseNet1_10_keras.add(layers.Dense(64,activation="sigmoid"))
convKeras.add(layers.Dense(10, activation='softmax'))

# ##================Shallow DNN ====================
# amodel = CNN(X_train.shape,name="shallow_dense")
# amodel.addLayer( Flatten() )
# amodel.addLayer( Dense(10,sigmoidL) )
# amodel.addLayer( Dense(10,softmaxL)




#
# leNet = Sequential()
# leNet.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same',
#                activation = 'relu', input_shape = (28,28,1)))
# leNet.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same',
#                activation = 'relu'))
# leNet.add(MaxPooling2D())
# leNet.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',
#                activation = 'relu'))
# leNet.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',
#                activation = 'relu'))
# leNet.add(MaxPooling2D())
# # Fully connected layers
# leNet.add(Flatten())
# leNet.add(Dense(128, activation = 'relu'))
# # Output layer has 10 nodes which is corresponding to the number of classes
# leNet.add(Dense(10, activation = 'softmax'))
#
# opt = SGD(lr = 0.01)
# leNet.compile(optimizer = opt,
#             loss = 'categorical_crossentropy',
#             metrics = ['accuracy'])


 #
 # def define_model():
 #  model = Sequential()
 #
 #  # layer 1
 #  model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same',
 #                   activation = 'relu', input_shape = (28,28,1)))
 #  model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same',
 #                   activation = 'relu'))
 #  model.add(MaxPooling2D())
 #
 #  # layer 2
 #  model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',
 #                   activation = 'relu'))
 #  model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',
 #                   activation = 'relu'))
 #  model.add(MaxPooling2D())
 #
 #  # Fully connected layers
 #  model.add(Flatten())
 #  model.add(Dense(128, activation = 'relu'))
 #
 #  # Output layer has 10 nodes which is corresponding to the number of classes
 #  model.add(Dense(10, activation = 'softmax'))
 #
 #  # Compile model
 #  # In this case, we use the Stochastic Gradient Descent (SGD) method with
 #  learning rate 0.01 for optimizing the loss function
 #  opt = SGD(lr = 0.01)
 #  model.compile(optimizer = opt,
 #                loss = 'categorical_crossentropy',
 #                metrics = ['accuracy'])
 #  return model
