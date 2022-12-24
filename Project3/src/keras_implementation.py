import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
import time
from scipy import signal
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.metrics import categorical_crossentropy
from keras.optimizers import SGD

from functions import scale_data, accuracy, to_categorical_numpy, cross_corr, pick_sample
from reformat_data import format_mnist8x8,format_mnist,format_svhn
from benchmarking import plot_accuracy_keras, confusion_matrix_keras


def get_data(dataset,plot):
    if(data_name=="MNIST"):
        return format_mnist(plot) #Hand-written digits, 28x28
    elif(data_name=="MNIST8x8"):
        return format_mnist8x8(plot) #Hand-written digits, 8x8
    elif(data_name=="SVHN"):
        return format_svhn(plot) #Housing numbers, 32x32

# data_name = "MNIST8x8"
#data_name="MNIST"
data_name = "MNIST"

data=(X_train,X_test,y_train,y_test)= get_data(data_name,plot=False)
X_train,X_test = X_train/255.0, X_test/255.0 #Normalize pixel values to be between 0 and 1 by dividing by 255.

print(X_train.shape, y_test.shape)

#--------------------------------------------------------------
#--------------------Import models-----------------------------

# from model_templates import LdenseKeras1_10
# from model_templates import LdenseKeras1_20
# from model_templates import LdenseKeras3_128
# from model_templates import LdenseKeras3_20
# from model_templates import LdenseKeras4

# from model_templates import LconvKeras1_3
from model_templates import LconvKeras4_20 #tuned for MNIST
# from model_templates import LconvKeras4_32
# from model_templates import LconvKeras6_32
from model_templates import LconvKerasC6_32 #tuned for SVHN

from model_templates import LleNet2 #tuned for SVHN
#--------------------------------------------------------------


#Pick model (convKerasc6_32>leNet2   convKeras4_20)
model,hyperparams = LconvKeras4_20


epochs,batch_size,eta,lmbd = hyperparams

#Tune hyperparameters here
tune = False
if(tune):
    epochs = 80
    batch_size = 128
    eta = 0.01

#Pick optimizer and cost function
opt =  keras.optimizers.SGD(eta)
model.compile(loss = categorical_crossentropy,
            optimizer = opt,
            metrics = ['accuracy'])

#Perform mini run with a subset of the data   (n_train,n_test)
X_train,X_test,y_train,y_test = pick_sample(data,1000,1000,sample=False)


n_train = X_train.shape[0]
n_batches = n_train/batch_size
stepsInEpoch = int(n_batches/10)



#Extract validation set from training
val_size = 0.1
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size)


#Fit the model
t_start = time.time()
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
                      steps_per_epoch = X_train.shape[0]//batch_size,
                      validation_data = (X_test, y_test),
                      validation_steps = X_test.shape[0]//batch_size, verbose = 1)

t_end = time.time()
t_train = t_end-t_start
print("======================================================================")
print(f"      Completed training with {epochs} epochs in {t_train:.3f} s = {t_train//60:.0f}min,{t_train%60:.0f}sec.")
print("======================================================================")

#Plot the running accuracy during training
plot_accuracy_keras(model,hyperparams,data_name)
# Make prediction and corresponding confusion matrix
y_pred = model.predict(X_test)
confusion_matrix_keras(model,X_test,y_test,hyperparams,data_name)
plt.show()
