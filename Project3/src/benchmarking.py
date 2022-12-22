import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions import accuracy,confusion_matrix,partition_data
import seaborn as sns


##==============================================================================
##=========================== Uses our own models ==============================
##==============================================================================
def evaluate_fit(model,data,hyperparams,data_name,val_size=0.1):
    ##=====================================================
    ##==================== Training =======================
    def partition_dataset(X,y,n_part):
        return partition_data(X,n_part),partition_data(y,n_part)
    print("===========================Training======================================")
    #Unpack arguments
    X_train,X_test,y_train,y_test = data
    epochs,M,eta,lmbd = hyperparams

    #Allocate validation set from training
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size)

    #Number of partitions needed
    n_part_train = 1
    n_part_test = 1

    Xp_train,yp_train = partition_dataset(X_train,y_train,n_part_train)
    acc_vals=[]
    acc_trains=[]
    epoch_list=[]
    t_train = 0
    for p in range(n_part_train):
        print(f"---------------------Training on partition {p}-----------------------------")
        #These are lists
        val,train,iters,t_part = model.train(Xp_train[p],yp_train[p],hyperparams,X_val,y_val)
        # val,train,iters = model.train(Xp_train[p],yp_train[p],hyperparams)
        t_train+=t_part
        acc_vals+=val
        acc_trains+=train
        iters =(np.array(iters)+p*epochs)/n_part_train #Convert to actual epoch number
        epoch_list+=iters.tolist()

    acc_vals,acc_trains,epoch_list = np.array(acc_vals),np.array(acc_trains),np.array(epoch_list)
    val_max = np.max(acc_vals)
    train_max = np.max(acc_trains)

    plt.figure()
    plt.title(f"Running accuracy of our '{model.name}' during training on {data_name}.\n" +
    f"$\eta={eta}$, $\lambda={lmbd}$, epochs={epochs}, batch size={M}")
    plt.plot(epoch_list,acc_vals, color = 'darkgreen', label = f"Val. acc. Max: {val_max:.3f}")
    plt.plot(epoch_list,acc_trains, color = 'mediumblue', label = f"Train. acc. Max: {train_max:.3f}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.legend()

    print("============================Testing=======================================")

    ##====================================================
    ##==================== Testing =======================
    # X_test,y_test = X_test[0:n_test],y_test[0:n_test]


    Xp_test,yp_test = partition_dataset(X_test,y_test,n_part_test)
    yp_pred= np.zeros(yp_test.shape)
    for p in range(n_part_test):
        yp_pred[p] = model.predict(Xp_test[p])

    #Reverse partitions
    y_pred = partition_data(yp_pred,n_part_test,reverse=True)


    conf_test = confusion_matrix(y_pred,y_test)
    acc_test = accuracy(y_pred,y_test[0:y_pred.shape[0]])

    write_to_file=True
    if(write_to_file==True):
        #Print accuracy and model used to file
        model_summary = model.summary() #String
        filename = "results/"+model.name+".txt"
        with open(filename,'a') as f:
            f.write("***************************************\n")
            f.write(f"{data_name}\n")
            f.write(f"Test accuracy={acc_test}\n")
            f.write("---------------------------------------\n")
            f.write(f"Training time: {t_train:.3f} s = {t_train//60:.0f}min,{t_train%60:.0f}sec.\n")
            f.write(f"epochs={epochs},batch size={M}\n")
            f.write(f"eta=   {eta},  lambda=     {lmbd}\n")
            f.write("***************************************\n")
            f.write(model_summary+"\n")


    n_categories = y_test.shape[1]
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*conf_test,xticklabels = np.arange(0,n_categories), yticklabels =np.arange(0,n_categories),
             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title(f"Confusion Matrix(%) on {data_name}_test with '{model.name}' using Sigmoid act.\n" +
    f"$\eta={eta}$, $\lambda={lmbd}$, epochs={epochs}, batch size={M}\n Total accuracy: {100*acc_test:.2f}%")
    ax.set_xlabel("Prediction",size=13)
    ax.set_ylabel("Label",size=13)

    # fig, ax = plt.subplots(figsize = (10, 10))
    # sns.heatmap(100*conf_train,xticklabels = np.arange(0,n_categories), yticklabels =np.arange(0,n_categories),
    #         annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    # ax.set_title("Confusion Matrix(%) on MNIST training data using DNN with SGD \n" +
    # f"epochs={epochs}, batch size={M}\n Total accuracy: {100*acc_train:.2f}%")
    # ax.set_xlabel("Prediction")
    # ax.set_ylabel("label")

    #plt.show()
    return acc_test,conf_test



def gridsearch(model,data,hyperparams,data_name,iterable1_prov=[],iterable2_prov=[]):
    ##===================================================================================
    #                               Model selection
    #===================================================================================
    print("===========================Gridsearch======================================")
    X_train,X_test,y_train,y_test = data
    epochs,M,eta,lmbd = hyperparams

    epoch_vals= [50,80,100,150,200,250]      #Epochs
    M_vals = [10,20,50,80,100]             #Batch_sizes
    epoch_vals= [1,3,5,10]
    M_vals = [1,3,5,10]
    # eta_vals data_name="MNIST"

    # lmbd_vals = np.logspace(-6, 0, 7)
    # eta_vals = np.logspace(-2, -1, 2)
    # lmbd_vals = np.logspace(-2, -1, 2)
    # eta_vals = [0.0005,0.001,0.005,0.01,0.02]
    # eta_vals = [0.005,0.01,0.02,0.1,0.2]
    # lmbd_vals = [0.0001,0.0005,0.001,0.005]
    eta_vals = [0.001,0.01,0.05]
    eta_vals = [0.001]
    eta_vals = [1e-4,5e-4,1e-3,2e-3]
    lmbd_vals =[0,1e-4]

    valList = [epoch_vals,  M_vals, eta_vals,  lmbd_vals] #List containing values we want to loop over
    iterableStr = ['epochs','batch size','eta','lambda']
                #       0        1         2       3
    itIndices=[2,3]
    iterable1 = valList[itIndices[0]]
    iterable2 = valList[itIndices[1]]

    #Use the provided values if they exist
    if(len(iterable1_prov)>0 and len(iterable2_prov)>0):
        iterable1 = iterable1_prov
        iterable2 = iterable2_prov

    acc_test = np.zeros((len(iterable1), len(iterable2)))
    acc_train= np.zeros((len(iterable1), len(iterable2)))

    # X_train, y_train = X_train[0:50], y_train[0:50]

    for i, it1 in enumerate(iterable1):
        for j, it2 in enumerate(iterable2):
            #Set the pertinent elements of hyperparams
            hyperparams[itIndices[0]] = it1
            hyperparams[itIndices[1]] = it2
            epochs,M,eta,lmbd =hyperparams
            hyperparams = [epochs,M,eta,lmbd]

            model.initialize_network() #Needed to reset the weights
            model.train(X_train,y_train,hyperparams,verbose=False)

            y_pred = model.predict(X_test)
            y_tilde = model.predict(X_train)

            acc_train[i,j] = accuracy(y_tilde,y_train)
            acc_test[i,j] = accuracy(y_pred,y_test)

    #Create list of indices of the hyperparameters not looped over (to be used in title)
    indices_not = [x for x in range(len(iterableStr))]
    indices_not.remove(itIndices[0])
    indices_not.remove(itIndices[1])
    titleStr=''
    for i in range(len(indices_not)):
        if(i>0):
            titleStr+=", "
        titleStr+=f"{iterableStr[indices_not[i]]}={hyperparams[indices_not[i]]}"


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_test,xticklabels = iterable2, yticklabels =iterable1,
             annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title(f"Test accuracy(%) on {data_name} data using '{model.name}' with Sigmoid\n" +
    titleStr)

    ax.set_xlabel(iterableStr[itIndices[1]])
    ax.set_ylabel(iterableStr[itIndices[0]])


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*acc_train,xticklabels = iterable2, yticklabels =iterable1,
            annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title(f"Training accuracy(%) on {data_name} data using '{model.name}' with Sigmoid\n" +
    titleStr)
    ax.set_xlabel(iterableStr[itIndices[1]])
    ax.set_ylabel(iterableStr[itIndices[0]])
    # plt.show()


##==============================================================================
##================================ Keras =======================================
##==============================================================================

def plot_accuracy_keras(model,hyperparams,data_name):
  epochs,M,eta = hyperparams
  lmbd = 0
  history = model.history

  print(history.history.keys())

  plt.figure()
  plt.title(f"Running accuracy of '{model.name}' during training on {data_name}.\n" +
  f"$\eta={eta}$,$\lambda={lmbd}$, epochs={epochs}, batch size={M}")
  plt.plot(history.history['accuracy'], color =  'mediumblue', label = "Training accuracy")
  plt.plot(history.history['val_accuracy'], color = 'darkgreen', label ="Validation accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
  plt.tight_layout()
  plt.legend()
  plt.draw()
  # plt.show()


def confusion_matrix_keras(model,X_test,y_test,hyperparams,data_name):
    epochs,M,eta = hyperparams
    lmbd = 0

    y_pred = model.predict(X_test)
    conf_test = confusion_matrix(y_pred,y_test)
    acc_test = accuracy(y_pred,y_test[0:y_pred.shape[0]])

    n_categories = y_test.shape[1]
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(100*conf_test,xticklabels = np.arange(0,n_categories), yticklabels =np.arange(0,n_categories),
           annot=True, ax=ax, cmap="rocket", fmt = '.2f',cbar_kws={'format': '%.0f%%'})
    ax.set_title(f"Confusion Matrix(%) on {data_name}_test with '{model.name}' using Sigmoid act.\n" +
    f"$\eta={eta}$, $\lambda={lmbd}$, epochs={epochs}, batch size={M}\n Total accuracy: {100*acc_test:.2f}%")
    ax.set_xlabel("Prediction",size=13)
    ax.set_ylabel("Label",size=13)

    return acc_test,conf_test
