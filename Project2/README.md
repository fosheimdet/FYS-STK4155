

# FYS-STK4155 - Project 2

optimizers.py            Script containing various optimization functions used to produce the plots of task a
neural_network.py        Class containing our neural network
franke_regression.py     Script used for evaluating the performance of the NN on Franke data
breast_cancer.py         Script used for 
logistic.py              Our code for logistical regression

functions.py             Various useful functions 
activation_functions.py  Various activation functions used in the neural network
gridsearch.py            A function which performs gridsearch using the NN over two specified parameters

franke_regression.py and breast_cancer.py contain the bool "modelSelection", which can be set to True to perform
a grid search over two specified variable or False to perform model assessment. 


## Files 
| File                        | Content                                                               |
|-----------------------------|-----------------------------------------------------------------------|
| activation.py               | Class with activation functions and derivatives                       |
| analysis.py                 | Class used to setup network and produce figures                       |
| neural_network.py           | Class with Layers, Neural network and Train Network functionality     |
| optimizer.py                | Class used for calculating change in in weights and bias              |
| scores.py                   | Class used for calculating Cost scores and derivatives                |
| test_analysis.py            | Script used to produce result obtained with FFNN binary classification |
| test_logistic_regression.py | Script used to produce results obtained with logistic regression      |


## NN Classification
All results from our classification analysis on the Winsconsing Breast Cancer
data can be reproduced by uncommenting and running the different sections in
test_analysis.py. The analysis was done with our own implementation of a Feed
Forward Neural Network 

## Logistic Regression Classification
To reproduce the results from our classification analysis with Logistic
Regression, uncomment and run the different sections in
test_logistic_regression.py. Our Logistic Regression code is just a subset of our
Neural Network with 0 hidden Layers.   
