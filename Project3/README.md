# Project 3: Image Classification using Convolutional Neural Networks

We use our own implementation as well as that of keras to compare the performance of convolutional and dense neural networks 
in classifying images of digits from 0 to 9. 

Three data sets are used for this purpose, namely MNIST(8x8) provided by sklearn, MNIST(28x28) provided through keras and 
SVHN(32x32x3). 

The SVHN (Street View Housing Numbers) dataset is found at: http://ufldl.stanford.edu/housenumbers/

train_32x32.mat and test_32x32.mat must be dowloaded and included in the same directory as reformat_data.py

The scipts 
"our_implementation.py" and 
"keras_implementation.py" 
perform training and evaluation of prespecified models from our own implementation or that of keras. The data set is chosen here and hyperparameters can be adjusted. To perform a quick test that everything is working, the function "get_sample()" picks a random sample of specified size from the full data set to use in its stead. Its optional parameter must be set to "True" this test-run to occur. 

The aformentioned prespecified models are located in "model_templates.py" and their architecture have been tuned to specific data sets.  
Their tuned hyperparameters are also set here, but can be altered elsewhere through "model.sethypererparams()".

We include a commented-out example on how to design and train a model using our implementation in "our_implementation.py". 
This is done in a similar way to Keras in order to hopefully provide some familiarity.  

The files below are ordered roughly based on their place in the hierarchy of the program. 

## Files

| File                        | Content                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------|
| our_implementation.py                  | Perform training and testing using our implementation only                    |
| keras_implementation.py                | Perform training and testing using keras with tensorflow backend           |
|                                                                                                         |
|                                                                                                         |
| model_templates.py                | Pre-defined models to be used by the two above scripts        |
| benchmark.py  |  Functions for benchmarking our/keras's models      |
| reformat_data.py  | Transforms the data to the format required by the models  |
|                                                                                                               |
|                                                                                                         |
| neura_network.py           | Contains the class CNN, which constructs a feed-forward/sequential model from layer instances    |
| convolutional.py            |Contains a class for convolutional layer    |
| max_pool.py           |Contain a class for max pool layer |
| flatten.py           | Contain a class for flatten layer |
| dense.py           | Contain a class for dense/fully connected layer |
|                                                                                                         |
|                                                                   |
| functions.py  |  Various useful functions      |
| activation_functions.py | Various activation functions to be used in the layers of our implementation  
|                                                                                                             |
| finitie_diff.py  | Contains a function for testing our backpropagation implementation     |
  

