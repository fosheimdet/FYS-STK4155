# Project 3: Image Classification using Convolutional Neural Networks

We use our own implementation as well as that of keras to compare the performance of convolutional and dense neural networks 
in classifying images of digits from 0 to 9. 

Three data sets are used for this purpose, namely MNIST(8x8) provided by sklearn, MNIST(28x28) provided through keras and 
SVHN(32x32x3). 

The SVHN dataset is found at: http://ufldl.stanford.edu/housenumbers/
train_32x32.mat and test_32x32.mat must be dowloaded and included in the same directory as reformat_data.py
## Files

| File                        | Content                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------|
| our_implementation.py                  | Perform training and testing using our implementation only                    |
| keras_implementation.py                | Perform training and testing using our keras with tensorflow backend           |

|model_templates.py                | Pre-defined models to be used by the two above scripts        |

| neura_network.py           | Contains the class CNN, which constructs sequential model from the provided layer instances    |
| convolutional.py            |Contains a class for convolutional layer    |
| max_pool.py           |Contain a class for max pool layer |
| flatten.py           | Contain a class for flatten layer |
| dense.py           | Contain a class for dense/fully connected layer |
|                                                                   |
| benchmark.py  |  Functions for benchmarking the ours/keras's models      |
| functions.py  |  Various useful functions      |
| activation_functions.py | Various activation functions to be used in the layers of our implementation  
|
| reformat_data.py  | Transforms the data to the format required by our models  |
| finitie_diff.py  | Contains a function for testing our backpropagation implementation     |
  

