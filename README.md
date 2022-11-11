# Automatic selection of CNN using genetic algorithm

This repository contains an implementation of the DENSER approach for automatic selection of a Convoluational Neural Network (CNN) architecture. The approach is described in the paper: [DENSER: Automatic Selection of Convolutional Neural Network Architectures](https://arxiv.org/abs/1904.08900).

## DENSER
The DENSER approach is based on the idea of using a grammatic to describe the architecture of a CNN. The grammatic is used to generate a population of CNN architectures. The population is then evaluated using a fitness function. The fitness function is based on the accuracy of the CNN on a validation set. The best performing CNN is then used to generate a new population. This process is repeated until a CNN is found that meets the desired accuracy on the validation set.

### Grammatic

## Usage
To run type:
```bash
$ python3 main.py
```
The programm will print onf best_organisms the best performing CNNs found during evolution.

## Structure of the repository
``` bash
├── data
├── images_net
├── main.py
├── main_test.py
├── README.md
├── results.csv
├── scripts
│   ├── dataloader.py
│   ├── train.py
│   └── utils.py
├── src
│   ├── cnn.grammar.txt
│   ├── dsge_level.py
│   ├── evolution.py
│   ├── ga_level.py
│   ├── grammar.py
│   ├── mutations.py
│   ├── nn_encoding.py
├── tests
│   │   
│   └── tests.py
└── todo.txt
``` 

## TO DO
[X] about crossover:
    the module (where the cut is placed) should be selected in tournament
    "we first select a given module from the two parents (selected by tournament), and then we apply a one-point crossover 
    to the module, without changing any of the remaining modules."

[X] add softmax (?) like it is in DENSER

[X] at the moment the mutations at GA level involve only features and classification layers 

* there is a problem with layers sizes which shows only some times (the size of the input becomes too small)
    I've tried to reset the min and max values of stride, kernel and padding, but further checks are needed
    (now an exception is raised when the network is not properly built)

[X] some indexes have to be checked (in initial crossover the choice of features cut start from one, for consistency the same has been done i mutation, but need to be checked, if we want to change it and function fixed_channels will probably need further controls)

[X] control maximum number of feature's and classification's block layers (classification was previously set to 10)

[X] add crossover rate (~70%)

[X] increase train set, decrease test set

[] decide which parts to mantain and which ones to remove from grammar
