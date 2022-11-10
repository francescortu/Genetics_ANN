# Automatic selection of CNN using genetic algorithm

This repository contains an implementation of the DENSER approach for automatic selection of a Convoluational Neural Network (CNN) architecture. The approach is described in the paper: [DENSER: Automatic Selection of Convolutional Neural Network Architectures](https://arxiv.org/abs/1904.08900).

## DENSER
The DENSER approach is based on the idea of using a grammatic to describe the architecture of a CNN. The grammatic is used to generate a population of CNN architectures. The population is then evaluated using a fitness function. The fitness function is based on the accuracy of the CNN on a validation set. The best performing CNN is then used to generate a new population. This process is repeated until a CNN is found that meets the desired accuracy on the validation set.

### Grammatic


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
│   ├── dsge_level.py
│   ├── evolution.py
│   ├── ga_level.py
│   ├── mutations.py
│   ├── nn_encoding.py
├── tests
│   │   
│   └── tests.py
└── todo.txt
``` 

## TO DO
[]- 


