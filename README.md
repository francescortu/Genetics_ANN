# Automatic selection of CNN using genetic algorithm

This repository contains an implementation of the DENSER approach for automatic selection of a Convoluational Neural Network (CNN) architecture. The approach is described in the paper: [DENSER: Automatic Selection of Convolutional Neural Network Architectures](https://arxiv.org/abs/1904.08900).

## DENSER
The DENSER approach is based on the idea of using a grammar to describe the architecture of a CNN. The grammar is used to dynamically generate a population of CNN architectures. Each individual of the population is then evaluated using a fitness function. The fitness function is based on the accuracy of the CNN on a validation set. The obtained CNNs are then used to generate a new population; through the use of genetic operators like crossovers and mutations new individuals are created from the previous ones. This process is repeated until a CNN is found that meets the desired accuracy on the validation set.

### Grammatic

## Usage
To run type:
```bash
$ python3 main.py <cifar10 | MNIST> <pop_size> <num_gen> <batch_size>
```
The programm will print onf best_organisms the best performing CNNs found during evolution.

## Structure of the repository
``` bash
├── data
├── images_net
├── main.py
├── main_test.py
├── README.md
|
├── results
|   ├── plots
|   └── all_generations_data.csv
|   
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

