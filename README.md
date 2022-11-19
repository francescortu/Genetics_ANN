# Automatic selection of CNN using genetic algorithm

This repository contains an implementation (in PyTorch) of the DENSER  ( Deep Evolutionary Network Structured Representation) approach for automatic selection of a Convoluational Neural Network (CNN) architecture. The followed approach is further described in this paper: [DENSER: Automatic Selection of Convolutional Neural Network Architectures](https://arxiv.org/abs/1904.08900).

## DENSER

The method proposed in the paper has to do with **Neuroevolution**, a field which deals with the automatic optimisation of ANN architectures and parametrization. 
The DENSER approach, in particular, combines **Genetic Algorithms** (GA) with **Dynamic structured grammatical evolution** (SGDE), which represents the genotype of each individual (a DNN in our case) through the use of a grammar, expressed in backus-naur form (BNF). (DSGE is slightly different from GE and SGE and these differences are explained in detail in the paper)

The grammar is used to dynamically generate a population of CNN architectures. Each individual of the population is then evaluated using a fitness function. As fitness function the **accuracy** of the CNN on a validation set is used. 
Evolution is performed in the following way:
* an initial population (of size $n$) of randomly generated CNNs is created (following the rules of the given grammar)

then, for the number of generations we initially set, we repeat the following steps:

* a fitness score is computed for each individual
* a pair of parents is chosen randomly from the fittest individuals
* through crossover operation we create two new individuals and we choose the longest one
* mutations are applied to the newly generated individual

The last three steps are repeated $n$ times for each generation, in order to obtain again a population of $n$ individuals.

The whole process is repeated until a CNN is found that meets the desired accuracy on the validation set.

### Grammar

The grammar used is almost the same as the one reported in the paper and can be found in 'src/cnn.grammar.txt'.


## Requirements

* numpy  >= 1.21.5
* python  >= 3.8.13
* pytorch >= 1.11.0
* tqdm >= 4.64.0
* matplotlib >= 3.5.2 
* torchvision >= 0.12.0 

## Usage
To run type:
```bash
$ python3 main.py {cifar10 │ MNIST} {pop_size} {num_gen} {batch_size} {subpath-on-you-want-to-save}
```
The programm will print onf best_organisms the best performing CNNs found during evolution.

## Structure of the repository
``` bash
├── data
├── main.py
├── main_test.py
├── README.md
├── results
│   ├── MNIST
│   └── cifar10
│ 
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
|
└── tests
    └── tests.py

``` 



