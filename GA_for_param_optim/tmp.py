import random
from multiprocessing import Pool
from time import time
import numpy as np
import os
#from utilities.data import load_dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Evaluator:
    """
        Stores the dataset, maps the phenotype into a trainable model, and
        evaluates it
        Attributes
        ----------
        dataset : dict
            dataset instances and partitions
        fitness_metric : function
            fitness_metric (y_true, y_pred)
            y_pred are the confidences
        Methods
        -------
        get_layers(phenotype)
            parses the phenotype corresponding to the layers
            auxiliary function of the assemble_network function
        get_learning(learning)
            parses the phenotype corresponding to the learning
            auxiliary function of the assemble_optimiser function
        assemble_network(keras_layers, input_size)
            maps the layers phenotype into a keras model
        assemble_optimiser(learning)
            maps the learning into a keras optimiser
        evaluate(phenotype, load_prev_weights, weights_save_path, parent_weights_path,
                 train_time, num_epochs, datagen=None, input_size=(32, 32, 3))
            evaluates the keras model using the keras optimiser
        testing_performance(self, model_path)
            compute testing performance of the model
    """

    def __init__(self):
        """
            Creates the Evaluator instance and loads the dataset.
            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """


    def get_layers(self, phenotype):
        """
            Parses the phenotype corresponding to the layers.
            Auxiliary function of the assemble_network function.
            Parameters
            ----------
            phenotye : str
                individual layers phenotype
            Returns
            -------
            layers : list
                list of tuples (layer_type : str, node properties : dict)
        """

        raw_phenotype = phenotype.split(' ')

        idx = 0
        first = True
        node_type, node_val = raw_phenotype[idx].split(':')
        layers = []

        while idx < len(raw_phenotype):
            if node_type == 'layer':
                if not first:
                    layers.append((layer_type, node_properties))
                else:
                    first = False
                layer_type = node_val
                node_properties = {}
            else:
                node_properties[node_type] = node_val.split(',')

            idx += 1
            if idx < len(raw_phenotype):
                node_type, node_val = raw_phenotype[idx].split(':')

        layers.append((layer_type, node_properties))

        return layers


    def get_learning(self, learning):
        """
            Parses the phenotype corresponding to the learning
            Auxiliary function of the assemble_optimiser function
            Parameters
            ----------
            learning : str
                learning phenotype of the individual
            Returns
            -------
            learning_params : dict
                learning parameters
        """

        raw_learning = learning.split(' ')

        idx = 0
        learning_params = {}
        while idx < len(raw_learning):
            param_name, param_value = raw_learning[idx].split(':')
            learning_params[param_name] = param_value.split(',')
            idx += 1

        for _key_ in sorted(list(learning_params.keys())):
            if len(learning_params[_key_]) == 1:
                try:
                    learning_params[_key_] = eval(learning_params[_key_][0])
                except NameError:
                    learning_params[_key_] = learning_params[_key_][0]

        return learning_params



