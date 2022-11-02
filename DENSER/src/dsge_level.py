import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scripts import utils

import copy


'''

This file contains the functions to handle the modules at the DSGE,
that is to say the layers inside each single module and their compatibility.

'''

MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 3
MIN_STRIDE = 1
MAX_STRIDE = 3
            
#####################
# Layers definition #
#####################

class layer_type(Enum):
    "Layer types for DSGE."
    POOLING = 0
    CONV = 1
    ACTIVATION = 2
    LINEAR = 3

class pool(Enum):
    "Pooling types for DSGE."
    MAX = 0
    AVG = 1

class activation(Enum):
    "Activation types for DSGE."
    RELU = 0
    SIGMOID = 1
    #TANH = 2
    #SOFTMAX = 3

class Layer:
    "Layer class."
    def __init__(self, type=None, c_in = None, c_out = None, param=None):
        if type is None: # Random init, no type specified (could be pooling, conv, activation, linear)
            self.random_init()
        else:
            self.init_form_encoding(type, param)
        self.channels = {'in': c_in, 'out': c_out}
        
        
    def random_init(self):
        self.type = layer_type(np.random.randint(len(layer_type)))  #randomly choose a type
        self.random_init_param()                  #randomly choose the parameters of the type

    def random_init_param(self):
        kernel_size = np.random.randint(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE)
        stride_size = np.random.randint(MIN_STRIDE, MAX_STRIDE)

        if self.type == layer_type.POOLING:           #randomly choose a pooling type
            padding_size = np.random.randint(0,int(kernel_size/2)+1) # pad should be smaller than or equal to half of kernel size
            self.param = {"pool_type" : pool(np.random.randint(len(pool))), "kernel_size": kernel_size, "stride": stride_size, "padding": padding_size}
        elif self.type == layer_type.CONV:         #randomly choose a kernel size, stride and padding
            padding_size = np.random.randint(int(kernel_size/2)+1, kernel_size+1)
            self.param = {'kernel_size': kernel_size, 'stride': stride_size, 'padding': padding_size}
        elif self.type == layer_type.ACTIVATION:   #randomly choose an activation type
            self.param = activation(np.random.randint(len(activation)))
        elif self.type == layer_type.LINEAR:     #linear layer has no parameters
            self.param = None
    
    def init_form_encoding(self, type, param=None):
        self.type = type   #set the type
        if param is None:   #if no parameters are specified, randomly choose them
            self.random_init_param()
        
    def compute_shape(self, input_shape):
        if self.type == layer_type.CONV or self.type == layer_type.POOLING:
            return utils.compute_output_conv2d(input_shape, kernel_size=self.param['kernel_size'], stride=self.param['stride'], padding=self.param['padding'])
        else:
            return input_shape
            
    def fix_channels(self, c_in=None, c_out=None):
        if c_in is not None:
            self.channels['in'] = c_in
            if self.type != layer_type.CONV and self.type != layer_type.LINEAR:
                self.channels['out'] = c_in
        if c_out is not None:
            self.channels['out'] = c_out
            if self.type != layer_type.CONV and self.type != layer_type.LINEAR:
                self.channels['in'] = c_out

    def get(self):  #return the gene
        return self.type, self.param, self.channels


#####################
# Modules definition #
#####################


class module_types(Enum):
    "Layer types for GA."
    FEATURES = 0
    CLASSIFICATION = 1
    LAST_LAYER = 2


class Module:
    "GA_encoding class. The GA_encoding is composed of a list of genes."
    def __init__(self, M_type, c_in = None, c_out = None):
        self.M_type = M_type #set the type
        self.layers = []
        self.param  = {"input_channels": c_in, 'output_channels': c_out}
        

        if self.M_type == module_types.CLASSIFICATION:
            self.layers.append(Layer(layer_type.LINEAR, c_in = c_in, c_out = c_out))
            self.layers.append(Layer(layer_type.ACTIVATION, c_in = c_out, c_out = c_out))
            
        elif self.M_type == module_types.LAST_LAYER:
            self.layers.append(Layer(layer_type.LINEAR, c_in = c_in, c_out = c_out))

        elif self.M_type == module_types.FEATURES:
            self.layers.append(Layer(layer_type.CONV, c_in = c_in, c_out = c_out))
            self.layers.append(Layer(layer_type.ACTIVATION, c_in = c_out, c_out = c_out))
            self.layers.append(Layer(layer_type.POOLING, c_in = c_out, c_out = c_out))

    def len(self):
        return len(self.layers)  
           

    def compute_shape(self, input_shape):
        output_shape = input_shape
        for i in range(self.len()):
            output_shape = self.layers[i].compute_shape(output_shape)
        return output_shape

    def fix_channels(self,c_in=None, c_out=None):
        "fix the channels of the layers"
        if (c_out is not None):
            for i in range(self.len()):
                self.layers[i].fix_channels(c_out=c_out)

            self.param['output_channels'] = c_out
            
        if(c_in is not None):
            self.layers[0].fix_channels(c_in=c_in)
            self.param['input_channels'] = c_in

    def get(self):
        return self.M_type, self.layers, self.param

    def print(self): #print the GA_encoding
        print( self.M_type)
        for i in range(len(self.layers)):
            print( self.layers[i].get())
        print("param: ", self.param)


        
###################
#    MUTATION      #
###################

"""
In DENSER we have three types of mutation at the dsge level:

* grammatical mutation: an expansion possibility is replaced by another one
* integer mutation: an integer block is replaced by another one
* float mutation: instead of randomly generating new values, a gaussian perturbation is applied 

"""

class dsge_mutation_type(Enum):
    GRAMMATICAL = 0
    INTEGER = 1
    #FLOAT = 2

def dsge_mutation(offspring, type = None):
    "Mutation of the DSGE encoding."
    if type is None:
        type = np.random.choice(list(dsge_mutation_type))
        
    if type == dsge_mutation_type.GRAMMATICAL:
        grammatical_mutation(offspring)
    elif type == dsge_mutation_type.INTEGER:
        integer_mutation(offspring)

    #elif mutation == mutation_type.FLOAT:
        #float_mutation(offspring)

    return offspring

def grammatical_mutation(offspring):
    "Grammatical mutation of the DSGE encoding."
    
    #randomly choose a random gene
    gene = np.random.randint(1, offspring._len())
     
    #identify the gene
    gene_type = offspring.GA_encoding(gene).M_type
    
    print("grammatical mutation", gene_type)
    if gene_type == module_types.FEATURES:
        #choose a layer inside the gene
        layer = np.random.randint(0, offspring.features[gene].len())
        #identify the layer
        type = offspring.features[gene].layers[layer].type
        
        new_layer = Layer(type, c_in = offspring.features[gene].param['input_channels'], c_out = offspring.features[gene].param['output_channels'])
        # add the new layer
        #offspring.features[gene].layers[layer] = new_layer


    elif gene_type == module_types.CLASSIFICATION:
        #choose a layer inside the gene
        layer = np.random.randint(0, offspring.classification[gene - offspring.len_features()].len())
        #identify the layer
        type = offspring.classification[gene - offspring.len_features()].layers[layer].type
        
        new_layer = Layer(type, c_in = offspring.classification[gene - offspring.len_features()].param['input_channels'], c_out = offspring.classification[gene - offspring.len_features()].param['output_channels'])
        # add the new layer
        offspring.classification[gene - offspring.len_features()].layers[layer] = new_layer
    
    else:
        #choose a layer inside the gene
        layer = np.random.randint(0, offspring.last_layer[0].len())
        #identify the layer
        type = offspring.last_layer[0].layers[layer].type
        
        new_layer = Layer(type, c_in = offspring.last_layer[0].param['input_channels'], c_out = offspring.last_layer[0].param['output_channels'])
        # add the new layer
        offspring.last_layer[0].layers[layer] = new_layer

    return offspring

def integer_mutation(offspring):
    "Integer mutation of the DSGE encoding."
    
    #randomly choose a random gene
    gene = np.random.randint(1, offspring._len())
    
    #identify the gene
    gene_type = offspring.GA_encoding(gene).M_type
    
    print("integer mutation", gene_type)
    #change expansion rules within the gene by creating a new module
    new_module = Module(gene_type, c_in = offspring.GA_encoding(gene).param['input_channels'], c_out = offspring.GA_encoding(gene).param['output_channels'])
    
    #replace new gene
    if gene_type == module_types.FEATURES:
        #offspring.features[gene] = new_module
        return
    elif gene_type == module_types.CLASSIFICATION:
        offspring.classification[gene - offspring.len_features()] = new_module
    
    return offspring


