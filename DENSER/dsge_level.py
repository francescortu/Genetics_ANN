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
        if self.type == layer_type.POOLING:           #randomly choose a pooling type
            self.param = {"pool_type" : pool(np.random.randint(len(pool))), "kernel_size": np.random.randint(2, 5), "stride": np.random.randint(1, 3), "padding": np.random.randint(0, 2)}
        elif self.type == layer_type.CONV:         #randomly choose a kernel size, stride and padding
            self.param = {'kernel_size': np.random.randint(1, 4), 'stride': np.random.randint(1, 2), 'padding': np.random.randint(1, 2)}
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
        

        if self.M_type == module_types.CLASSIFICATION :
            self.layers.append(Layer(layer_type.LINEAR, c_in = c_in, c_out = c_out))
            self.layers.append(Layer(layer_type.ACTIVATION, c_in = c_out, c_out = c_out))
            
        elif self.M_type == module_types.LAST_LAYER :
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


        
