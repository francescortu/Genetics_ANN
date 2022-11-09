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

from matplotlib import pyplot
from math import cos, sin, atan
import random


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
            if self.type == layer_type.POOLING and self.param["pool_type"] == pool.AVG:
                return utils.compute_output_avgpool2d(input_shape, self.param["kernel_size"], self.param["stride"], self.param["padding"])
            else:
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
        return self.M_type, self.layers

    def print(self, index=None): #print the GA_encoding
        print(f"\n module: {index}")
        print(f"{self.M_type}")
        for i in range(len(self.layers)):
            print(self.layers[i].get())
        print("param: ", self.param)
  

    



    ########################################
    # Plot the neural network architecture
    ########################################

    def draw_features(self, start, length_f, last = None):
        c_in = self.param['input_channels']
        c_out = self.param['output_channels']
        kernel_size = self.layers[0].param['kernel_size']
        for i in range(int(c_out/3)):
            x1 = 0.5 + i*0.2 + start
            x2 = x1 + 5
            x = [x1,x2,x2,x1]
            y1 = -2.5+i*(-0.3)
            y2 = 2.5 - i*0.3
            y = [y1,y1,y2,y2]
            trapezoid = pyplot.Polygon(xy=list(zip(x,y)),  facecolor='#ba5b83', edgecolor='#803655', linewidth=0.8)
            pyplot.gca().add_patch(trapezoid)
        
        
        next = x2 + 9

        if length_f <= 4: font_size = 6 
        else:   font_size = int(32/length_f)
        
        plt.tick_params(axis='x', labelsize=10)
        # get pooling type
        pool_type = str(self.layers[2].param["pool_type"])[5:]
        plt.text(x2+0.5, 5, f'POOL {pool_type}', fontsize=font_size, fontweight='bold',  color='black')

        colors = ['#227046', '#bf5600']

        color = colors[0] if pool_type == "MAX" else colors[1]
            
        plt.annotate('', xy=(next, 0), xycoords='data',
            xytext=(x2+0.6, 0), textcoords='data',
            arrowprops=dict(facecolor=color, edgecolor="none", width=1.2, headwidth=4, headlength=4))
        
        
        # get activation function type
        activation_layer_type = str(self.layers[1].param)[11:]
        plt.text(x1, y1 - 6, f'Conv2d + \n{activation_layer_type}', fontsize=font_size, fontweight='bold',  color='black')
        plt.text(x1, y1 - 10, f'in: {c_in}, out: {c_out}\nkernel: {kernel_size}x{kernel_size}', fontsize=font_size, color='black')

        if last:
            for i in range(10):
                x1 = next + 3
                x2 = x1 + 1
                x = [x1,x2,x2,x1]
                y1 = 10 - i*2
                y2 = y1 - 2
                y = [y1,y1,y2,y2]
                trapezoid = pyplot.Polygon(xy=list(zip(x,y)),  facecolor='#f2d585', edgecolor='#c9b069', linewidth=0.8)
                pyplot.gca().add_patch(trapezoid)

            plt.text(x1-0.5, y1 - 7, 'Flatten\n layer', fontsize=font_size, fontweight='bold',  color='black')
            next = x2 + 9

            self.add_label(0.5,x2, 'Feature extraction', font_size)


        #update the start position
        return next

       

    def draw_classification(self, start, length_c, length_f, index, node_in=None, last = None):
        c_in = self.param['input_channels']
        c_out = self.param['output_channels']
        if index == 0:
            c_in = 10

        circle_radius = 0.8
        node_input = []
        node_output = []
        if node_in is None:
            color = random.choice(['#154e7a', '#3d9dad', '#415fba'])
            for i in range(c_in):
                x, y = start, (circle_radius*5)*(i-c_in/2)
                circle = pyplot.Circle((x,y), radius=circle_radius, facecolor=color, linewidth=1.5, zorder=2)
                pyplot.gca().add_patch(circle)
                node_input.append({'x': x, 'y': y})
        else:
            node_input = node_in


        if index != length_c : # ifwe are not in the last layer
            c_out = int(c_out* 7/30) + 1
            
        color = random.choice(['#154e7a', '#3d9dad', '#415fba'])

        for i in range(c_out):
            x, y = start + 8, (circle_radius*5)*(i-c_out/2)
            circle = pyplot.Circle((x,y), radius= circle_radius,facecolor=color,  linewidth=1.5, zorder=2)
            pyplot.gca().add_patch(circle)
            node_output.append({'x': x, 'y': y})

        # add connections
        for node1 in node_input:
            for node2 in node_output:
                self.line_between_two_nodes(node1, node2) 
        
        # font size
        if length_f <= 4: font_size = 6 
        else:   font_size = int(32/length_f)

        # if first layer add connection between flatten and input nodes
        if index == 0:
            for i,node in enumerate(node_input):
                self.line_between_two_nodes(node, {'x': start - 9, 'y': -4.5 + i})

            end = start + 8*(length_c+1)
            self.add_label(start, end, 'Classification', font_size)


        node_input = node_output
        next = start + 8
        return next, node_input


    def line_between_two_nodes(self, node1, node2):
        line = pyplot.Line2D((node1['x'], node2['x']), (node1['y'], node2['y']), color='#333232', linewidth=0.5, zorder=-1)
        pyplot.gca().add_line(line)

    def add_label(self, x1, x2, name, font_size):
        line = pyplot.Line2D((x1, x2), (-25, -25), color='#333232', linewidth=0.5)
        line1 = pyplot.Line2D((x1, x1), (-25, -22), color='#333232', linewidth=0.5)
        line2 = pyplot.Line2D((x2, x2), (-25, -22), color='#333232', linewidth=0.5)
        pyplot.gca().add_line(line)
        pyplot.gca().add_line(line1)
        pyplot.gca().add_line(line2)
        plt.text(x1 + (x2 -x1)/3 - 1, -27, name,  fontweight='bold', fontsize=font_size,  color='black')