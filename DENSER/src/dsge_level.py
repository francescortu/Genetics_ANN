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
import src.grammar as g

import copy

from matplotlib import pyplot
from math import cos, sin, atan
import random


'''

This file contains the functions to handle the modules at the DSGE,
that is to say the layers inside each single module and their compatibility.

'''

PATH = 'src/cnn.grammar.txt'
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 8
MIN_STRIDE = 1
MAX_STRIDE = 3

MAX_LEN_FEATURES = 10
MAX_LEN_CLASSIFICATION = 2 # 2 in DENSER

MAX_LEN_BLOCK_FEATURES = 1 # 2 in DENSER 
MIN_CHANNEL_FEATURES = 9
MAX_CHANNEL_FEATURES = 50
MIN_CHANNEL_CLASSIFICATION = 64
MAX_CHANNEL_CLASSIFICATION = 1024


            
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
    ADP_MAX = 1
    AVG = 2
    ADP_AVG = 3


class activation(Enum):
    "Activation types for DSGE."
    RELU = 0
    SIGMOID = 1
    SOFTMAX = 2

class padding_type(Enum):
    "Convolution types for DSGE."
    PADDING_SAME = "same"
    PADDING_VALID = "valid"

class Layer:
    "Layer class."
    def __init__(self, type=None, c_in = None, c_out = None, param=None):

        self.channels = {'in': c_in, 'out': c_out}
        if type is None: # Random init, no type specified (could be pooling, conv, activation, linear)
            self.random_init()
        else:
            self.init_form_encoding(type, param)
        
        
    def random_init(self):
        self.type = layer_type(np.random.randint(len(layer_type)))  #randomly choose a type
        self.random_init_param()                  #randomly choose the parameters of the type

    def random_init_param(self):
        kernel_size = random.randrange(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE, 2)
        # kernel_size = np.random.randint(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE)
        stride_size = np.random.randint(MIN_STRIDE, MAX_STRIDE)
        padding = np.random.choice(list(padding_type))
        
        if padding.value == "same": # if the padding is same, the stride must be 1; this mode doesn’t support any stride values other than 1
            stride_size = 1
            
        #elif padding.value == "valid": # padding valid is the same as no padding
        padding_value = 0

        # if module is of type pooling randomly choose a pooling type (max, avg) and set the other parmeters
        if self.type == layer_type.POOLING:           
            pool_type = pool(np.random.randint(len(pool)))

            """ if pool_type == pool.MAX:
                if padding.value == "same":
                    padding_value = utils.compute_padding_same_max_pool2d(self.channels["in"], self.channels["out"], kernel_size, stride_size)
            elif pool_type == pool.AVG:
                if padding.value == "same":
                    padding_value = utils.compute_padding_same_avg_pool2d(self.channels["in"], self.channels["out"], kernel_size, stride_size)
             """
            # set parameters
            self.param = {"pool_type" : pool_type, 
                    "kernel_size": kernel_size, 
                    "stride": stride_size, 
                    "padding": padding_value}
                          
        elif self.type == layer_type.CONV:         #randomly choose a kernel size, stride and padding
            self.param = {'kernel_size': kernel_size, 'stride': stride_size, 'padding': padding.value}
        elif self.type == layer_type.ACTIVATION:   #randomly choose an activation type
            self.param = activation(np.random.randint(len(activation)-1))
        elif self.type == layer_type.LINEAR:     #linear layer has no parameters
            self.param = None
    
    def init_form_encoding(self, type, param=None):
        self.type = type   #set the type
        if param is None:   #if no parameters are specified, randomly choose them
            self.random_init_param()
        else:
            self.param = param
        
    def compute_shape(self, input_shape):
        if self.type == layer_type.CONV or self.type == layer_type.POOLING:
            if self.type == layer_type.POOLING and (self.param["pool_type"] == pool.ADP_MAX or (self.param["pool_type"] == pool.ADP_AVG)): # adaptive pooling leaves the input shape unchanged
                return input_shape
            elif self.type == layer_type.POOLING and self.param["pool_type"] == pool.AVG:
                return  utils.compute_output_avgpool2d(input_shape, self.param["kernel_size"], self.param["stride"], self.param["padding"])
            else:
                return utils.compute_output_conv2d(input_shape, kernel_size=self.param['kernel_size'], 
                                                   stride=self.param['stride'], padding=self.param['padding'])
        else:
            return input_shape
            
    def fix_channels(self, c_in=None, c_out=None):
        if (c_in is not None and c_out is not None) or (c_in is not None):
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

        self.grammar = g.Grammar(PATH)

        if self.M_type == module_types.CLASSIFICATION:
            self.layers.append(Layer(layer_type.LINEAR, c_in = c_in, c_out = c_out))
            self.layers.append(Layer(layer_type.ACTIVATION, c_in = c_out, c_out = c_out))
            
        elif self.M_type == module_types.LAST_LAYER:
            self.layers.append(Layer(layer_type.LINEAR, c_in = c_in, c_out = c_out))
            self.layers.append(Layer(layer_type.ACTIVATION, c_in = c_out, c_out = c_out, param = activation.SOFTMAX))

        elif self.M_type == module_types.FEATURES:
            self.initialise('features', MAX_LEN_BLOCK_FEATURES, c_in, c_out)
    
    def check_conv(self):
        for l in self.layers:
            if l.type == layer_type.CONV:
                return True
        return False

    def initialise(self, type, init_max,  c_in, c_out, reuse=0.2):
        """
            Randomly creates a module
            Parameters
            ----------
            grammar : Grammar
                grammar instace that stores the expansion rules
            reuse : float
                likelihood of reusing an existing layer
            Returns
            -------
            score_history : dict
                training data: loss and accuracy
        """
        #for later purpose init_max should be of lenght 3, each a entry for a type of module
        # num_expansions = np.random.randint(2,init_max[type])
        num_expansions = MAX_LEN_BLOCK_FEATURES
        tmp_cin = c_in
        tmp_cout = c_in
        layer_pheno = []
        #Initialise layers
        last_conv = 0

        for idx in range(num_expansions):
            pheno = self.grammar.initialise(type)
            pheno_decoded = self.grammar.decode(type, pheno)
            pheno_dict = self.get_layers(pheno_decoded)
            l_type = self.l_type(pheno_dict[0][0])

            layer_pheno.append(l_type)
            if pheno_dict[0][0] == 'conv':
                last_conv = idx
        
        for idx in range(num_expansions):
            if idx==last_conv:
                tmp_cout = c_out
            elif layer_pheno[idx]!=layer_type.POOLING:
                tmp_cout = np.random.randint(7,30) 

            self.layers.append(Layer(layer_pheno[idx], c_in = tmp_cin , c_out = tmp_cout))
            self.layers.append(Layer(layer_type.ACTIVATION, c_in = tmp_cout , c_out = tmp_cout))
            #setting channels for next iter
            tmp_cin = tmp_cout

    
    def init_random_channel(self, C_in, C_out, len):
        tmp = C_in
        channels = []
        for i in range(len-1):
            out  = np.random.randint(7,30)
            channels.append( (tmp, out ) )
            tmp = out
    
    def l_type(self,l_name):
        if l_name == 'conv':
            return layer_type.CONV
        elif l_name == 'pool-avg' or l_name == 'pool-max':
            return layer_type.POOLING

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

        # if self.M_type == module_types.FEATURES and c_out is not None and c_in is not None:
        #     #find_last_conv
        #     last_conv = -1
        #     for i in range(self.len()):
        #         if self.layers[i].type == layer_type.CONV:
        #             last_conv = i

        #     #fix_channels
        #     tmp_cin = c_in
        #     tmp_cout = c_in
        #     for i in range(self.len()):
        #         tmp_cout = tmp_cin #if it's pooling or act
        #         if i==last_conv:
        #                 tmp_cout = c_out
        #         elif self.layers[i]!=layer_type.POOLING:
        #             tmp_cout = self.layers[i].channels['out']
        #         self.layers[i].fix_channels(c_in=tmp_cin, c_out=tmp_cout)
        #         tmp_cin = tmp_cout
        #     # in case there are only pool and act 
        #     if last_conv == -1:
        #         self.layers.append(Layer(layer_type.CONV,c_in=tmp_cin, c_out=c_out))
        #     self.param['output_channels'] = c_out
        #     self.param['input_channels'] = c_in

        # elif self.M_type == module_types.FEATURES and c_out is not None:
        #     last_conv = -1
        #     for i in range(self.len()):
        #         if self.layers[i].type == layer_type.CONV:
        #             last_conv = i
        #     #if there's no conv layer we can't change c_out
        #     if last_conv == -1:
        #         self.layers.append(Layer(layer_type.CONV,c_in=self.layers[last_conv].channels['in'], c_out=last_conv))
        #     else:
        #         tmp_cin = self.layers[last_conv].channels['in']
        #         tmp_cout = c_out
        #         for i in range(last_conv,self.len()):
        #             self.layers[i].fix_channels(c_in=tmp_cin, c_out=tmp_cout)
        #             tmp_cin = tmp_cout
        #     self.param['output_channels'] = c_out
        # elif self.M_type == module_types.FEATURES and c_in is not None:
        #     tmp_cin = c_in
        #     tmp_cout = self.layers[0].channels['out']
        #     i = 0
        #     while i < self.len() and self.layers[i] != layer_type.CONV:
        #         self.layers[i].fix_channels(c_in=tmp_cin, c_out=tmp_cin)
        #         i +=1
        #     if i < self.len():
        #         self.layers[i].fix_channels(c_in=tmp_cin)
        #         self.param['input_channels'] = c_in     
        #     # case in which all layers are pool or act
        #     if i == self.len():
        #         self.param['input_channels'] = c_in     #
        #         self.layers.append(Layer(layer_type.CONV,c_in=c_in, c_out=self.param['output_channels']))
                
        


    # return module type and layers
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

        text_color = 'white'
        colors = ['#ba5b83', '#bf5600']
        edgecolors = ['#803655', '#8f4101']
        layer_type = self.layers[0].type
        color = colors[0] if layer_type == layer_type.CONV else colors[1]
        edgecolor = edgecolors[0] if layer_type == layer_type.CONV else edgecolors[1]

        plotted = int(c_out/3) if c_out >= 3 else c_out
    
        for i in range(plotted):
            x1 = 0.5 + i*0.2 + start
            x2 = x1 + 5
            x = [x1,x2,x2,x1]
            y1 = -2.5+i*(-0.3)
            y2 = 2.5 - i*0.3
            y = [y1,y1,y2,y2]
            trapezoid = pyplot.Polygon(xy=list(zip(x,y)),  facecolor=color, edgecolor=edgecolor, linewidth=0.8)
            pyplot.gca().add_patch(trapezoid)
        
        
        next = x2 + 9

        if length_f <= 4: font_size = 6 
        else:   font_size = int(32/length_f)
        
        
        activation_layer_type = ''
        #layer_type = str(self.layers[0].type)
        if layer_type == layer_type.CONV:
            # get activation function type
            activation_layer_type = str(self.layers[1].param)[11:]

            plt.text(x1, y1 - 6, f'Conv2d', fontsize=font_size, fontweight='bold',  color=text_color)
            plt.text(x1, y1 - 12, f'in: {c_in}, out: {c_out}\nkernel: {kernel_size}x{kernel_size}', fontsize=font_size, color=text_color)
        elif layer_type == layer_type.POOLING:
            plt.text(x1, y1 - 6, f'Pool2d ', fontsize=font_size, fontweight='bold',  color=text_color)
        
        # add arrow to next module with activation type label if present
        plt.tick_params(axis='x', labelsize=10)
        plt.text(x2+0.5, 5, f'{activation_layer_type}', fontsize=font_size, fontweight='bold',  color=text_color)

        colors = ['#35b3b5', '#3562b5', '#35b575', '#c4c3c2']#'#474747']

        if activation_layer_type:
            color = colors[self.layers[1].param.value]
        else:
            color = colors[3]
        

        plt.annotate('', xy=(next, 0), xycoords='data',
            xytext=(x2+0.6, 0), textcoords='data',
            arrowprops=dict(facecolor=color, edgecolor="none", width=2, headwidth=5, headlength=5))


        if last: # represent flatten layer
            for i in range(10):
                x1 = next + 3
                x2 = x1 + 2
                x = [x1,x2,x2,x1]
                y1 = 10 - i*2
                y2 = y1 - 2
                y = [y1,y1,y2,y2]
                trapezoid = pyplot.Polygon(xy=list(zip(x,y)),  facecolor='#f2d585', edgecolor='#a88d3e', linewidth=1)
                pyplot.gca().add_patch(trapezoid)

            plt.text(x1-2, y1 - 7, 'Flatten\n layer', fontsize=font_size, fontweight='bold',  color=text_color)
            next = x2 + 9

            self.add_label(0.5,x2, 'Feature extraction', font_size)

        #update the start position
        return next

       

    def draw_classification(self, start, length_c, length_f, index, node_in=None, last = None):
        c_in = self.param['input_channels']
        c_out = self.param['output_channels']
        if index == 0:
            c_in = 10

        circle_radius = 1
        # horizontal and vertical distance between nodes
        vertical_space = 6
        horizontal_space = 9 + length_c

        node_input = []
        node_output = []

        
        color = random.choice(['#154e7a', '#3d9dad', '#415fba'])
        # input nodes
        if node_in is None:
            for i in range(c_in):
                x, y = start, (circle_radius*vertical_space)*(i-c_in/2)
                circle = pyplot.Circle((x,y), radius=circle_radius, facecolor=color, linewidth=1.5, zorder=2)
                pyplot.gca().add_patch(circle)
                node_input.append({'x': x, 'y': y})
        else:
            node_input = node_in

        max_output_ch = 13
        if index != length_c : # if we are not in the last layer
            c_out = int(c_out*max_output_ch/MAX_CHANNEL_CLASSIFICATION)  # get a representable number of nodes
            if c_out < 2: c_out = 2

        color = random.choice(['#154e7a', '#3d9dad', '#415fba'])
        for i in range(c_out):
            x, y = start + horizontal_space, (circle_radius*vertical_space)*(i-c_out/2)
            if index == length_c: # last layer has always the same color
                color = '#069655'
            
            circle = pyplot.Circle((x,y), radius= circle_radius, facecolor=color,  linewidth=1.5, zorder=2)
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

            end = start + horizontal_space*(length_c+1)
            self.add_label(start, end, 'Classification', font_size)


        node_input = node_output
        next = start + horizontal_space 
        return next, node_input


    def line_between_two_nodes(self, node1, node2):
        connections_color = '#c4c3c2' #'#333232'  
        line = pyplot.Line2D((node1['x'], node2['x']), (node1['y'], node2['y']), color=connections_color, linewidth=0.5, zorder=-1)
        pyplot.gca().add_line(line)

    def add_label(self, x1, x2, name, font_size):
        text_color = 'white'
        connections_color = '#c4c3c2' #'#333232'  
        y1 = -40
        y2 = y1 + 3
        line = pyplot.Line2D((x1, x2), (y1, y1), color=connections_color, linewidth=0.5)
        line1 = pyplot.Line2D((x1, x1), (y1, y2), color=connections_color, linewidth=0.5)
        line2 = pyplot.Line2D((x2, x2), (y1, y2), color=connections_color, linewidth=0.5)
        pyplot.gca().add_line(line)
        pyplot.gca().add_line(line1)
        pyplot.gca().add_line(line2)
        plt.text(x1 + (x2 -x1)/3 - 1, y1 - 2, name,  fontweight='bold', fontsize=font_size,  color=text_color)