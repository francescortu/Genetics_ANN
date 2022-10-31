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
from torchsummary import summary

DEBUG = 0

# add static method!
def get_act(name):
    if name == 'linear':
        return nn.Linear()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()

# I want them const
def compute_output_conv2d(input_shape,out_channel, kernel_size, stride, padding, dilation=1):
    """
        Compute the output shape after a conv layer.
        Parameters
        ----------
        input_shape : tuple
            (width, length, channel)
        out_channel : int
        kernel_size : int
        stride: int
        padding : int
        dilation : int
        
        Return
        ----------
        output : tuple
            (width, length, channel)

    """
    if isinstance(input_shape, tuple) and len(input_shape)==3:
        return (int((input_shape[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1),
            int((input_shape[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1), out_channel)
    elif isinstance(input_shape, tuple) and len(input_shape)==2:
        return (int((input_shape[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1),
            int((input_shape[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1))
    else:
        return int((input_shape + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)



def compute_input_conv2d(output_shape, kernel_size, stride, padding, dilation=1):
    "Compute the input shape of a 2D convolution layer."
    if isinstance(output_shape, tuple) and len(output_shape)==2:
        return (int((output_shape[0] - 1)*stride[0] + dilation[0]*(kernel_size[0]-1) - 2*padding[0] + 1),
            int((output_shape[1] - 1)*stride[1] + dilation[1]*(kernel_size[1]-1) - 2*padding[1] + 1))
    else:
        return int((output_shape - 1)*stride + dilation*(kernel_size-1) - 2*padding + 1)

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
    def assemble_network(self, torch_layers, input_size):
        """
            Maps the layers phenotype into a keras model
            Parameters
            ----------
            keras_layers : list
                output from get_layers
            input_size : tuple
                network input shape
            Returns
            -------
            model : keras.models.Model
                keras trainable model
        """

        #input layer
        input_l = input_size

        #Create layers -- ADD NEW LAYERS HERE
        layers = []
        for layer_type, layer_params in torch_layers:
            if DEBUG == 0:
                print(f'layer_type: {layer_type}')
            #convolutional layer
            if layer_type == 'conv':
                conv_layer= nn.Conv2d(in_channels = input_l[2], 
                                      out_channels = int(layer_params['num-filters'][0]), # not sure about it 
                                      kernel_size = (int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0])), 
                                      stride= (int(layer_params['stride'][0]), int(layer_params['stride'][0])), 
                                      padding= layer_params['padding'][0], 
                                      dilation=1, 
                                      groups=1, 
                                      bias=eval(layer_params['bias'][0]), 
                                      padding_mode='zeros', 
                                      device=None, 
                                      dtype=None)
                
                act_layer = get_act(layer_params['act'][0])
                
                # if padding is same we need to compute so that input shape = output shape
                if layer_params['padding'][0] == 'valid':
                    padding_tmp = [0,0] 
                else: 
                    raise Exception("sorry didn't implemented this case yet")

                input_l = compute_output_conv2d(input_l,
                                                int(layer_params['num-filters'][0]), 
                                                (int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0])), 
                                                (int(layer_params['stride'][0]), int(layer_params['stride'][0])), 
                                                padding_tmp, 
                                                dilation=[1,1])

                layers.extend([conv_layer, act_layer])

            #batch-normalisation
            #fix num_features
            elif layer_type == 'batch-norm':
                batch_norm = nn.BatchNorm2d(num_features, 
                                            eps=1e-05, 
                                            momentum=0.1, 
                                            affine=True, 
                                            track_running_stats=True, 
                                            device=None, 
                                            dtype=None)
                layers.append(batch_norm)

            #average pooling layer
            elif layer_type == 'pool-avg':
                pool_avg = nn.AvgPool2d(kernel_size = (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])),
                                              stride=int(layer_params['stride'][0]), 
                                              padding=0, #need to be fixed 
                                              ceil_mode=False, 
                                              count_include_pad=True, 
                                              divisor_override=None)
                layers.append(pool_avg)

            #max pooling layer
            elif layer_type == 'pool-max':

                pool_max = nn.MaxPool2d(kernel_size = (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])), 
                                        stride = int(layer_params['stride'][0]), 
                                        padding= layer_params['padding'][0], 
                                        dilation=1, 
                                        return_indices=False, 
                                        ceil_mode=False)
                layers.append(pool_max)


            #dropout layer
            elif layer_type == 'dropout':
                dropout = nn.Dropout2d(p=float(layer_params['rate'][0]), 
                                       inplace=False)

                layers.append(dropout)


            """
            #fully-connected layer
            elif layer_type == 'fc':
                fc = nn.Linear(in_features, 
                               out_features, 
                               bias=True, 
                               device=None, 
                               dtype=None)
                layers.append(fc)
            """
            #END ADD NEW LAYERS

        """
        #Connection between layers
        for layer in keras_layers:
            layer[1]['input'] = map(int, layer[1]['input'])


        first_fc = True
        data_layers = []
        invalid_layers = []

        for layer_idx, layer in enumerate(layers):
            
            try:
                if len(keras_layers[layer_idx][1]['input']) == 1:
                    if keras_layers[layer_idx][1]['input'][0] == -1:
                        data_layers.append(layer(inputs))
                    else:
                        if keras_layers[layer_idx][0] == 'fc' and first_fc:
                            first_fc = False
                            flatten = keras.layers.Flatten()(data_layers[keras_layers[layer_idx][1]['input'][0]])
                            data_layers.append(layer(flatten))
                            continue

                        data_layers.append(layer(data_layers[keras_layers[layer_idx][1]['input'][0]]))

                else:
                    #Get minimum shape: when merging layers all the signals are converted to the minimum shape
                    minimum_shape = input_size[0]
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        if input_idx != -1 and input_idx not in invalid_layers:
                            if data_layers[input_idx].shape[-3:][0] < minimum_shape:
                                minimum_shape = int(data_layers[input_idx].shape[-3:][0])

                    #Reshape signals to the same shape
                    merge_signals = []
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        if input_idx == -1:
                            if inputs.shape[-3:][0] > minimum_shape:
                                actual_shape = int(inputs.shape[-3:][0])
                                merge_signals.append(keras.layers.MaxPooling2D(pool_size=(actual_shape-(minimum_shape-1), actual_shape-(minimum_shape-1)), strides=1)(inputs))
                            else:
                                merge_signals.append(inputs)

                        elif input_idx not in invalid_layers:
                            if data_layers[input_idx].shape[-3:][0] > minimum_shape:
                                actual_shape = int(data_layers[input_idx].shape[-3:][0])
                                merge_signals.append(keras.layers.MaxPooling2D(pool_size=(actual_shape-(minimum_shape-1), actual_shape-(minimum_shape-1)), strides=1)(data_layers[input_idx]))
                            else:
                                merge_signals.append(data_layers[input_idx])

                    if len(merge_signals) == 1:
                        merged_signal = merge_signals[0]
                    elif len(merge_signals) > 1:
                        merged_signal = keras.layers.concatenate(merge_signals)
                    else:
                        merged_signal = data_layers[-1]

                    data_layers.append(layer(merged_signal))
            except ValueError as e:
                data_layers.append(data_layers[-1])
                invalid_layers.append(layer_idx)
                if DEBUG:
                    print(keras_layers[layer_idx][0])
                    print(e)

        """
        model =  torch.nn.Sequential(*layers)
  
        
        if DEBUG == 0:
            input_debug =(input_size[2],input_size[0],input_size[1]) #summary() wants channels as first element
            summary(model,input_debug,device ='cpu')

        return model
    


    def assemble_optimiser(self, learning, model):
        """
            Maps the learning into a keras optimiser
            Parameters
            ----------
            learning : dict
                output of get_learning
            Returns
            -------
            optimiser : keras.optimizers.Optimizer
                keras optimiser that will be later used to train the model
        """

        if learning['learning'] == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(),
                                       lr = float(learning['lr']),
                                       rho = float(learning['rho']),
                                       decay = float(learning['decay']))
                    
        
        elif learning['learning'] == 'gradient-descent':
            return torch.optim.SGD(model.parameters(), 
                                   lr= float(learning['lr']), 
                                   momentum = float(learning['momentum']), 
                                   dampening=0, 
                                   weight_decay = float(learning['decay']), 
                                   nesterov= bool(learning['nesterov']), 
                                   maximize=False, 
                                   foreach=None, 
                                   differentiable=False)


        elif learning['learning'] == 'adam':
            return torch.optim.Adam(model.parameters(), 
                                    lr = float(learning['lr']), 
                                    betas = (float(learning['beta1']), float(learning['beta2'])), 
                                    eps=1e-08, 
                                    weight_decay = float(learning['decay']), 
                                    amsgrad=False, 
                                    foreach=None, 
                                    maximize=False, 
                                    capturable=False, 
                                    differentiable=False, 
                                    fused=False)
    



