import random
import dataloader
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
import fitness_metrics as fm
from time import sleep
from tqdm import tqdm
from datetime import datetime 
N_CLASSES = 10
DEBUG = 0
FOLDER_SAVE = "/home/alexserra98/uni/prog_deep/Genetics_ANN/GA_for_param_optim/model_data" #path where to save intermediate train results
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu") # the device type is automatically chosen





class genNN(nn.Module):

    """
        Generic Neural Network class

        Attributes:
        -----------
            fe: torch.nn.Sequential()
                feature sequential submodule
            c: torch.nn.Sequential()
                classification sequential submodule
            last: torch.nn.Sequential()
                last layer submodule
        Method:
        -----------
            forward(self,x)
                feed NN with input and compute output

    """

    def __init__(self,fe,c,last):
        super(genNN, self).__init__()
        self.fe = fe
        self.c = c
        self.last = last
    def forward(self, x):
        """
            feed NN with input and compute output

            Parameters:
            ------------
                x: torch.Tensor
                 input tensor
            Return:
            -------------
                logits, probs: tuple(torch.Tensor,torch.Tensor)
                             logits and probability of each class

        """
        x = self.fe(x)
        x = torch.flatten(x, 1)
        x = self.c(x)
        logits = self.last(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

# add static method!
def get_act(name):
    """
    Return the activation function from the string passed as input

    Parameters
    ----------
        name : str
                name of the activation function
    Return
    ----------
        activation function : torch obj

    """
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()

# I want them const, maybe some stuff inside is useless
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

def compute_output_avgpool2d(input_shape,out_channel, kernel_size, stride, padding, dilation=1):
    """
        Compute the output shape after a avgpool2d layer.
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
    return (int((input_shape[0] + 2*padding[0] - kernel_size[0])/stride[0] + 1),
            int((input_shape[1] + 2*padding[1] - kernel_size[1])/stride[1] + 1), out_channel)


    

def compute_input_conv2d(output_shape, kernel_size, stride, padding, dilation=1):
    "Compute the input shape of a 2D convolution layer."
    if isinstance(output_shape, tuple) and len(output_shape)==2:
        return (int((output_shape[0] - 1)*stride[0] + dilation[0]*(kernel_size[0]-1) - 2*padding[0] + 1),
            int((output_shape[1] - 1)*stride[1] + dilation[1]*(kernel_size[1]-1) - 2*padding[1] + 1))
    else:
        return int((output_shape - 1)*stride + dilation*(kernel_size-1) - 2*padding + 1)




    

class Net_encoding:
    """
    Assembling the net and the macro rules using phenotype

    Attributes:
    -----------
        features: List
            list of features layers
        classification: List
            list of classification layers
        input_shape = Tuple 3-dim
            channel,width,height of input
        feat_pheno = str
            phenotype of feature block
        class_pheno = str
            phenotype of classification block
    Methods
    -----------
        get_layers(self, phenotype)
            Parses the phenotype corresponding to the layers.
        assemble_block(self, torch_layers, input_size):
            Maps the layers phenotype into a sequential model
        assemble_model(self)
            Merge in a torch module the sequential blocks of feature and classification
        get_learning(self)
            Parses the phenotype corresponding to the learning
        assemble_optimiser(self, model)
            Maps the learning into a torch optimiser            
    """
    def __init__(self, len_features, len_classification, c_in, c_out, input_shape, feat_pheno,class_pheno,learn_pheno):
  
        self.features = []
        self.classification = []
        self.input_shape = input_shape
        self.feat_pheno = feat_pheno
        self.class_pheno = class_pheno
        self.learn_pheno = learn_pheno
        channels = self.init_random_channel(c_in, c_out, len_features + len_classification + 1 )

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
        
    

    def assemble_block(self, torch_layers, input_size):
            """
                Maps the layers phenotype into a sequential model
                Parameters
                ----------
                torch_layers : list
                    output from get_layers
                input_size : tuple
                    network input shape
                Returns
                -------
                model : torch.models.Model
                    torch trainable model
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
                    act_layer = get_act(layer_params['act'][0])
                    
                    # if padding is same we need to compute so that input shape = output shape
                    if layer_params['padding'][0] == 'valid':
                        padding_tmp = [0,0]
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
                        input_l = compute_output_conv2d(input_l,
                                                    int(layer_params['num-filters'][0]), 
                                                    (int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0])), 
                                                    (int(layer_params['stride'][0]), int(layer_params['stride'][0])), 
                                                    padding_tmp, 
                                                    dilation=[1,1]) 
                        if DEBUG==1:
                            print(f'{input_l}')
                    else: 
                        #not sure about it!!
                        #padding = same can't work with striding
                        conv_layer= nn.Conv2d(in_channels = input_l[2], 
                                        out_channels = int(layer_params['num-filters'][0]), # not sure about it 
                                        kernel_size = (int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0])), 
                                        padding= layer_params['padding'][0], 
                                        dilation=1, 
                                        groups=1, 
                                        bias=eval(layer_params['bias'][0]), 
                                        padding_mode='zeros', 
                                        device=None, 
                                        dtype=None)
                        input_l = (int(input_l[0]), 
                                int(input_l[1]),
                                int(layer_params['num-filters'][0]))
                        if DEBUG==1:
                            print(f'{input_l}')


                    layers.extend([conv_layer, act_layer])

                #batch-normalisation
                #fix num_features
                elif layer_type == 'batch-norm':
                    batch_norm = nn.BatchNorm2d(num_features=input_l[2], 
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
                    padding_tmp = [0,0] #needs to be fixed
                    input_l = compute_output_avgpool2d(input_l,
                                                    input_l[2], 
                                                    (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])), 
                                                    (int(layer_params['stride'][0]), int(layer_params['stride'][0])), 
                                                    padding_tmp, 
                                                    dilation=[1,1]) 
                    if DEBUG==1:
                            print(f'{input_l}')
                    layers.append(pool_avg)

                #max pooling layer
                elif layer_type == 'pool-max':

                    pool_max = nn.MaxPool2d(kernel_size = (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])), 
                                            stride = int(layer_params['stride'][0]), 
                                            padding= 0, #this need to be fixed
                                            dilation=1, 
                                            return_indices=False, 
                                            ceil_mode=False)
                    padding_tmp = [0,0] #needs to be fixed
                    input_l = compute_output_conv2d(input_l,
                                                    input_l[2], 
                                                    (int(layer_params['kernel-size'][0]), int(layer_params['kernel-size'][0])), 
                                                    (int(layer_params['stride'][0]), int(layer_params['stride'][0])), 
                                                    padding_tmp, 
                                                    dilation=[1,1]) 
                    if DEBUG==1:
                            print(f'{input_l}')
                    layers.append(pool_max)


                #dropout layer
                elif layer_type == 'dropout':
                    dropout = nn.Dropout2d(p=float(layer_params['rate'][0]), 
                                        inplace=False)

                    layers.append(dropout)


                #fully-connected layer
                elif layer_type == 'fc':
                    fc = nn.Linear(in_features = input_l, 
                                out_features = int(layer_params['num-units'][0]), 
                                bias=eval(layer_params['bias'][0]), 
                                device=None, 
                                dtype=None)
                    act_layer = get_act(layer_params['act'][0])
                    layers.extend([fc,act_layer])
                    input_l = int(layer_params['num-units'][0])
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
    
            
            if DEBUG == 1:
                input_debug =(input_size[2],input_size[0],input_size[1]) #summary() wants channels as first element
                summary(model,input_debug,device ='cpu')

            return (model,input_l)

    def assemble_model(self):
        """
            Merge in a torch module the sequential blocks of feature and classification

            Parameters:
            -----------
                feat_pheno: str
                        feature phenotype
                class_pheno: str
                        classification phenotype 

        """
        self.features = self.get_layers(self.feat_pheno)
        self.classification = self.get_layers(self.class_pheno)
        feat_seq, feat_output = self.assemble_block(self.features, self.input_shape)
        new_input = np.prod(feat_output)
        class_seq, new_input = self.assemble_block(self.classification,new_input)
        last_layer = torch.nn.Sequential(torch.nn.Linear(new_input,N_CLASSES)) #correct input!!!
        newNN = genNN(feat_seq, class_seq,last_layer)
        return newNN

    def get_learning(self):
        """
            Parses the phenotype corresponding to the learning
            Auxiliary function of the assemble_optimiser function
            Parameters
            ----------

            Returns
            -------
            learning_params : dict
                learning parameters
        """

        raw_learning = self.learn_pheno.split(' ')

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


    def assemble_optimiser(self, model):
        """
            Maps the learning into a torch optimiser
            Parameters
            ----------
            model : torch.nn.Model
                   model on which we're going to use the optimizer
            Returns
            -------
            optimiser : torch.optimizers.Optimizer
                torch optimiser that will be later used to train the model
        """

        learning = self.get_learning()
        if learning['learning'] == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(),
                                    lr = float(learning['lr']),
                                    )
                    
        
        elif learning['learning'] == 'gradient-descent':
            return torch.optim.SGD(model.parameters(), 
                                lr= float(learning['lr']), 
                                momentum = float(learning['momentum']), 
                                dampening=0, 
                                weight_decay = float(learning['decay']), 
                                nesterov= bool(learning['nesterov']), 
                                maximize=False,)


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


    def _len(self):
        x =  len(self.features) + len(self.classification) + 1
        return  x 

    def len_features(self):
        x = len(self.features)
        return x
        
    def len_classification(self):
        x = len(self.classification) 
        return x  

    def GA_encoding(self, i):
        "Give the module at position i"
        if i < self.len_features():
            return self.features[i]
        elif i < self.len_features() + self.len_classification():
            return self.classification[i - self.len_features() ]
        elif i == self.len_features() + self.len_classification():
            return self.last_layer[0]
        else:
            return self.last_layer[0]

    def init_random_channel(self, C_in, C_out, len):
        tmp = C_in
        channels = []
        for i in range(len-1):
            out  = np.random.randint(7,30)
            channels.append( (tmp, out ) )
            tmp = out

        channels.append((tmp, C_out)) 
        return channels
    
    def compute_shape_features(self, input_shape = 32):
        "like the forward pass, compute the output shape of the features block"
        output_shape = input_shape
        for i in range(self.len_features()):
            output_shape = self.features[i].compute_shape(output_shape)
        return output_shape

    def get(self):
        return self.GA_encoding
        
    def print(self):
        print("Net encoding len:", self._len())
        for i in range(self._len()):
            print( self.GA_encoding(i).print())

    def print_GAlevel(self):
        "print only if the module is FEATURES or CLASSIFICATION"
        print("Net len:", self._len())
        for i in range(self._len()):
            print("-",i, self.GA_encoding(i).M_type, ' ',self.GA_encoding(i).param['input_channels'], ' ', self.GA_encoding(i).param['output_channels'])


    def fix_channels(self, cut1, cut2):
        "Given a new list of modules between cut1 and cut2, fix the channels of the modules"
        if cut1 != 0:
            
            c_out = self.GA_encoding(cut1-1).param['output_channels']
            c_in = self.GA_encoding(cut1).param['input_channels']
            new = min(c_in, c_out)
            self.GA_encoding(cut1-1).fix_channels(c_out = new)
            self.GA_encoding(cut1).fix_channels(c_in = new)

            c_out = self.GA_encoding(cut2-1).param['output_channels']
            c_in = self.GA_encoding(cut2).param['input_channels']
            new = min(c_in, c_out)
            self.GA_encoding(cut2-1).fix_channels(c_out = new)
            self.GA_encoding(cut2).fix_channels(c_in = new)
            
        # fix in channels of the first classification block
        last_in = (self.compute_shape_features(self.input_shape) ** 2) * self.features[-1].param['output_channels']
        self.classification[0].fix_channels(c_in = last_in)




class Module:
    """
        GA_encoding class, unit of the outer-level genotype
        Attributes
        ----------
            m_type : str
                non-terminal symbol
            min_expansions : int
                minimum expansions of the block
            max_expansions : int
                maximum expansions of the block
            layers : list
                list of layers of the module

        Methods
        ----------
            initialise(grammar, reuse)
                        Randomly creates a module
            len(self)
                Number of layers
            get(self)
                M_type, num of layers
    """

    def __init__(self, m_type, min_expansions, max_expansions):
        
        self.module = m_type
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        #self.levels_back = levels_back
        self.layers = []
        #self.connections = {}        
    
    def initialise(self, grammar, reuse, init_max):
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
        num_expansions = init_max[self.module]

        #Initialise layers
        for idx in range(num_expansions):
            if idx>0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module))

        """"
        Now all the connections are feed-forward, must add skip connections!!!
        #Initialise connections: feed-forward and allowing skip-connections
        self.connections = {}
        for layer_idx in range(num_expansions):
            if layer_idx == 0:
                #the -1 layer is the input
                self.connections[layer_idx] = [-1,]
            else:
                connection_possibilities = list(range(max(0, layer_idx-self.levels_back), layer_idx-1))
                if len(connection_possibilities) < self.levels_back-1:
                    connection_possibilities.append(-1)

                sample_size = random.randint(0, len(connection_possibilities))
                
                self.connections[layer_idx] = [layer_idx-1] 
                if sample_size > 0:
                    self.connections[layer_idx] += random.sample(connection_possibilities, sample_size)
        """

    def len(self):
        return len(self.layers)  
           
    def get(self):
        return self.module, self.layers

    """
    def print(self): #print the GA_encoding?
        print(self.M_type)
        for i in range(len(self.layers)):
            print( self.layers[i].get())
        print("param: ", self.param)
    """

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

        evaluate(phenotype, load_prev_weights, weights_save_path, parent_weights_path,
                 train_time, num_epochs, datagen=None, input_size=(32, 32, 3))
            evaluates the keras model using the keras optimiser
        testing_performance(self, model_path)
            compute testing performance of the model
    """

    def __init__(self, fitness_metric, batch_size, model):
        """
            Creates the Evaluator instance and loads the dataset.
            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        #self.dataset = dataset(batch_size)
        self.fitness_metric = fitness_metric
        self.model = model

    def eval(self,testloader,criterion):
        '''
            Compute testing performance of the model
            Paramaters:
            ------------
                testloader: torch.nn.dataset.Dataloader()
                    dataloader instance of test set
            Return:
                self.model: torch.nn.module()
                    trained model on which we're evaluating performance
                epoch_loss: float
                    loss computed for that specific epoch
        '''
        
        self.model.eval()
        running_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        for data in tqdm(testloader, desc="evaluating"):
            X, y_true = data
            X, y_true = X.to(DEVICE), y_true.to(DEVICE)
            # calculate outputs by running images through the network
            # Forward pass and record loss
            y_hat, _ = self.model(X) 
            loss = criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(testloader.dataset)
        return self.model,epoch_loss


    def train(self,load_prev_weights, weights_save_path, trainloader,validloader, optimizer, batch_size=1, epochs = 1, inspected = 1000,print_every = 1):
        '''
            Train the model given as arg

            Parameters:
            -----------
                load_prev_weights: bool
                    True if we want to employ previously computed weights
                weights_save_path: str
                    path to .pt file 
                trainloader: torch.datasets.DataLoader()
                    the dataloader for the training set
                validloader: torch.datasets.DataLoader()
                    the dataloader for the  test set
                optimizer: torch.nn.Optimize()
                    torch optimizer
                batch_size: int 
                    the batch size used to construct the trainloader
                epochs: int 
                    the number of epochs to train the model
                inspect: int 
                    the number of items to be used for training before printing the loss
                print_every: int
                    how often should we print performance measures about epochs
        '''
        
        os.makedirs(FOLDER_SAVE, exist_ok=True)
        filename = os.path.join(FOLDER_SAVE, "model.pt")

        
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer
        
        self.model.to(DEVICE)
        inspected = inspected
        iterations = int(inspected / batch_size)

        checkpoint_dict = {
            "parameters": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 0,
            "iteration": 0,
            "lr_scheduler": optimizer.state_dict()['param_groups'][0]['lr'] 
            }
        
        

        

        if load_prev_weights:
            model = model.load_state_dict(torch.load(weights_save_path))
        else:
            best_loss = 1e10
            train_losses = []
            valid_losses = []
            for epoch in range(epochs):  # loop over the dataset multiple times

                running_loss = 0
                self.model.train()
                iter = 0
                #------------train model------------
                for data in tqdm(trainloader,desc="training"):
                    try:
                        X,y_true = data
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        X = X.to(DEVICE)
                        y_true = y_true.to(DEVICE)

                        # Forward pass
                        y_hat, _ = self.model(X) 
                        loss = criterion(y_hat, y_true) 
                        running_loss += loss.item() * X.size(0)

                        # calculate outputs by running images through the network
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        #update checkpoint
                        checkpoint_dict["parameters"] = self.model.state_dict()
                        checkpoint_dict["optimizer"] = optimizer.state_dict()
                        checkpoint_dict["epoch"] = epoch
                        checkpoint_dict["iteration"] = iter
                        iter +=1


                    except StopIteration:
                        # save
                        filename = os.path.join(FOLDER_SAVE, "checkpoint.pt")
                        torch.save(checkpoint_dict, filename)
                        print("StopIteration, not enough data")
                #-------------------------------------------------------
     
                epoch_loss = running_loss / len(trainloader.dataset)
                train_losses.append(epoch_loss)
                # validation
                with torch.no_grad():
                    model, valid_loss = self.eval(validloader, criterion)
                    valid_losses.append(valid_loss)

                if epoch % print_every == (print_every - 1):
                    
                    train_acc = fm.get_accuracy(model, trainloader, device=DEVICE)
                    valid_acc = fm.get_accuracy(model, validloader, device=DEVICE)
                        
                    print(f'{datetime.now().time().replace(microsecond=0)} --- '
                        f'Epoch: {epoch}\t'
                        f'Train loss: {epoch_loss:.4f}\t'
                        f'Valid loss: {valid_loss:.4f}\t'
                        f'Train accuracy: {100 * train_acc:.2f}\t'
                        f'Valid accuracy: {100 * valid_acc:.2f}')
                       # save
            filename = os.path.join(FOLDER_SAVE, "checkpoint.pt")
            torch.save(checkpoint_dict, filename)
                
        return self.model,train_losses




"""
Not sure on how to use this one
def evaluate(args):

        Function used to deploy a new process to train a candidate solution.
        Each candidate solution is trained in a separe process to avoid memory problems.
        Parameters
        ----------
        args : tuple
            cnn_eval : Evaluator
                network evaluator
            phenotype : str
                individual phenotype
            load_prev_weights : bool
                resume training from a previous train or not
            weights_save_path : str
                path where to save the model weights after training
            parent_weights_path : str
                path to the weights of the previous training
            train_time : float
                maximum training time
            num_epochs : int
                maximum number of epochs
        Returns
        -------
        score_history : dict
            training data: loss and accuracy
    

    cnn_eval, phenotype, load_prev_weights, weights_save_path, parent_weights_path, train_time, num_epochs, datagen, datagen_test = args

    try:
        return cnn_eval.evaluate(phenotype, load_prev_weights, weights_save_path, parent_weights_path, train_time, num_epochs, datagen, datagen_test)
    except tensorflow.errors.ResourceExhaustedError as e:
        return None
    """


class Individual:
    """
        Candidate solution.
        Attributes
        ----------
        network_structure : list
            ordered list of tuples formated as follows 
            [(non-terminal, min_expansions, max_expansions), ...]
        output_rule : str
            output non-terminal symbol
        macro_rules : list
            list of non-terminals (str) with the marco rules (e.g., learning)
        modules : list
            list of Modules (genotype) of the layers
        output : dict
            output rule genotype
        macro : list
            list of Modules (genotype) for the macro rules
        phenotype : str
            phenotype of the candidate solution
        fitness : float
            fitness value of the candidate solution
        metrics : dict
            training metrics
        num_epochs : int
            number of performed epochs during training
        trainable_parameters : int
            number of trainable parameters of the network
        time : float
            network training time
        current_time : float
            performed network training time
        train_time : float
            maximum training time
        id : int
            individual unique identifier
        Methods
        -------
            initialise(grammar, levels_back, reuse)
                Randomly creates a candidate solution
            decode(grammar)
                Maps the genotype to the phenotype
            evaluate(grammar, cnn_eval, weights_save_path, parent_weights_path='')
                Performs the evaluation of a candidate solution
    """


    def __init__(self, network_structure, macro_rules, ind_id):
        """
            Parameters
            ----------
            network_structure : list
                ordered list of tuples formated as follows 
                [(non-terminal, min_expansions, max_expansions), ...]
            macro_rules : list
                list of non-terminals (str) with the marco rules (e.g., learning)
            output_rule : str
                output non-terminal symbol
            ind_id : int
                individual unique identifier
        """


        self.network_structure = network_structure
        #self.output_rule = output_rule
        self.macro_rules = macro_rules
        self.modules = []
        self.output = None
        self.macro = []
        self.phenotype = None
        self.fitness = None
        self.metrics = None
        self.num_epochs = None
        self.trainable_parameters = None    
        self.time = None    
        self.current_time = 0   
        self.train_time = 0 
        self.id = ind_id    

    def initialise(self, grammar, reuse, init_max):    
        """ 
            Randomly creates a candidate solution   
            Parameters  
            ----------  
            grammar : Grammar   
                grammar instaces that stores the expansion rules    
            levels_back : dict  
                number of previous layers a given layer can receive as input    
            reuse : float   
                likelihood of reusing an existing layer 
            Returns 
            ------- 
            candidate_solution : Individual 
                randomly created candidate solution 
        """ 

        for non_terminal, min_expansions, max_expansions in self.network_structure: 
            new_module = Module(non_terminal, min_expansions, max_expansions)
            new_module.initialise(grammar, reuse, init_max) 

            self.modules.append(new_module) 

        #Initialise output
        #self.output = grammar.initialise(self.output_rule) later we might include to customize ot


        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))
            print(f'self.macro: {self.macro}')

        return self


    def decode(self, grammar):
        """
            Maps the genotype to the phenotype
            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules
            Returns
            -------
            phenotype : str
                phenotype of the individual to be used in the mapping to the keras model.
        """

        phenotype = ''
        offset = 0
        layer_counter = 0
        for module in self.modules:
            offset = layer_counter
            for layer_idx, layer_genotype in enumerate(module.layers):
                layer_counter += 1
                phenotype += ' ' + grammar.decode(module.module, layer_genotype) # ' input:'+",".join(map(str, np.array(module.connections[layer_idx])+offset))

        #phenotype += ' '+grammar.decode(self.output_rule, self.output)+' input:'+str(layer_counter-1) check up

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += ' '+grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype


    def evaluate(self, grammar, cnn_eval, datagen, datagen_test, weights_save_path, parent_weights_path=''):
        """
            Performs the evaluation of a candidate solution
            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules
            cnn_eval : Evaluator
                Evaluator instance used to train the networks
            datagen : keras.preprocessing.image.ImageDataGenerator
                Data augmentation method image data generator
        
            weights_save_path : str
                path where to save the model weights after training
            parent_weights_path : str
                path to the weights of the previous training
            Returns
            -------
            fitness : float
                quality of the candidate solutions
        """

        phenotype = self.decode(grammar)
        start = time()
        pool = Pool(processes=1)

        load_prev_weights = True
        if self.current_time == 0:
            load_prev_weights = False

        train_time = self.train_time - self.current_time

        result = pool.apply_async(evaluate, [(cnn_eval, phenotype, load_prev_weights,\
                                               weights_save_path, parent_weights_path,\
                                               train_time, self.num_epochs, datagen, datagen_test)])

        pool.close()
        pool.join()
        metrics = result.get()

        if metrics is not None:
            self.metrics = metrics
            self.fitness = self.metrics['accuracy_test']
            self.num_epochs += len(self.metrics['val_acc'])
            self.trainable_parameters = self.metrics['trainable_parameters']
            self.current_time += (self.train_time-self.current_time)
        else:
            self.metrics = None
            self.fitness = -1
            self.num_epochs = 0
            self.trainable_parameters = -1
            self.current_time = 0

        self.time = time() - start

        return self.fitness

