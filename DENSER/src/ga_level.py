from src.dsge_level import *

'''

This file contains all the functions which are used to handle the GA level.

* It deals with the modules at the GA level and the crossover operations to 
    obtain the new offspring from two parents.

* It also contains the functions to mutate the offspring at the GA level, that is to 
    say those operations which manipulate the network structure

'''
MAX_LEN_FEATURES = 10
MAX_LEN_CLASSIFICATION = 3 # 2 in DENSER
LAST_LAYER_SIZE = 1

class Net_encoding:
    "Describe the encoding of a network."
    def __init__(self, len_features, len_classification, c_in, c_out, input_shape):
  
        self.features = []
        self.classification = []
        self.last_layer = []
        self.input_shape = input_shape
        channels = self.init_random_channel(c_in, c_out, len_features + len_classification + 1 )
        
        if len_features > MAX_LEN_FEATURES:
            len_features = MAX_LEN_FEATURES
        if len_classification > MAX_LEN_CLASSIFICATION:
            len_classification = MAX_LEN_CLASSIFICATION
        
        # add features blocks
        for i in range(len_features):
            self.features.append(Module(module_types.FEATURES, c_in = channels[i][0], c_out = channels[i][1]))
            
        k = len_features
        # set the input channels of the classification block: the flatten output of the features block
        channels[k] = ((self.compute_shape_features(self.input_shape)**2) * channels[k-1][1], channels[k][1])  

        # add classification blocks
        for i in range(len_classification):
            self.classification.append(Module(module_types.CLASSIFICATION,  c_in = channels[k+i][0], c_out = channels[k+i][1]))

        # add last block
        self.last_layer.append(Module(module_types.LAST_LAYER,  c_in = channels[self.len_classification() + k][0], c_out = c_out))
        
        self.param = {'input_channels': c_in,'output_channels': c_out}

    # return the length of the encoding:
    # the number of features block and classification block + one last block
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



##############################################
# CROSSOVER
##############################################

class cross_type(Enum):
    "crossover typology"
    ONE_POINT = 0
    BIT_MASK = 1


def GA_crossover(parent1, parent2, type = None):
    "randomly choose the crossover type"
    if type == None:
        type = np.random.choice(list(cross_type))
    if type == cross_type.ONE_POINT:
        return GA_one_point(parent1, parent2)
    else:
        return GA_bit_mask(parent1, parent2)

def GA_bit_mask(parent1, parent2):
        "Crossover with bit mask"
        N = len(module_types)
        mask = np.zeros(N, dtype=int)
        K = np.random.randint(0, N -1)
        mask[:K]  = 1
        np.random.shuffle(mask)
        mask1 = mask
        mask2 = 1 - mask1
        p = [parent1, parent2]
      
        child1 = Net_encoding(p[mask1[0]].len_features(), p[mask1[1]].len_classification(), p[mask1[0]].param['input_channels'], p[mask1[2]].param['output_channels'], p[mask1[0]].input_shape)
        child2 = Net_encoding(p[mask2[0]].len_features(), p[mask2[1]].len_classification(), p[mask2[0]].param['input_channels'], p[mask2[2]].param['output_channels'], p[mask2[0]].input_shape)
        # copy features
        child1.features = copy.deepcopy(p[mask1[0]].features)
        child2.features = copy.deepcopy(p[mask2[0]].features)
        # copy classification
        child1.classification = copy.deepcopy(p[mask1[1]].classification)
        child2.classification = copy.deepcopy(p[mask2[1]].classification)
         # copy last layer
        child1.last_layer = copy.deepcopy(p[mask1[2]].last_layer)
        child2.last_layer = copy.deepcopy(p[mask2[2]].last_layer)
        
        # fix channels
        child1.fix_channels(child1.len_features(), child1.len_features())
        child2.fix_channels(child2.len_features(), child2.len_features())
        
        child1.fix_channels(child1.len_features() + child1.len_classification(), child1.len_features() + child1.len_classification())
        child2.fix_channels(child2.len_features() + child2.len_classification(), child2.len_features() + child2.len_classification())
       
        return child1, child2
        
def GA_one_point(parent1, parent2):
        "cut the parent1 and parent2 at random position and swap the two parts"
        #find cutting point
        cut_parent1 = np.random.randint(1, parent1._len()-1)
        
        #identify the type of the cut
        cut1_type = parent1.GA_encoding(cut_parent1).M_type

        #find a cut on the same module also on parent2
        if cut1_type == module_types.FEATURES:
            cut_parent2 = np.random.randint(1, parent2.len_features()-1)
        elif cut1_type == module_types.CLASSIFICATION:
            cut_parent2 = np.random.randint(parent2.len_features(), parent2.len_features() + parent2.len_classification())
        
        if cut1_type == module_types.FEATURES:
            aux1 = copy.deepcopy(parent1.features[cut_parent1:])
            aux2 = copy.deepcopy(parent2.features[cut_parent2:])
   
            parent1.features = copy.deepcopy(parent1.features[:cut_parent1])
            parent2.features = copy.deepcopy(parent2.features[:cut_parent2])
            
            parent1.features.extend(aux2)
            parent2.features.extend(aux1)
            parent1.fix_channels(cut_parent1, parent1.len_features())
            parent2.fix_channels(cut_parent2, parent2.len_features())

            return parent1, parent2

        elif cut1_type == module_types.CLASSIFICATION:
            aux1 = copy.deepcopy(parent1.classification[cut_parent1 - parent1.len_features():])
            aux2 = copy.deepcopy(parent2.classification[cut_parent2 - parent2.len_features():])
   
            parent1.classification = copy.deepcopy(parent1.classification[:cut_parent1 - parent1.len_features()])
            parent2.classification = copy.deepcopy(parent2.classification[:cut_parent2 - parent2.len_features()])
            
            parent1.classification.extend(aux2)
            parent2.classification.extend(aux1)
            parent1.fix_channels(cut_parent1, parent1.len_features() + parent1.len_classification())
            parent2.fix_channels(cut_parent2, parent2.len_features() + parent2.len_classification())
            return parent1, parent2

        return parent1, parent2


##################################
# MUTATION
##################################


class ga_mutation_type(Enum):
    ADDITION = 0
    REPLACE = 1
    REMOVAL = 2

def choose_cut(netcode):
    "choose a random cut"
    cut = np.random.randint(1, netcode._len()-1)
    #identify the type of the cut
    cut_type = netcode.GA_encoding(cut).M_type

    #control we have not reached the maximum number of layers per module
    if cut_type == module_types.FEATURES and netcode.len_features() > MAX_LEN_FEATURES :
        if netcode.len_classification() < MAX_LEN_CLASSIFICATION:
            cut = np.random.randint(netcode.len_features(), netcode._len()-1) # if the max number of features is reached, the cut is chosen on classification module
        else:
            cut = None
    elif cut_type == module_types.CLASSIFICATION and netcode.len_classification() > MAX_LEN_CLASSIFICATION:
        if netcode.len_features() < MAX_LEN_FEATURES:
            cut = np.random.randint(1, netcode.len_features())
        else:
            cut = None

    return cut

def GA_mutation(offspring, type=None):
    "randomly choose the mutation type"
    if type == None:
        type = np.random.choice(list(ga_mutation_type))
    print("GA mutation type ", type)
    if type == ga_mutation_type.ADDITION:
        cut = choose_cut(offspring)
        if cut is not None:
            c_in =  np.random.randint(7,30) 
            c_out =  np.random.randint(7,30)
            cut_type = offspring.GA_encoding(cut).M_type
            if cut_type == module_types.FEATURES:
                module = Module(module_types.FEATURES, c_in = c_in, c_out = c_out)
            elif cut_type == module_types.CLASSIFICATION:
                module = Module(module_types.CLASSIFICATION, c_in = c_in, c_out = c_out)

            return GA_add(offspring, cut, module)
        else:
            return offspring
    elif type == ga_mutation_type.REPLACE:
        return GA_replace(offspring)
    else:
        return GA_remove(offspring)


def GA_add(offspring, cut, module):
    
    #identify the type of the cut
    cut_type = offspring.GA_encoding(cut).M_type

    # add control to check if we have reached the maximum number of modules
    if cut_type == module_types.FEATURES:
        tmp = copy.deepcopy(offspring.features[cut:])
        offspring.features = copy.deepcopy(offspring.features[:cut + 1])

        # add the module
        offspring.features[cut] = module
        offspring.features.extend(tmp) # add the rest of the modules

        #fix channels before and after the cut
        offspring.fix_channels(cut, cut+1) 
        return offspring

    elif cut_type == module_types.CLASSIFICATION:
        tmp = copy.deepcopy(offspring.classification[cut - offspring.len_features():])
        offspring.classification = copy.deepcopy(offspring.classification[:cut - offspring.len_features() + 1])
        
        # add the module
        offspring.classification[cut- offspring.len_features()] = module
        offspring.classification.extend(tmp) # add the rest of the modules
        #fix channels before and after the cut        
        offspring.fix_channels(cut, cut+1)
        return offspring

    return offspring

def GA_replace(offspring):
    "replace a module at random position"
    #find the position of the layer we want to copy
    cut1 = choose_cut(offspring) # choose_cut function controls we have not reached the limit of layers per module
    
    if cut1 is not None:
        #identify the type of the cut
        cut_type = offspring.GA_encoding(cut1).M_type

        # now find where we want to relocate (add) it
        if cut_type == module_types.FEATURES:
            cut2 = np.random.randint(1, offspring.len_features()-1)
        elif cut_type == module_types.CLASSIFICATION:
            cut2 = np.random.randint(offspring.len_features(), offspring.len_features() + offspring.len_classification())
        
        # The copy must be done by reference
        module = offspring.GA_encoding(cut1) # determine the module to copy
        GA_add(offspring, cut2, module)


    return offspring


def GA_remove(offspring):
    "remove a module at random position"
    #find cutting point
    cut = np.random.randint(1, offspring._len()-1)
    
    #identify the type of the cut
    cut_type = offspring.GA_encoding(cut).M_type

    if cut_type == module_types.FEATURES and offspring.len_features() <= 1: # if cut is of type features and there is only one module, we cannot remove it and we change it
        if offspring.len_classification() > 1:
            cut = np.random.randint(offspring.len_features(), offspring._len()-1) 
        else:
            cut = None
    elif cut_type == module_types.CLASSIFICATION and offspring.len_classification() <=1: # if cut is of type classification and there is only one module, we cannot remove it and we change it
        if offspring.len_features() > 2: # this has to be updated
            cut = np.random.randint(1, offspring.len_features()-1)
        else:
            cut = None

    if cut is not None:
        if cut_type == module_types.FEATURES:
            #remove the module
            offspring.features.pop(cut)
            #fix channels
            offspring.fix_channels(cut, offspring.len_features())
            return offspring

        elif cut_type == module_types.CLASSIFICATION:
            #remove the module
            offspring.classification.pop(cut - offspring.len_features())
            #fix channels
            offspring.fix_channels(cut, offspring.len_features() + offspring.len_classification())
            return offspring

    return offspring