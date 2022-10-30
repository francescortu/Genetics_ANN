from dsge_level import *

'''

This file contains all the functions which are used to handle the GA level.
It deals with the modules at the GA level and the crossover operations to 
obtain the new offspring from two parents.

'''


class Net_encoding:
    "Describe the encoding of a network."
    def __init__(self, len_features, len_classification, c_in, c_out, input_shape):
  
        self.features = []
        self.classification = []
        self.last_layer = []
        self.input_shape = input_shape
        channels = self.init_random_channel(c_in, c_out, len_features + len_classification + 1 )
        
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
        last_in = (self.compute_shape_features(self.input_shape) ** 2) * self.features[-1].param['output_channels']
        self.classification[0].fix_channels(c_in = last_in)



##############################################
# CROSSOVER
##############################################

class cross_type(Enum):
    "crossover typology"
    ONE_POINT = 0
    BIT_MASK = 1


def GA_crossover(parent1, parent2):
    "randomly choose the crossover type"
    type = np.random.randint(len(cross_type))
    if type == 'ONE_POINT':
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

        child1, child2 = parent1, parent2
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
        child1.fix_channels(p[mask1[0]].len_features(), parent1.len_features()+1)
        child2.fix_channels(p[mask2[0]].len_features(), parent2.len_features()+1)
        
        child1.fix_channels(p[mask1[0]].len_features() + parent1.len_classification(), parent1.len_features() + parent1.len_classification()+1)
        child2.fix_channels(p[mask2[0]].len_features() + parent1.len_classification(), parent2.len_features() + parent1.len_classification() +1)

        return child1, child2
        
def GA_one_point(parent1, parent2):
        "cut the parent1 and parent2 at random position and swap the two parts"
        #find cutting point
        cut_parent1 = np.random.randint(1, parent1._len()-1)
        
        #identify the type of the cut
        cut1_type = parent1.GA_encoding(cut_parent1).M_type

        #find a cut on the same module also on parent2
        if cut1_type == module_types.FEATURES:
            cut_parent2 = np.random.randint(0, parent2.len_features()-1)
        elif cut1_type == module_types.CLASSIFICATION:
            cut_parent2 = np.random.randint(parent2.len_features(), parent2.len_features() + parent2.len_classification()-1)
        
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
