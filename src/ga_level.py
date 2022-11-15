from src.dsge_level import *
import sys

'''

This file contains all the functions which are used to handle the GA level.

* It deals with the modules at the GA level and the crossover operations to 
    obtain the new offspring from two parents.

* It also contains the functions to mutate the offspring at the GA level, that is to 
    say those operations which manipulate the network structure

'''
DEBUG = 0

class Net_encoding:
    "Describe the encoding of a network."
    def __init__(self, len_features, len_classification, c_in, c_out, input_shape):
  
        self.features = []
        self.classification = []
        self.last_layer = []
        self.input_shape = input_shape
        self.input_channels = c_in
        
        
        if len_features > MAX_LEN_FEATURES:
            len_features = MAX_LEN_FEATURES
        if len_classification > MAX_LEN_CLASSIFICATION:
            len_classification = MAX_LEN_CLASSIFICATION

        # add features blocks
        for i in range(0,len_features):
            self.features.append(Module(module_types.FEATURES))

        # add classification blocks
        for i in range(len_classification):
            self.classification.append(Module(module_types.CLASSIFICATION))

        # add last block
        self.last_layer.append(Module(module_types.LAST_LAYER, c_out = c_out))
        
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

    
    def compute_shape_features(self, input_shape = 32, max_len = None):
        "like the forward pass, compute the output shape of the features block"
        output_shape = input_shape
        if max_len == None:
            max_len = self.len_features()

        for i in range(max_len):
            output_shape = self.GA_encoding(i).compute_shape(output_shape)
        return output_shape
    
    def setting_channels(self):
        c_in = self.input_channels
        for i in range(self._len()):
            self.GA_encoding(i).param['input_channels'] = c_in
            for j in range(self.GA_encoding(i).len()):
                self.GA_encoding(i).layers[j].channels['in'] = c_in
                if self.GA_encoding(i).layers[j].type == layer_type.CONV or self.GA_encoding(i).layers[j].type == layer_type.LINEAR:
                    c_in = self.GA_encoding(i).layers[j].channels['out']
                else:
                    self.GA_encoding(i).layers[j].channels['out'] = c_in
            self.GA_encoding(i).param['output_channels'] = self.GA_encoding(i).layers[-1].channels['out']
        
        self.fix_first_classification()
    
    def update_encoding(self):
        current_input_shape = self.input_shape
        # check features layers
        # self.print_dsge_level()
        i = 0
        while i < self.len_features():
            invalid = False
            for j in range(self.GA_encoding(i).len()):
                if self.GA_encoding(i).layers[j].type == layer_type.CONV or self.GA_encoding(i).layers[j].type == layer_type.POOLING:
                    new_shape = self.GA_encoding(i).layers[j].compute_shape(current_input_shape)
                    if current_input_shape > self.GA_encoding(i).layers[j].param["kernel_size"] and new_shape > 0: 
                        current_input_shape = new_shape
                    # if the kernel size is bigger than input shape, the shape is too small and the layer must be removed
                    else:
                        invalid = True

            if invalid:
                self.features.pop(i)
            else:
                i += 1


                


    def get_input_shape(self):
        return self.input_shape

    def get(self):
        return self.GA_encoding
        
    def print_dsge_level(self):
        print(f"######## len: {self._len()} ##########")
        for i in range(self._len()):
            self.GA_encoding(i).print(i)
            if self.GA_encoding(i).M_type == module_types.FEATURES:
                print("output shape", self.compute_shape_features(self.input_shape, i+1)  )
        print("######################################")

    def print_GAlevel(self):
        "print only if the module is FEATURES or CLASSIFICATION"
        print("Net len:", self._len())
        for i in range(self._len()):
            print("-",i, self.GA_encoding(i).M_type, ' ',self.GA_encoding(i).param['input_channels'], ' ', self.GA_encoding(i).param['output_channels'])

    def fix_first_classification(self):
        # fix in channels of the first classification block
        last_in = (self.compute_shape_features(self.input_shape) ** 2) * self.GA_encoding(self.len_features()-1).param['output_channels']
        self.GA_encoding(self.len_features()).param['input_channels'] = last_in
        self.GA_encoding(self.len_features()).layers[0].channels['in'] = last_in


    def draw(self, gen):
        "draw the network"
        self.setting_channels()
        global START
        START = 0
        node_input = None
        
        length_f = self.len_features()
        length_c = self.len_classification()

        # initial background
        plt.figure(facecolor='black')

        for i in range(self._len()):
            if self.GA_encoding(i).M_type == module_types.FEATURES:
                is_last = False
                if i == self.len_features() - 1:
                    is_last = True
                START = self.GA_encoding(i).draw_features(START, length_f, last = is_last)

            elif self.GA_encoding(i).M_type == module_types.CLASSIFICATION:
                index = i - self.len_features()
                START, node_input = self.GA_encoding(i).draw_classification(START, length_c, length_f, index, node_in = node_input)
            
            else:
                START, node_input = self.GA_encoding(i).draw_classification(START, length_c, length_f, length_c, node_in = node_input) 


        # add title
        plt.title(f"Network representation, generation: {gen}", color = 'white')

        # remove axis
        pyplot.axis('equal')
        plt.axis('off')
        # change background color

        # save image
        plt.savefig(f'images_net/gen{gen:003}.png', dpi=300, transparent=True)
        plt.close()
        #pyplot.show()

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
       
    
    return child1, child2
        


def GA_one_point(parent1, parent2):
    "cut the parent1 and parent2 at random position and swap the two parts"



    # randomly choose if the cut is in the features or in the classification
    cut_parent1 = cut_parent2 = None
    #print(list(module_types)(-1))
    type = np.random.choice(list(module_types)[:-1])
    
    if type == module_types.FEATURES and parent1.len_features() > 1 and parent2.len_features() > 1:
        cut_parent1 = np.random.randint(1, parent1.len_features())
        cut_parent2 = np.random.randint(1, parent2.len_features())
    
    elif type == module_types.CLASSIFICATION and parent1.len_classification() > 1 and parent2.len_classification() > 1:
        cut_parent1 = np.random.randint(parent1.len_features()+1, parent1._len()-1)
        cut_parent2 = np.random.randint(parent2.len_features()+1, parent2._len()-1)

    #print("cuts are: ", cut_parent1, ' ', cut_parent2)
    
    # cut type
    if cut_parent1: cut1_type = parent1.GA_encoding(cut_parent1).M_type 
    else: cut1_type = None

    if cut1_type == module_types.FEATURES:
        aux1 = copy.deepcopy(parent1.features[cut_parent1:])
        aux2 = copy.deepcopy(parent2.features[cut_parent2:])

        parent1.features = copy.deepcopy(parent1.features[:cut_parent1])
        parent2.features = copy.deepcopy(parent2.features[:cut_parent2])
        
        parent1.features.extend(aux2)
        parent2.features.extend(aux1)
     

    elif cut1_type == module_types.CLASSIFICATION:
        aux1 = copy.deepcopy(parent1.classification[cut_parent1 - parent1.len_features():])
        aux2 = copy.deepcopy(parent2.classification[cut_parent2 - parent2.len_features():])

        parent1.classification = copy.deepcopy(parent1.classification[:cut_parent1 - parent1.len_features()])
        parent2.classification = copy.deepcopy(parent2.classification[:cut_parent2 - parent2.len_features()])
        
        parent1.classification.extend(aux2)
        parent2.classification.extend(aux1)
      

    return parent1, parent2

