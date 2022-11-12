from src.evolution import *
from scripts.train import test_model
from scripts.dataloader import MNIST, cifar10
import sys

# set std param for MNIST dataset on which we will test the network
DATASET = MNIST
BATCH_SIZE = 4
NUM_CLASSES = 10
INPUT_SIZE = 28
INPUT_CHANNELS = 1 #3 for CIFAR10

MAX_LEN_FEATURES = 10
MAX_LEN_CLASSIFICATION = 2



def test_crossover(trainloader):
    parent1 = generate_random_net() 
    parent2 = generate_random_net()

    print(bcolors.HEADER + "\nTesting one-point crossover between two random networks" + bcolors.ENDC)
    child1, child2 = GA_crossover(parent1, parent2, type = cross_type.ONE_POINT)
    child1.print_dsge_level()
    child2.print_dsge_level()

    assert(test_model(Net(child1),trainloader)) == True, "Should be True if new netowrk1 is valid"
    assert(test_model(Net(child2),trainloader)) == True, "Should be True if new netowrk2 is valid"

    print(bcolors.HEADER + "\nTesting bit-mask crossover between two random networks" + bcolors.ENDC)
    child3, child4 = GA_crossover(child1, child2, type = cross_type.BIT_MASK)
    child3.print_dsge_level()
    child4.print_dsge_level()

    assert(test_model(Net(child3),trainloader)) == True, "Should be True if new netowrk3 is valid"
    assert(test_model(Net(child4),trainloader)) == True, "Should be True if new netowrk4 is valid"


def test_mutation_GA_level(trainloader):
    netcode = Net_encoding(1,2,INPUT_CHANNELS,NUM_CLASSES,INPUT_SIZE)

    print(bcolors.HEADER +  "Random generated net:\n" + bcolors.ENDC)
    netcode.print_GAlevel()

    # test random mutation at GA level
    print(bcolors.HEADER + "\n\nTesting random mutation at GA level...\n" + bcolors.ENDC)
    GA_mutation(netcode)
    netcode.print_GAlevel()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"
    
    # test addition mutation at GA level
    print(bcolors.HEADER + "\nTesting addition mutation at GA level...\n" + bcolors.ENDC)
    GA_mutation(netcode, type = ga_mutation_type.ADDITION)
    netcode.print_GAlevel()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    # test replace mutation at GA level
    print(bcolors.HEADER + "\nTesting replace mutation at GA level...\n" + bcolors.ENDC)
    netcode.print_dsge_level()
    GA_mutation(netcode, type = ga_mutation_type.REPLACE)
    netcode.print_GAlevel()
    assert(test_model(Net(netcode),trainloader)) == True, bcolors.RED + "Should be True if new netowrk is valid" + bcolors.ENDC

    # test delete mutation at GA level
    print(bcolors.HEADER + "\nTesting delete mutation at GA level...\n" + bcolors.ENDC)
    GA_mutation(netcode, type = ga_mutation_type.REMOVAL)
    netcode.print_GAlevel()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"




def test_mutation_dsge_level(trainloader):
    netcode = generate_random_net()

    print(bcolors.HEADER + "Random generated net:\n" + bcolors.ENDC)
    # netcode.print_dsge_level()

    # perform mutation at dsge level
    print(bcolors.HEADER + "\n\nTesting random mutation at dsge level...\n" + bcolors.ENDC)
    dsge_mutation(netcode)
    # netcode.print_dsge_level()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    # test grammatical mutation at dsge level
    print(bcolors.HEADER + "\nTesting grammatical mutation at dsge level...\n" + bcolors.ENDC)
    dsge_mutation(netcode, type = dsge_mutation_type.GRAMMATICAL)
    # netcode.print_dsge_level()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    # test integer mutation at dsge level
    print(bcolors.HEADER + "\nTesting integer mutation at dsge level...\n" + bcolors.ENDC)
    dsge_mutation(netcode, type =dsge_mutation_type.INTEGER)
    # netcode.print_dsge_level()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"



'''
The following function tests everything together on completely random networks
'''
def test_evolution(trainloader):

    population_size = 5
    nets = []

    # initialize population
    for i in range(population_size):
        encoding = generate_random_net()
        nets.append(encoding)
        assert(test_model(Net(encoding),trainloader)) == True, bcolors.RED + "Should be True if new netowrk is valid" + bcolors.ENDC

    # test evolution
    generations = 5
    new_population = []

    print(bcolors.HEADER + "\nTesting the combination of crossover, mutation at GA and dsge level in evolution of population\n" + bcolors.ENDC)
    for i in range(generations):
        print(bcolors.HEADER + "\nGeneration: " + str(i) +  bcolors.ENDC)

        for j in range(population_size):
          
            parent_1_idx, parent_2_idx = np.random.choice(a=np.arange(population_size), size=2, replace=False)
            print(parent_1_idx, parent_2_idx)
            
            offspring = None
            parent1 = copy.deepcopy(nets[parent_1_idx])
            parent2 = copy.deepcopy(nets[parent_2_idx])

            try:
                # crossover
                child1, child2 = GA_crossover(parent1, parent2)
                offspring = child1 if child1._len() < child2._len() else child2
                assert(test_model(Net(offspring),trainloader)) == True, bcolors.RED +  "Should be True if new netowrk is valid" + bcolors.ENDC
       

                # mutation GA level
                GA_mutation(offspring)

                assert(test_model(Net(offspring),trainloader)) == True, bcolors.RED +  "Should be True if new netowrk is valid" + bcolors.ENDC
                
                # mutation dsge level
                dsge_mutation(offspring)
     
                assert(test_model(Net(offspring),trainloader)) == True, "Should be True if new netowrk is valid"

            except Exception as e:
                print(bcolors.RED + "Error in mutation:" + bcolors.ENDC)
                print(e)
                print(bcolors.RED)
                offspring.print_dsge_level()
                print(bcolors.ENDC)
                sys.exit(1)
 

            print(bcolors.HEADER + "Individual: " + str(j) +  bcolors.ENDC)
            
            new_population.append(offspring)
        
        nets = new_population


'''
auxiliary functions
'''

def generate_random_net():
    num_feat = np.random.randint(1, MAX_LEN_FEATURES)
    num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
    return Net_encoding(num_feat, num_class, INPUT_CHANNELS, NUM_CLASSES, INPUT_SIZE)

def test_generation_networks(trainloader, num_net = 100, only_print = False ):
    if only_print:
        print(bcolors.HEADER + "\nSimply print a net\n" + bcolors.ENDC)
        netcode = generate_random_net()
        netcode.print_dsge_level()
        netcode.setting_channels()
        netcode.print_dsge_level()
        return
    
    print(bcolors.HEADER + "\nTesting 100 random generation of networks" + bcolors.ENDC)
    number_of_errors = 0
    for i in range(num_net):
        netcode = generate_random_net()
    
        try:
        # netcode.print_dsge_level()
            model = Net(netcode)
            assert(test_model(model,trainloader)) == True, "Should be True if new netowrk is valid"
        

        except Exception as e:
            print(bcolors.ALT + "Error in generation: " + str(i) +  bcolors.ENDC)
            print(e)
            number_of_errors += 1
    if num_net == 1:
        netcode.print_dsge_level()
    print(bcolors.ALT + "Number of errors: " + str(number_of_errors) +  bcolors.ENDC)






# just for pretty print

class bcolors:
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    ALT = '\033[94m'
    RED = '\033[31m'
    YELLOW = '\033[89m'

    