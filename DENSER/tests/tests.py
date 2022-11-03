from src.evolution import *
from scripts.train import test_model
from scripts.dataloader import MNIST


# set std param for MNIST dataset on which we will test the network
DATASET = MNIST
BATCH_SIZE = 4
NUM_CLASSES = 10
INPUT_SIZE = 28

MAX_LEN_FEATURES = 3
MAX_LEN_CLASSIFICATION = 2


def generate_random_net():
    num_feat = np.random.randint(1, MAX_LEN_FEATURES)
    num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
    return Net_encoding(num_feat, num_class, LAST_LAYER_SIZE, NUM_CLASSES, INPUT_SIZE)


def test_crossover(trainloader):
    parent1 = generate_random_net()
    parent2 = generate_random_net()

    print(bcolors.HEADER + "\nTesting one-point crossover between two random networks" + bcolors.ENDC)
    child1, child2 = GA_crossover(parent1, parent2, type = cross_type.ONE_POINT)
    assert(test_model(Net(child1),trainloader)) == True, "Should be True if new netowrk is valid"
    assert(test_model(Net(child2),trainloader)) == True, "Should be True if new netowrk is valid"

    print(bcolors.HEADER + "\nTesting bit-mask crossover between two random networks" + bcolors.ENDC)
    child3, child4 = GA_crossover(child1, child2, type = cross_type.BIT_MASK)
    assert(test_model(Net(child3),trainloader)) == True, "Should be True if new netowrk is valid"
    assert(test_model(Net(child4),trainloader)) == True, "Should be True if new netowrk is valid"



def test_mutation_GA_level(trainloader):
    netcode = generate_random_net()

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
    GA_mutation(netcode, type = ga_mutation_type.REPLACE)
    netcode.print_GAlevel()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    # test delete mutation at GA level
    print(bcolors.HEADER + "\nTesting delete mutation at GA level...\n" + bcolors.ENDC)
    GA_mutation(netcode, type = ga_mutation_type.REMOVAL)
    netcode.print_GAlevel()
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"




def test_mutation_dsge_level(trainloader):
    netcode = generate_random_net()

    print(bcolors.HEADER + "Random generated net:\n" + bcolors.ENDC)
    netcode.print_dsge_level()

    # perform mutation at dsge level
    print(bcolors.HEADER + "\n\nTesting random mutation at dsge level...\n" + bcolors.ENDC)
    new = dsge_mutation(netcode)
    netcode.print_dsge_level()
    assert(test_model(Net(new),trainloader)) == True, "Should be True if new netowrk is valid"

    # test grammatical mutation at dsge level
    print(bcolors.HEADER + "\nTesting grammatical mutation at dsge level...\n" + bcolors.ENDC)
    new = dsge_mutation(netcode, type = dsge_mutation_type.GRAMMATICAL)
    netcode.print_dsge_level()
    assert(test_model(Net(new),trainloader)) == True, "Should be True if new netowrk is valid"

    # test integer mutation at dsge level
    print(bcolors.HEADER + "\nTesting integer mutation at dsge level...\n" + bcolors.ENDC)
    new = dsge_mutation(netcode, type =dsge_mutation_type.INTEGER)
    netcode.print_dsge_level()
    assert(test_model(Net(new),trainloader)) == True, "Should be True if new netowrk is valid"



'''
The following function tests everything together on completely random networks
'''
def test_evolution(trainloader):

    population_size = 10
    nets = []

    # initialize population
    for i in range(population_size):
        encoding = generate_random_net()
        nets.append(encoding)
        assert(test_model(Net(encoding),trainloader)) == True, "Should be True if new netowrk is valid"

    
    # test evolution
    generations = 5
    new_population = []

    print(bcolors.HEADER + "\nTesting the combination of crossover, mutation at GA and dsge level in evolution of population\n" + bcolors.ENDC)
    for i in range(generations):
        print(bcolors.HEADER + "\nGeneration: " + str(i) +  bcolors.ENDC)

        for j in range(population_size):
            parent_1_idx = np.random.randint(0, population_size)
            parent_2_idx = np.random.randint(0, population_size)
            
            
            child1, child2 = GA_crossover(nets[parent_1_idx], nets[parent_2_idx])
            offspring = child1 if child1._len() < child2._len() else child2
            try:
                offspring = GA_mutation(offspring)
                assert(test_model(Net(offspring),trainloader)) == True, "Should be True if new netowrk is valid"
                
                offspring = dsge_mutation(offspring)
                assert(test_model(Net(offspring),trainloader)) == True, "Should be True if new netowrk is valid"

            except:
                print("Error in mutation")
                offspring.print_dsge_level() # if mutation fails, print the netcode to see what went wrong

            print(bcolors.HEADER + "Individual: " + str(j) +  bcolors.ENDC)
            new_population.append(offspring)
        
        nets = new_population



# just for pretty print

class bcolors:
    HEADER = '\033[95m'
    ENDC = '\033[0m'
