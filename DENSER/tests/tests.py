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

    print("\nTesting one-point crossover between two random networks")
    child1, child2 = GA_crossover(parent1, parent2, type = cross_type.ONE_POINT)
    assert(test_model(Net(child1),trainloader)) == True, "Should be True if new netowrk is valid"
    assert(test_model(Net(child2),trainloader)) == True, "Should be True if new netowrk is valid"

    print("\nTesting bit-mask crossover between two random networks")
    child3, child4 = GA_crossover(child1, child2, type = cross_type.BIT_MASK)
    assert(test_model(Net(child3),trainloader)) == True, "Should be True if new netowrk is valid"
    assert(test_model(Net(child4),trainloader)) == True, "Should be True if new netowrk is valid"



def test_mutation_GA_level(trainloader):
    netcode = generate_random_net()

    print("Random generated net:\n")
    netcode.print_GAlevel()

    # test random mutation at GA level
    print("\n\nTesting random mutation at GA level")
    GA_mutation(netcode)
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    # test addition mutation at GA level
    netcode = generate_random_net()#Net_encoding(4, 2, 1, 10, 28)
    print("\nTesting addition mutation at GA level")
    GA_mutation(netcode, type = ga_mutation_type.ADDITION)
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    # test deletion mutation at GA level
    netcode = generate_random_net()#Net_encoding(1, 1, 1, 10, 28)
    print("\nTesting replace mutation at GA level")
    netcode.print_GAlevel()

    GA_mutation(netcode, type = ga_mutation_type.REPLACE)
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"

    netcode = generate_random_net()#Net_encoding(1, 1, 1, 10, 28)
    print("\nTesting deletion mutation at GA level")
    netcode.print_GAlevel()

    GA_mutation(netcode, type = ga_mutation_type.REMOVAL)
    assert(test_model(Net(netcode),trainloader)) == True, "Should be True if new netowrk is valid"




def test_mutation_dsge_level(trainloader):
    netcode = generate_random_net()

    print("Random generated net:\n")
    netcode.print_GAlevel()
    # perform mutation at dsge level
    print("\n\nTesting random mutation at dsge level")
    new = dsge_mutation(netcode)
    assert(test_model(Net(new),trainloader)) == True, "Should be True if new netowrk is valid"

    # test grammatical mutation at dsge level
    netcode = generate_random_net()
    print("\nTesting grammatical mutation at dsge level")
    new = dsge_mutation(netcode, type = dsge_mutation_type.GRAMMATICAL)
    assert(test_model(Net(new),trainloader)) == True, "Should be True if new netowrk is valid"

    # test integer mutation at dsge level
    netcode = generate_random_net()
    print("\nTesting integer mutation at dsge level")
    new = dsge_mutation(netcode, type =dsge_mutation_type.INTEGER)
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
    generations = 10
    new_population = []

    print("\nTesting the combination of crossover, mutation at GA and dsge level in evolution of population\n")
    for i in range(generations):
        print("Generation: ", i)

        for i in range(population_size - 1):
            parent_1_idx = i % 6
            parent_2_idx = min(population_size - 1, int(np.random.exponential(6)))
            
            child1, child2 = GA_crossover(nets[parent_1_idx], nets[parent_2_idx])
            offspring = child1 if child1._len() < child2._len() else child2
            offspring = GA_mutation(offspring)
            offspring = dsge_mutation(offspring)

            assert(test_model(Net(offspring),trainloader)) == True, "Should be True if new netowrk is valid"

            new_population.append(offspring)
        
        nets = new_population

