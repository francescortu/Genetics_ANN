from nn_encoding import *
from scripts.train import train, eval
from scripts.dataloader import MNIST
from evolution import evolution
from torchsummary import summary

def test_mutation_GA_level():
    netcode = Net_encoding( 3, 2, 1, 10, 28)
    netcode.print_GAlevel()

    GA_mutation(netcode)
    print("\n\nAfter mutation:")
    netcode.print_GAlevel()

def test_mutation_dsge_level():
    netcode = Net_encoding( 3, 2, 1, 10, 28)
    summary(Net(netcode))

    new = dsge_mutation(netcode)
    print("\n\nAfter mutation:")
    summary(Net(new))
    #new.print_GAlevel()

def test_crossover():
    parent1 = Net_encoding(4, 2, 1, 10, 28)
    parent2 = Net_encoding(3, 2, 1, 10, 28)

    parent1.print_GAlevel()
    parent2.print_GAlevel()

    child1, child2 = GA_crossover(parent1, parent2)

    print('\n\nchild1:')
    child1.print_GAlevel()
    print('\n\nchild2:')
    child2.print_GAlevel()

# build a network and test it of mnist
def test_nn_encoding(trainloader, testloader, n_classes, batch_size, input_size):
    netcode = Net_encoding( 3, 2, 1, n_classes, input_size)
    model = Net(netcode)
    print("Built model:\n")
    netcode.print()

    train(model, trainloader, batch_size, epochs=1, inspected=1000)
    eval(model, testloader)


def test_evolution(trainloader, testloader, batch_size):
    curr_env = evolution(population_size=2, holdout=0.6, mating=True, trainloader=trainloader, testloader=testloader, batch_size=batch_size)

    # get current most suitable network (organism)
    best_net, score = curr_env.get_best_organism()
    
    print("Best accuracy obtained: ", score)

if __name__ == "__main__":
    # look at construction thorugh network enconding, how crossover is performed
    # and the new obtained encodings
    #test_crossover()

    # test mutation
    test_mutation_dsge_level()

    # load dataset for the following test function
    """ batch_size = 4
    trainloader, testloader, input_size, n_classes = MNIST(batch_size)

    #print("\n\n Construction of a network and test it of mnist: \n\n")
    #test_nn_encoding(trainloader, testloader, n_classes, batch_size, input_size)

    print("\n\n Evolution of a population of networks: \n\n")
    test_evolution(trainloader, testloader, batch_size) """

    
