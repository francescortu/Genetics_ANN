from src.nn_encoding import *
from scripts.train import train, eval
from scripts.dataloader import MNIST, cifar10
from src.evolution import evolution

import csv
import sys
from os import listdir
import time

from plot_results import *

def run_evolution(dataset, population_size = 2, num_generations=2, batch_size=4, subpath =''):
    '''
    input: 
        - the dataset we want to train the population on
        - population_size: how many individuals for each generation
        - the number of generations we want to train
        - the batch_size associated to trainloader and testloader
        - subpath: the path where we want to save the results
    '''
    # create a population of random networks
    curr_env = evolution(population_size, holdout=0.6, mating=True, dataset=dataset, batch_size=batch_size)
    
    # run evolution and write result on file
    path = 'results/'
    if subpath:
        path += subpath 
        if not os.path.isdir(path):
            os.mkdir(path)
     
    res = []

    generations = num_generations
    for i in range(generations):
        gen = curr_env.generation()
        this_generation_best, best_score = curr_env.get_best_organism()
        best_net = this_generation_best
        print("Generation ", i , "'s best network accuracy: ", best_score, "%")
        for j in range(population_size):
            res.append([i, j, gen[j]['score'], gen[j]['len'], best_score, best_net._len()])
            # save encoding of best network for each generation
            net_obj_py = open(f"results/best_net_encoding_res/gen{i:003}.pkl", "wb")
            pickle.dump(gen[j]['genotype'], net_obj_py)
            net_obj_py.close()

    # test last generation best organism
    trainloader , testloader, _, _, _ = dataset(batch_size, test = True)
    model = train(Net(best_net), trainloader , batch_size, all=True)
    acc = eval(model, testloader)
    
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(f'{path}/best_organism', 'w+') as d:
        sys.stdout = d
        print("Best organism accuracy: ", acc, "%")
        best_net.print_dsge_level()
        sys.stdout = original_stdout

    # save best organism object in specific subfolder
    net_obj_py = open(f"{path}/best_organism.pkl", "wb")
    pickle.dump(Net(best_net), net_obj_py)
    net_obj_py.close()

    # save results to file
    f = open(f'{path}/all_generations_data.csv', 'w+', newline='')
    # create the csv writer
    writer = csv.writer(f)

    fieldnames = ['generation', 'individual', 'accuracy', 'num_layers', 'best_accuracy', 'best_num_layers']
    writer.writerow(fieldnames)
    
    print("Best accuracy obtained: ", best_score)
    writer.writerows(res)
    f.close() 



def print_usage():
    print("Usage: python main.py [dataset] [population_size] [num_generations] [batch_size] [subpath]")
    # add more info about which datasets are available
    sys.exit(1)

if __name__ == "__main__":
   
    # read arguments provided by user
    args = len(sys.argv) 

    # provide them all, otherwise they are set to default
    if args > 5: 
        if not isinstance(sys.argv[1], str) or not sys.argv[2].isdigit() or not sys.argv[3].isdigit() or not sys.argv[4].isdigit() or not isinstance(sys.argv[5], str):  
            print_usage()
        else:
            # choose dataset
            if str(sys.argv[1]).lower() == 'cifar10':
                dataset = cifar10
            elif str(sys.argv[1]).lower() == 'mnist':
                dataset = MNIST
            else: 
                print_usage()
            
            # set population size
            population_size = int(sys.argv[2])
            # set number of generations
            num_generations = int(sys.argv[3])
            # set batch size
            batch_size = int(sys.argv[4])
            # set subpath
            subpath = str(sys.argv[5])

    # set default values
    else: 
        dataset = MNIST
        population_size = 2
        num_generations = 2
        batch_size = 4
        subpath = ''
    
    
    # run evolution
    print(f"\n\n Evolution of a population of networks: \n dataset: {dataset}, population_size: {population_size}, number of generation: {num_generations},  batch size: {batch_size}, path: {subpath} \n\n")
    print("Running Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu") )
    run_evolution(dataset, population_size, num_generations, batch_size, subpath = subpath) 
    #read_results(population_size, subpath = subpath)

    # check best network saved
    """ filename = f"results/{subpath}/best_organism.pkl"
    with open(filename, 'rb') as f:
        net = pickle.load(f)
        trainloader , testloader, _, _, _ = dataset(batch_size)
        model = train(net, trainloader , batch_size, all=True)
        eval(model, testloader) """