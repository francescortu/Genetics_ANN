from src.nn_encoding import *
from scripts.train import train, eval
from scripts.dataloader import MNIST
from src.evolution import evolution
from torchsummary import summary

import csv
import sys
original_stdout = sys.stdout

def test_evolution(dataset, batch_size):
    curr_env = evolution(population_size=2, holdout=0.6, mating=True, dataset=dataset, batch_size=batch_size)
    
    # run evolution and write result on file
    f = open('results.csv', 'w+')
    # create the csv writer
    writer = csv.writer(f)

    fieldnames = ['generation', 'best_score', 'num_layers']
    writer.writerow(fieldnames)
    res = []

    generations = 2
    for i in range(generations):
        curr_env.generation()
        this_generation_best, score = curr_env.get_best_organism()
        best_net = this_generation_best
        print("Generation ", i , "'s best network accuracy: ", score, "%")
        res.append([i, score, best_net._len()])

    with open('best_organism', 'w+') as d:
        sys.stdout = d
        best_net.print_dsge_level()
        sys.stdout = original_stdout
        
    print("Best accuracy obtained: ", score)
    writer.writerows(res)
    f.close() 


if __name__ == "__main__":
   
    # load dataset for the following test function
    batch_size = 4

    dataset = MNIST
    print("\n\n Evolution of a population of networks: \n\n")
    test_evolution(dataset, batch_size)

    
