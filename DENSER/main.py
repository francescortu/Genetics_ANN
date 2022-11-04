from src.nn_encoding import *
from scripts.train import train, eval
from scripts.dataloader import MNIST
from src.evolution import evolution
from torchsummary import summary

import csv

import imageio
from os import listdir

def test_evolution(dataset, batch_size):
    curr_env = evolution(population_size=2, holdout=0.6, mating=True, dataset=dataset, batch_size=batch_size)
    
    # run evolution and write result on file
    f = open('results.csv', 'w+')
    # create the csv writer
    writer = csv.writer(f)

    fieldnames = ['generation', 'best_score', 'num_layers']
    writer.writerow(fieldnames)
    res = []

    generations = 1
    for i in range(generations):
        curr_env.generation()
        this_generation_best, score = curr_env.get_best_organism()
        best_net = this_generation_best
        print("Generation ", i , "'s best network accuracy: ", score, "%")
        res.append([i, score, best_net._len()])
    
    print("Best accuracy obtained: ", score)
    writer.writerows(res)
    f.close() 

def generate_random_net():
    num_feat = np.random.randint(1, MAX_LEN_FEATURES)
    num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
    return Net_encoding(num_feat, num_class, LAST_LAYER_SIZE, 10, 28)

def create_random_gif():
    for i in range(8):
        enc = generate_random_net()
        enc.draw(gen=i)

    # Build GIF
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in listdir('images_net'):
            image = imageio.imread('images_net/'+filename)
            writer.append_data(image)

if __name__ == "__main__":
   
    # load dataset for the following test function
    batch_size = 4

    dataset = MNIST
    print("\n\n Evolution of a population of networks: \n\n")
    test_evolution(dataset, batch_size)

    
    