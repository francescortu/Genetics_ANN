from src.nn_encoding import *
from scripts.train import train, eval
from scripts.dataloader import MNIST, cifar10
from src.evolution import evolution
from torchsummary import summary

import csv
import sys
import imageio
from os import listdir

LAST_LAYER_SIZE = 1

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

    # test last generation best organism
    trainloader , testloader, _, _, _ = dataset(batch_size)
    model = train(Net(best_net), trainloader , batch_size, all=True)
    acc = eval(model, testloader )


    original_stdout = sys.stdout # Save a reference to the original standard output
    with open('best_organism', 'w+') as d:
        sys.stdout = d
        print("Best organism accuracy: ", acc, "%")
        best_net.print_dsge_level()
        sys.stdout = original_stdout

    

    print("Best accuracy obtained: ", score)
    writer.writerows(res)
    f.close() 


def generate_random_net():
    num_feat = np.random.randint(1, MAX_LEN_FEATURES)
    num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
    return Net_encoding(num_feat, num_class, c_in, c_out, input_size)

def create_random_gif():
    for i in range(8):
        enc = generate_random_net()
        enc.draw(i)

    frames = []
    # Build GIF

    for filename in listdir('images_net'):
        image = imageio.imread('images_net/'+filename)
        frames.append(image)

    imageio.mimsave('nn_evolution.gif', frames, format='GIF', duration=1)

if __name__ == "__main__":
   
    # load dataset for the following test function
    batch_size = 4

    dataset = MNIST
    print("\n\n Evolution of a population of networks: \n\n")
    test_evolution(dataset, batch_size)

