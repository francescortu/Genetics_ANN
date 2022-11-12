from src.nn_encoding import *
from scripts.train import train, eval
from scripts.dataloader import MNIST, cifar10
from src.evolution import evolution
from torchsummary import summary

import csv
import sys
import imageio
from os import listdir


def run_evolution(dataset, population_size = 2, num_generations=2, batch_size=4):
    # create a population of random networks
    curr_env = evolution(population_size, holdout=0.6, mating=True, dataset=dataset, batch_size=batch_size)
    
    # run evolution and write result on file
    f = open('results/all_generations_data.csv', 'w+')
    # create the csv writer
    writer = csv.writer(f)

    fieldnames = ['generation', 'individual', 'accuracy', 'num_layers']
    writer.writerow(fieldnames)
    res = []

    generations = num_generations
    for i in range(generations):
        gen = curr_env.generation()
        this_generation_best, score = curr_env.get_best_organism()
        best_net = this_generation_best
        print("Generation ", i , "'s best network accuracy: ", score, "%")
        for j in range(population_size):
            res.append([i, j, gen[j]['score'], gen[j]['len']])
        #res.append([i, score, best_net._len()])

    # test last generation best organism
    trainloader , testloader, _, _, _ = dataset(batch_size)
    model = train(Net(best_net), trainloader , batch_size, all=True)
    acc = eval(model, testloader)

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open('best_organism', 'w+') as d:
        sys.stdout = d
        print("Best organism accuracy: ", acc, "%")
        best_net.print_dsge_level()
        sys.stdout = original_stdout

    
    print("Best accuracy obtained: ", score)
    writer.writerows(res)
    f.close() 

'''
def generate_random_net():
    num_feat = np.random.randint(1, MAX_LEN_FEATURES)
    num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
    return Net_encoding(num_feat, num_class, 1, 10,28)

def create_random_gif():
    for i in range(8):
        enc = generate_random_net()
        enc.draw(i)

    # Build GIF
    frames = []

    for filename in listdir('images_net'):
        if filename.endswith('.png'):
            image = imageio.imread('images_net/'+filename)
            frames.append(image)

        imageio.mimsave('images_net/nn_evolution.gif', frames, format='GIF', duration=1)
   
'''

if __name__ == "__main__":
   
    # read arguments provided by user
    args = len(sys.argv) 

    if args > 1 and not isinstance(sys.argv[1], str):
        print("Usage: python main.py [dataset] [population_size] [num_generations] [batch_size]")
        sys.exit(1)

    
    # choose dataset
    if args > 1 and sys.argv[1]: dataset = sys.argv[1]
    else: dataset = MNIST

    if args > 2 and sys.argv[2]: population_size = int(sys.argv[2])
    else: population_size = 2

    if args > 3 and sys.argv[3]: num_generations = int(sys.argv[3])
    else: num_generations = 2

    if args> 4 and sys.argv[4]: batch_size = int(sys.argv[4])
    else: batch_size = 4

    # run evolution
    print("\n\n Evolution of a population of networks: \n\n")
    run_evolution(dataset, population_size, num_generations, batch_size)


