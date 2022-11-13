import csv
import imageio
from os import listdir
import matplotlib.pyplot as plt

def plot_individual_accuracy(x,y):
    plt.plot(x, y, 'bo')
    plt.xlabel('Individual', color = 'white')
    plt.ylabel('fitness (%)', color = 'white')
    plt.tick_params(axis='x', colors="white")
    plt.tick_params(axis='y', colors="white")
  
    plt.title(f"Accuracy for each individual in each generation", color = 'white')
    plt.savefig(f'results/plot/individual_accuracy.png', dpi=300, transparent=True)
    plt.close()

def plot_generation_accuracy(best_net_acc):
    x = [i for i in range(len(best_net_acc))]
    plt.plot(x, best_net_acc)

    plt.xlabel('Generation', color = 'white')
    plt.ylabel('fitness (%)', color = 'white')
    plt.tick_params(axis='x', colors="white")
    plt.tick_params(axis='y', colors="white")
    plt.xticks(range(len(best_net_acc))) # show only integer values

    plt.title(f"Best fitness value obtained for each generation", color = 'white')
    plt.savefig(f'results/plot/generation_accuracy.png', dpi=300, transparent=True)
    plt.close()


def plot_generation_netlen(best_net_len):
    x = [i for i in range(len(best_net_len))]
    plt.plot(x, best_net_len)

    plt.xlabel('Generation', color = 'white')
    plt.ylabel('fitness (%)', color = 'white')
    plt.tick_params(axis='x', colors="white")
    plt.tick_params(axis='y', colors="white")
    plt.xticks(range(len(best_net_len))) # show only integer values
    plt.yticks(range(max(best_net_len)))

    plt.title(f"Number of layers obtained for each generation's best individual", color = 'white')
    plt.savefig(f'results/plot/generation_net_len.png', dpi=300, transparent=True)
    plt.close()

def plot_results(population_size):

    # plot fitness for each individual in each generation
    # read data
    with open(f'results/all_generations_data.csv', mode='r') as csv_file:
        data = list(csv.reader(csv_file, delimiter = ','))

    n_row = len(data)
    x = []
    y = []
    best_net_acc = []
    best_net_len = []

    # store data
    for i in range(1, n_row):
        # x is the position of each individual in the population
        x.append(int(data[i][0])*population_size + int(data[i][1]))
        # y is the accuracy of each individual
        y.append(float(data[i][2]))
        if i % population_size == 0:
            best_net_acc.append(float(data[i][4]))
            best_net_len.append(int(data[i][5]))

    plot_individual_accuracy(x,y)
    
    plot_generation_accuracy(best_net_acc)

    plot_generation_netlen(best_net_len)


""" 
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
   
"""