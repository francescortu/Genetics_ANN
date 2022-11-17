import csv
import imageio
from os import listdir
import matplotlib.pyplot as plt

from src.nn_encoding import Net_encoding
import numpy as np
import pickle

import random

def plot_individual_accuracy(x,y,color, path):
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c = color[i], cmap = 'viridis')

    plt.xlabel('Individual', color = 'white')
    plt.ylabel('fitness (%)', color = 'white')
    plt.tick_params(axis='x', colors="white")
    plt.tick_params(axis='y', colors="white")
  
    plt.title(f"Accuracy for each individual in each generation", color = 'white')
    plt.savefig(f'{path}/individual_accuracy.png', dpi=300, transparent=True)
    plt.close()

def plot_generation_accuracy(best_net_acc, path):
    x = [i for i in range(len(best_net_acc))]
    plt.plot(x, best_net_acc)

    plt.xlabel('Generation', color = 'white')
    plt.ylabel('fitness (%)', color = 'white')
    plt.tick_params(axis='x', colors="white")
    plt.tick_params(axis='y', colors="white")

    plt.title(f"Best fitness value obtained for each generation", color = 'white')
    plt.savefig(f'{path}/generation_accuracy.png', dpi=300, transparent=True)
    plt.close()


def plot_generation_netlen(best_net_len, path):
    x = [i for i in range(len(best_net_len))]
    plt.plot(x, best_net_len)

    plt.xlabel('Generation', color = 'white')
    plt.ylabel('Number of layers', color = 'white')
    plt.tick_params(axis='x', colors="white")
    plt.tick_params(axis='y', colors="white")
    plt.yticks(range(max(best_net_len)))

    plt.title(f"Number of layers obtained for each generation's best individual", color = 'white')
    plt.savefig(f'{path}/generation_net_len.png', dpi=300, transparent=True)
    plt.close()


'''
this function is used to plot the results of the evolution
- the accuracy of each individual in each generation
- the best accuracy obtained in each generation
- the number of layers of the best individual in each generation
'''
def plot_results(data, path):
    n_row = len(data)
    x = []
    y = []
    col_list = []
    best_net_acc = []
    best_net_len = []

    population_size = int(data[-1][1]) + 1
    num_gen = int(data[-1][0]) + 1
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_gen)]

    # store data
    for i in range(1,n_row):
        # x is the position of each individual in the population
        x.append(int(data[i][0])*population_size + int(data[i][1]))
        # y is the accuracy of each individual
        y.append(float(data[i][2]))
        col_list.append(colors[int(data[i][0])])

        if i % population_size == 0:
            best_net_acc.append(float(data[i][4]))
            best_net_len.append(int(data[i][5]))

    #plot_individual_accuracy(x,y,col_list, path)
    
    #plot_generation_accuracy(best_net_acc, path)

    plot_generation_netlen(best_net_len, path)


'''
this function reads all the data saved in file of results
'''
def read_results(subpath=''):

    # plot fitness for each individual in each generation
    # read data
    path = 'results/'
    if subpath:
        path += subpath 

    with open(f'{path}/cifar10/pop50_gen50_run1/all_generations_data.csv', mode='r') as csv_file:
        data = list(csv.reader(csv_file, delimiter = ','))
    
    # plot results 
    plot_results(data, path)

       
def plot_net_representation(subpath):
    path = 'results/'
    if subpath:
        path += subpath
    # take network encoding from file of results
    for filename in listdir(f"{path}/best_net_encoding_res"):
        if filename.endswith('.pkl'):
            with open(f'{path}/best_net_encoding_res/{filename}', 'rb') as f:
                net_encoding = pickle.load(f)
                gen_num = int(filename[-7:-4])
                net_encoding.draw(gen_num)

    # Build GIF
    """ frames = []

    for filename in listdir('images_net'):
        if filename.endswith('.png'):
            image = imageio.imread('images_net/'+filename)
            frames.append(image)

        imageio.mimsave('images_net/nn_evolution.gif', frames, format='GIF', duration=1) """
   
