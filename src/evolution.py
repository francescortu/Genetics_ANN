from src.nn_encoding import *
from scripts.train import train, eval

import multiprocessing as mp

MUTATION_RATE = 30
CROSSOVER_RATE = 70


class evolution():
    def __init__(self, population_size=10, holdout=1, mating=True, dataset=None, batch_size=4):
        """
        initial function fun is a function to produce nets, used for the original population
        scoring_function must be a function which accepts a net as input and returns a float
        """
        print(dataset)
        try:
            trainloader, testloader, input_size, n_classes, input_channels = dataset(batch_size)

        except Exception as e:
            print("Error: dataset not found")
            print(e)
            return

        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = batch_size
        self.dataset = dataset

        self.population_size = population_size
        self.population = []
        self.scores = []

        for _ in range(self.population_size):
            num_feat = np.random.randint(1, MAX_LEN_FEATURES)
            num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
            self.population.append(Net_encoding(num_feat, num_class, input_channels, n_classes, input_size))

        self.get_best_organism()
        self.holdout = max(1, int(holdout * population_size))

        self.mating = mating
        
        

    def generation(self):
        # statistics for each individual
        generation = [{"individual": i, "score": self.scores[i], "len": self.population[i]._len(), "genotype": self.population[i]} for i in range(self.population_size)]

        # create new population 
        new_population = [self.best_organism] # Ensure best organism survives

        for i in range(self.population_size - 1):
            parent_1_idx = i % self.holdout
            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx

            if np.random.randint(100) < CROSSOVER_RATE:
                child1, child2 = GA_crossover(self.population[parent_1_idx], self.population[parent_2_idx])
                offspring = child1 if child1._len() < child2._len() else child2
            else:
                offspring = self.population[parent_1_idx]
    
            if np.random.randint(0, 100) < MUTATION_RATE:
                GA_mutation(offspring)
            if np.random.randint(0, 100) < MUTATION_RATE:
                dsge_mutation(offspring)
            new_population.append(offspring)
        
        self.population = new_population

        return generation

    
    def get_best_organism(self):   
        
        #self.scores = [self.scoring_function(x) for x in self.population]
        
        ## added code for multiprocessing
        scores = mp.Array('i', range(self.population_size))
        processes = []
        for i,x in enumerate(self.population):
            p = mp.Process(target=self.scoring_function, args=(x,i,scores,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join() 
                   
        self.scores = scores[:]
        ##

        self.population = [self.population[x] for x in np.argsort(self.scores)[::-1]]
        
        self.best_organism = copy.deepcopy(self.population[0])
        self.best_score = sorted(self.scores)[-1]

        return self.best_organism, self.best_score

    def training_function(self, model):
        train(model, self.trainloader, self.batch_size)
        return model

    def scoring_function(self, modelcode, index, scores):
        model = Net(modelcode)
        model = train(model, self.dataset, self.trainloader, self.batch_size)
        print("eval")
        accuracy = eval(model, self.testloader)
        print("scoring")
        scores[index] = accuracy
        return accuracy

    