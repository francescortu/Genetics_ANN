from src.nn_encoding import *
from scripts.train import train, eval, test_model

MUTATION_RATE = 33
CROSSOVER_RATE = 70

class evolution():
    def __init__(self, population_size=10, holdout=1, mating=True, dataset=None, batch_size=4):
        """
        initial function fun is a function to produce nets, used for the original population
        scoring_function must be a function which accepts a net as input and returns a float
        """
        try:
            trainloader, testloader, input_size, n_classes = dataset(batch_size)
            input_channels = trainloader.dataset.data.shape[3]
        except:
            print("Error: dataset not found")
            return

        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = batch_size
        

        self.population_size = population_size
        self.population = []
   
        for _ in range(self.population_size):
            num_feat = np.random.randint(1, MAX_LEN_FEATURES)
            num_class = np.random.randint(1, MAX_LEN_CLASSIFICATION)
            self.population.append(Net_encoding(num_feat, num_class, input_channels, n_classes, input_size))

        self.get_best_organism()
        self.holdout = max(1, int(holdout * population_size))

        self.mating = mating
        
        

    def generation(self):
        new_population = [self.best_organism] # Ensure best organism survives

        for i in range(self.population_size - 1):
            parent_1_idx = i % self.holdout
            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx
            child1, child2 = GA_crossover(self.population[parent_1_idx], self.population[parent_2_idx])
            offspring = child1 if child1._len() < child2._len() else child2
    
            if np.random.randint(0, 100) < MUTATION_RATE:
                offspring = GA_mutation(offspring)
            if np.random.randint(0, 100) < MUTATION_RATE:
                offspring = dsge_mutation(offspring)
            new_population.append(offspring)
        
        self.population = new_population

    def get_best_organism(self):   
        scores = [self.scoring_function(x) for x in self.population]
        self.population = [self.population[x] for x in np.argsort(scores)[::-1]]
        
        self.best_organism = copy.deepcopy(self.population[0])
        self.best_score = sorted(scores)[-1]

        return self.best_organism, self.best_score

    def training_function(self, model):
        train(model, self.trainloader, self.batch_size)
        return model

    def scoring_function(self, modelcode):
        model = Net(modelcode)
        accuracy = 0
        if test_model(model, self.trainloader): # if model is properly working
            model = self.training_function(model)
            accuracy = eval(model, self.testloader)
        
        return accuracy