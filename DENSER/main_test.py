from tests.tests import *
from scripts.dataloader import MNIST


if __name__ == "__main__":
   trainloader, testloader, input_size, n_classes = MNIST(batch_size=4)

   #print("TEST CROSSOVER BETWEEN TWO NETWORKS...")
   #test_crossover(trainloader)

   #print("TEST OF MUTATION AT GA LEVEL...")
   #test_mutation_GA_level(trainloader)
    
   #print("TEST OF MUTATION AT DSGE LEVEL...")
   #test_mutation_dsge_level(trainloader) 

   print("TEST EVOLUTION...")
   test_evolution(trainloader)

   print("\n\nEverything passed")