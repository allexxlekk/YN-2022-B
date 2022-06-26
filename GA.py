from numpy.random import randint
from numpy.random import rand
from matplotlib import pyplot as plt
import random
import TFIDFcalc

# Load tf-idf value metric for each word in the vocabulary.
tfidf_list = TFIDFcalc.load("TFIDF.dat")

VOCAB_SIZE = 8520
POPULATION_SIZE = 200
n_GENERATIONS = 200

INIT_PROB = 0.25
MUTATE_PROB = 0.1
CROSSOVER_PROB = 0.6



class G:
    '''Defines the genome of a population.'''
    
    def __init__(self, start = False):
        '''Creates the genome.'''
        
        # Binary represetation.
        self.list = [0 for i in range(VOCAB_SIZE)]
        self.score = 0.0
        
        # Initalizing procedure if the genotype is part
        # of the 1st generation.
        if start:
            for i in range(VOCAB_SIZE):
                if INIT_PROB > rand():
                    self.list[i] = 1
            self.calc_score()
    
    def mutate(self):
        '''Mutates the genome's chromosomes '''

        for i in range(VOCAB_SIZE):
            if MUTATE_PROB > rand():
                self.list[i] = abs(self.list[i] - 1)  # Toggle bit.

    def crossover(p1, p2):
        '''Implements 2-point crossover for two parents. Returns two offsprings.'''
        
        if CROSSOVER_PROB > random.uniform(0, 1):
            offspring1 = G()
            offspring2 = G()
            
            # Calculate crossover points.
            index1 = random.randrange(1, VOCAB_SIZE-1)
            index2 = random.randrange(index1+1, VOCAB_SIZE)
                        
            # Perform crossover.
            offspring1.list[0:index1] = p1.list[0:index1]
            offspring1.list[index1:index2] = p2.list[index1:index2]
            offspring1.list[index2:] = p1.list[index2:]
            
            offspring2.list[0:index1] = p2.list[0:index1]
            offspring2.list[index1:index2] = p1.list[index1:index2]
            offspring2.list[index2:] = p2.list[index2:]
            
            return [offspring1, offspring2]
        
        return [p1, p2]  # Returns parents if crossover doesn't occurr.

    def calc_score(self):
        sum = 0.0
        size = self.list.count(1)
        
        if size == 0:
            self.score = 0.00001
        # Calculate average word value.
        for i in range(VOCAB_SIZE):
            if self.list[i] == 1:
                sum += tfidf_list[i]
        
        avg_word_val = float(sum/size)
        
        # Apply penalty if the number of words are not legal.
        if self.list.count(1) > 4000 or self.list.count(1) < 1000:
            avg_word_val = avg_word_val / 2
        
        self.score =  avg_word_val

def selection(current_pop):
    '''Creates the mating pool using tournament selection.'''
    
    k = 5   # Number of contestants in the tournament.
    rand_contestant = randint(len(current_pop)) # Choose first contestant.
    
    for i in randint(0, len(current_pop), k-1): # Choose the rest.
        
        # Compare and select the fittest.
        if current_pop[i].score > current_pop[rand_contestant].score :
            rand_contestant = i
    return current_pop[rand_contestant]
                
def gen_best(current_pop):
    '''Returns the best score and genome.'''
    best = 0
    for g in current_pop:
        if g.score > best:
            best_g = g
            best = g.score
    return best,best_g

def gen_avrg(current_pop):
    '''Retursn the score average of a generation.'''
    sum = 0.0
    for g in current_pop:
        sum += g.score
    return float(sum/len(current_pop))

def GA():
    
    history = []
    # Create starting population.
    current_population = []
    for i in range(POPULATION_SIZE):
        g = G(start = True)
        current_population.append(g)
    
    # Get best genome and score of the generation.
    current_best, current_best_genome= gen_best(current_population)
        
    history.append(current_best)
    gen = 0
    
    es_counter = 0  # Early stopping counter.
    for i in range(n_GENERATIONS):
        
        
        gen = i+1
        
        # Create the next generation.
        # Select parents for mating.
        mating_pool = [selection(current_population) for _ in range(POPULATION_SIZE)]
        next_gen = []
        for i in range(0, POPULATION_SIZE-1, 2):
            
            # Get the parents:
            p1 = mating_pool[i]
            p2 = mating_pool[i+1]
            
            # Get the offsprings
            offsprings = G.crossover(p1, p2)

            # Mutate and evaluate the offsprings
            for os in offsprings:
                os.mutate()
                os.calc_score()
                
                next_gen.append(os)

        current_population = next_gen
        new_best, new_best_genome= gen_best(current_population)
        
        # Early stopping if not better than the previous for 50 generations.
        
        if float((new_best-current_best)/current_best) < 0.005:
            es_counter += 1
            
            if es_counter >= 50:
                if new_best > current_best:
                    best = new_best
                    best_genome = new_best_genome
                else:
                    best = current_best
                    best_genome = current_best_genome
                history.append(best)
                break
        else:
            es_counter = 0
        
        current_best = new_best
        current_best_genome = new_best_genome

        best = current_best
        best_genome = current_best_genome

        history.append(current_best)
    print(f'Best score: {best}, in {gen} Generations.')
    
    # Returns statistics of the geneticc algorithm
    return {"max_score" : best, "gens" : gen, "history" : history, "genome" : best_genome}


# Run 10 times.
stats = [GA() for _ in range(10)]

# Calculate average max_score and average generation number.
# Plot the best performing run.
max_score = 0.0
avg_max_score = 0.0
avg_gens = 0.0
for s in stats:
    avg_max_score += s["max_score"]
    avg_gens += s["gens"]
    
    if s["max_score"] > max_score:
        best_run = s
        
avg_max_score = avg_max_score / 10
avg_gens = avg_gens / 10

print(f"Average best score: {avg_max_score}, Average n.o generations: {avg_gens}")
plt.plot(range(len(s["history"])), s["history"])
plt.xlabel('Generation')
plt.ylabel('Average word value')
plt.show()