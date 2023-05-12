import random
import numpy as np
import dimod

import itertools
from dwave.system import DWaveCliqueSampler,DWaveSampler,EmbeddingComposite 
import neal
import dimod


from scipy.spatial.distance import hamming

def create_reference_solution(chromosome_length):

    number_of_ones = int(chromosome_length / 2)

    # Build an array with an equal mix of zero and ones
    reference = np.zeros(chromosome_length)
    reference[0: number_of_ones] = 1

    # Shuffle the array to mix the zeros and ones
    np.random.shuffle(reference)
    
    return reference


def create_starting_population(individuals, chromosome_length):
    # Set up an initial array of all zeros
    population = np.zeros((individuals, chromosome_length))
    # Loop through each row (individual)
    for i in range(individuals):
        # Choose a random number of ones to create
        ones = random.randint(0, chromosome_length)
        # Change the required number of zeros to ones
        population[i, 0:ones] = 1
        # Sfuffle row
        np.random.shuffle(population[i])
    
    return population


def calculate_fitness(reference, population):
    # Create an array of True/False compared to reference
    identical_to_reference = population == reference
    # Sum number of genes that are identical to the reference
    fitness_scores = np.array(identical_to_reference).sum(axis=1)
    
    return fitness_scores


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)
    
    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size-1)
    fighter_2 = random.randint(0, population_size-1)
    
    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]
    
    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2
    
    # Return the chromsome of the winner
    return population[winner, :]


def select_individual_by_tournament_mu(population, scores, mu):
    # Get population size
    population_size = len(scores)
    mu_pop = []
    scores_pop = []
    for i in range(mu):
        
    # Pick individuals for tournament
        fighter_idx = random.randint(0, population_size-1)
        mu_pop.append(fighter_idx)
        scores_pop.append(scores[fighter_idx])
   
    winner = mu_pop[np.argmax(scores_pop)]   
    
    
    # Return the chromsome of the winner
    return population[winner, :]


def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)
    
    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    
    child_2 = np.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
    
    # Return children
    return child_1, child_2
    

def randomly_mutate_population(population, mutation_probability):
    
    # Apply random mutation
        random_mutation_array = np.random.random(
            size=(population.shape))
        
        random_mutation_boolean = \
            random_mutation_array <= mutation_probability

        population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])
        
        # Return mutation population
        return population
    
def select_individuals_quantum(population, scores, k):
    # Get population size
    population_size = len(population)
    pop_index = list(range(population_size))
    #print(pop_index)
    alpha =10
    beta_ = 1
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    for idx, i_ in enumerate(scores):
        #if np.isfinite(i_):
            #i_ =0.
        bqm.add_variable(idx, -(1+ i_*alpha))
        #print(1+ i_*alpha)
        
    for p0, p1 in itertools.combinations(pop_index, 2):
        dist_ = hamming(population[p0], population[p1])
        #print(dist_)
        bqm.add_interaction(p0, p1, (dist_*beta_)) 
        
    bqm.update(dimod.generators.combinations(pop_index, k, strength=500))
    #sampler = DWaveCliqueSampler(solver=dict(qpu=True))
    sampler = neal.SimulatedAnnealingSampler()
    sample_ = sampler.sample(bqm, num_reads=1000)
    #print(sample_)
    sample_= sample_.first
    sample_energy = sample_.energy
    sample = sample_.sample
    #print(sample)
    
    parents = np.where(sample==1)
    #print(parents)
    
    parent_idx = []
    for idx, i in enumerate(scores):
        #selected_features[k-1, fi] = sample[f]
        #print(f, fi)
        if sample[idx] == 1.0:
            parent_idx.append(idx)
     
    #print(scores[parent_idx])
    
    return population[parent_idx], scores[parent_idx]


if __name__ == '__main__':
    # Set general parameters
    chromosome_length =20
    population_size = 15
    maximum_generation = 100 
    alpha = 10.
    beta = 1.

    prob = 1.
    k = 2

    best_score_progress = [] # Tracks progress

    # Create reference solution 
    # (this is used just to illustrate GAs)
    reference = create_reference_solution(chromosome_length)

    # Create starting population
    population = create_starting_population(population_size, chromosome_length)
    scores = calculate_fitness(reference, population)
   
    best_score = 0

    for generation in range(maximum_generation):
        # Create an empty list for new population
        new_population = []
        parent_pool, scores_pool = select_individuals_quantum(population, scores, k)
        # Create new popualtion generating two children at a time
        for i in range(int(population_size/2)):
            parent_1 = select_individual_by_tournament(parent_pool, scores_pool)
            parent_2 = select_individual_by_tournament(parent_pool, scores_pool)
            child_1, child_2 = breed_by_crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        # Replace the old population with the new one
        population = np.array(new_population)

        # Apply mutation
        mutation_rate = 0.02
        population = randomly_mutate_population(population, mutation_rate)
        #population = randomly_mutate_population_quantum(population, mutation_rate, mutation_n = )

        # Score best solution, and add to tracker
        scores = calculate_fitness(reference, population)
        best_score_candidate = np.max(scores)/chromosome_length * 100
        if best_score_candidate > best_score:
            best_score = best_score_candidate
        best_score_progress.append(best_score)
        best_score_idx = np.argmax(scores)
        print(best_score)

    # GA has completed required generation
    print ('End best score, % target: ', best_score)

    # Plot progress
    import matplotlib.pyplot as plt
    plt.plot(best_score_progress)
    plt.xlabel('Generation')
    plt.ylabel('Best score (% target)')
    plt.show()


    
    

    