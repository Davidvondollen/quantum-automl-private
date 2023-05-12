from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

import numpy as np
import scipy as sp
import random
import itertools
from dwave.system import DWaveCliqueSampler,DWaveSampler,EmbeddingComposite
import neal
import dimod
from scipy import stats


def in_bounds(point, bounds):
    # enumerate all dimensions of the point
    for d in range(len(bounds)):
        # check if out of bounds for this dimension
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True


def calculate_pairwise_distance_population(population):
    combinations = list(itertools.permutations(population, 2))
    # print(combinations)
    dists = []
    for i in combinations:
        dists.append(np.abs(np.linalg.norm(i[0] - i[1])))
    # print(dists)
    return np.mean(dists)


def select_individuals_quantum(population, scores, k, sampler):
    # Get population size
    population_size = len(population)
    # print(population_size)
    pop_index = list(range(population_size))
    # print(pop_index)
    alpha = 10
    beta_ = 1000
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # scores = np.abs(np.array(scores))
    # scores = np.array(scores)/np.max(np.array(scores))
    # scores = scores.tolist()
    for idx, i_ in enumerate(pop_index):
        # if np.isfinite(i_):
        # i_ =0.
        bqm.add_variable(idx, (scores[idx])) * alpha
        # print(1+ i_*alpha)
    # dists = []
    for p0, p1 in itertools.combinations(pop_index, 2):
        # print(population[p0], population[p1])
        dist_ = np.abs(stats.pearsonr(population[p0], population[p1])[0])
        # dist_ = np.abs(np.linalg.norm(population[p0]-population[p1]))
        bqm.add_interaction(p0, p1, (dist_) * beta_)

        # dists.append(np.linalg.norm(population[p0]-population[p1]))
        # print(dist_)
    # dists = np.array(dists)/np.max(np.array(dists))
    # dists = dists.tolist()
    # for dist_ in dists:
    # bqm.add_interaction(p0, p1, -(dist_)*beta_)

    bqm.update(dimod.generators.combinations(pop_index, k, strength=100000))
    # sampler = DWaveCliqueSampler(solver=dict(qpu=True))
    # sampler = neal.SimulatedAnnealingSampler()
    sample_ = sampler.sample(bqm)
    # print(sample_)
    sample_ = sample_.first
    sample_energy = sample_.energy
    sample = sample_.sample
    # print(sample.values())

    # parents = np.where(sample==1)
    # print(parents)

    new_scores = []
    new_pop = []
    for idx, i in enumerate(scores):
        # selected_features[k-1, fi] = sample[f]
        # print(f, fi)
        if sample[idx] == 1.0:
            new_scores.append(i)
            new_pop.append(population[idx])

    # print(scores[parent_idx])

    return new_pop, new_scores


def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1, chromosome_length - 1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))

    # Return children
    return child_1, child_2


# evolution strategy (mu + lambda) algorithm
def es_plus(objective, bounds, n_iter, step_size, mu, lam, decay_rate):
    geno_divers = []
    best_fitness = []
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    # print(population)
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        best = population[np.argmax(np.array(scores))]
        # rank scores in ascending order
        ranks = argsort(scores)
        # select the indexes for the top mu ranked solutions
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        selected = selected[:mu]
        pop = [population[i] for i, _ in enumerate(ranks) if ranks[i] < mu]
        pop = pop[:mu]
        geno_divers.append(calculate_pairwise_distance_population(pop))
        print(calculate_pairwise_distance_population(pop))
        # print(selected)
        # create children from parents
        children = list()
        for idx, i in enumerate(range(mu)):
            # check if this parent is the best solution ever seen
            # print(i)

            parent1 = best
            # parent1 = pop[random.randint(0,len(pop)-1)]
            # children.append(parent1)
            # children.append(best)
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    parent2 = pop[random.randint(0, len(pop) - 1)]
                    child, child_2 = breed_by_crossover(parent1, parent2)
                    # child = pop[random.randint(0,len(pop)-1)]
                    child = child + randn(len(bounds)) * step_size
                    # child = child + np.random.randn(len(parent1)) * np.random.uniform(low=0.001, high=0.01)
                    # x_ti = population[np.random.choice(list(range(len(population))))]
                    # x_ti = population[np.random.choice(selected)]
                    # x_si = population[i]
                    # child = x_si + np.random.uniform(low=0, high=1) * randn(len(bounds))
                    # child = x_si + np.random.uniform(low=0, high=1) * (x_si - x_ti)
                children.append(child)
        # replace population with children
        population = children
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
        print(len(population))
        best_fitness.append(best_eval)
        # print(len(population))
        # step_size *= decay_rate
    return [best, best_eval], best_fitness, geno_divers


# evolution strategy (mu, lambda) algorithm
def es_comma(objective, bounds, n_iter, step_size, mu, lam, decay_rate):
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = []
    geno_divers = []
    best_fitness = []
    for _ in range(lam):
        candidate = None
        # TODO Redo bounds from function level
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        # rank scores in ascending order
        # print(scores)
        # print(argsort(scores))
        ranks = argsort(scores)

        # print(ranks)
        # select the indexes for the top mu ranked solutions
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        selected = selected[:mu]
        pop = [population[i] for i, _ in enumerate(ranks) if ranks[i] < mu]
        pop = pop[:mu]
        geno_divers.append(calculate_pairwise_distance_population(pop))
        print(calculate_pairwise_distance_population(pop))
        # print(selected)
        # create children from parents
        children = list()
        for idx, i in enumerate(range(mu)):
            # check if this parent is the best solution ever seen
            # print(i)

            # parent1 = best
            # parent1 = pop[random.randint(0,len(pop)-1)]
            parent1 = pop[idx]
            # children.append(parent1)
            # children.append(population[i])
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    parent2 = pop[random.randint(0, len(pop) - 1)]
                    child, child_2 = breed_by_crossover(parent1, parent2)
                    # child = pop[random.randint(0,len(pop)-1)]
                    child = child + randn(len(bounds)) * step_size
                    # child = child + np.random.randn(len(parent1)) * np.random.uniform(low=0.001, high=0.01)
                    # x_ti = population[np.random.choice(list(range(len(population))))]
                    # x_si = population[i]
                    # child = x_si + np.random.uniform(low=0, high=1) * randn(len(bounds))
                    # child = x_si + np.random.uniform(low=0, high=1) * (x_si - x_ti)
                children.append(child)
        # replace population with children
        population = children
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))

        print(len(population))
        best_fitness.append(best_eval)
        # step_size *= decay_rate
    return [best, best_eval], best_fitness, geno_divers


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)

    # Pick individuals for tournament
    fighter_1 = np.random.randint(0, population_size - 1)
    fighter_2 = np.random.randint(0, population_size - 1)

    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    ighter_2_fitness = scores[fighter_2]

    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness > fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2
    # k_sampled = np.random.uniform(low=0, high=population_size, size=(k,))
    # winner = np.argmax(k_sampled)

    # Return the chromsome of the winner
    return winner


def select_individual_by_tournament_k(population, scores, k):
    # Get population size
    population_size = len(scores)

    # Pick individuals for tournament
    # fighter_1 = np.random.randint(0, population_size-1)
    # fighter_2 = np.random.randint(0, population_size-1)

    # Get fitness score for each
    # fighter_1_fitness = scores[fighter_1]
    # ighter_2_fitness = scores[fighter_2]

    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    # if fighter_1_fitness > fighter_2_fitness:
    # winner = fighter_1
    # else:
    # winner = fighter_2
    k_sampled = np.random.uniform(low=0, high=population_size, size=(k,))
    winner = np.argmax(k_sampled)

    # Return the chromsome of the winner
    return winner


# evolution strategy (mu, lambda) algorithm
def es_tournament(objective, bounds, n_iter, step_size, mu, lam, decay_rate):
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    geno_divers = []
    best_fitness = []
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                print(epoch, best_eval)
                best, best_eval = population[idx], i
        # rank scores in ascending order
        # print(scores)
        # print(argsort(scores))

        # print(ranks)
        # select the indexes for the top mu ranked solutions
        # selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        # pop = [population[i] for i,_ in enumerate(ranks) if ranks[i] < mu]
        geno_divers.append(calculate_pairwise_distance_population(population))
        print(calculate_pairwise_distance_population(population))
        # print(selected)
        # create children from parents
        children = list()

        # create children for parent
        for _ in range((lam)):
            child = None
            while child is None or not in_bounds(child, bounds):
                cand = select_individual_by_tournament_k(population, scores, mu)
                child = population[cand] + randn(len(bounds)) * step_size
                # x_ti = population[np.random.choice(list(range(len(population))))]
                # x_si = population[i]
                # child = x_si + np.random.uniform(low=0, high=1) * randn(len(bounds))
                # child = x_si + np.random.uniform(low=0, high=1) * (x_si - x_ti)
            children.append(child)
        # replace population with children
        population = children
        print(len(population))
        best_fitness.append(best_eval)
        step_size *= decay_rate
    return [best, best_eval], best_fitness, geno_divers


# evolution strategy (mu, lambda) algorithm
def es_comma_mean(objective, bounds, n_iter, step_size, mu, lam, pop_size):
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
        # rank scores in ascending order
        # print(scores)
        # print(argsort(scores))
        ranks = argsort(argsort(scores))

        # print(ranks)
        # select the indexes for the top mu ranked solutions
        selected = [population[i] for i, _ in enumerate(ranks) if ranks[i] < mu]
        # print(selected)
        # create children from parents
        mean_selected = np.array(selected).mean(axis=0)
        print("mean: ", mean_selected)
        print("best score: ", best_eval)
        print("mean eval: ", objective(mean_selected))
        # print(mean_selected)
        children = []
        children.append(mean_selected)
        for _ in range(pop_size - 1):
            child = None
            while child is None or not in_bounds(child, bounds):
                child = mean_selected + randn(len(bounds)) * step_size
            children.append(child)
        # replace population with children
        population = children
    return [best, best_eval]


# evolution strategy (mu, lambda) algorithm
def es_comma_mean(objective, bounds, n_iter, step_size, mu, lam, pop_size, decay_rate):
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                print(epoch, best_eval)
                best, best_eval = population[idx], i
        # rank scores in ascending order
        # print(scores)
        # print(argsort(scores))
        ranks = argsort(argsort(scores))

        # print(ranks)
        # select the indexes for the top mu ranked solutions
        selected = [population[i] for i, _ in enumerate(ranks) if ranks[i] < mu]
        # print(selected)
        # create children from parents
        mean_selected = np.array(selected).mean(axis=0)
        print("mean: ", mean_selected)
        print("best score: ", best_eval)
        print("mean eval: ", objective(mean_selected))
        # print(mean_selected)
        children = []
        children.append(mean_selected)
        for _ in range(pop_size - 1):
            child = None
            while child is None or not in_bounds(child, bounds):
                child = mean_selected + (randn(len(bounds)) * (step_size * decay_rate))
            children.append(child)
        # replace population with children
        population = children
    return [population, best, best_eval]


# evolution strategy (mu + lambda) algorithm
def qa_es_plus(sampler, objective, bounds, n_iter, step_size, mu, lam):
    # sampler = DWaveCliqueSampler(solver=dict(qpu=True))
    # sampler = neal.SimulatedAnnealingSampler()
    decay_rate = 1
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    geno_divers = []
    best_fitness = []
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        # best = population[np.argmin(np.array(scores))]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
        # rank scores in ascending order
        # ranks = argsort(argsort(scores))
        # select the indexes for the top mu ranked solutions
        # selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        selected_pop, selected_scores = select_individuals_quantum(population, scores, mu, sampler)
        # if epoch >= 1:

        # selected_pop.append(best)
        #  ranked_selected = np.argsort(np.argsort(selected_scores))
        # selected_pop, selected_scores = np.array(selected_pop)[ranked_selected], np.array(selected_scores)[ranked_selected]
        # selected_pop, selected_scores = selected_pop[:mu], selected_scores[:mu]
        # create children from parents
        # print(len(selected_pop))
        pop = selected_pop
        geno_divers.append(calculate_pairwise_distance_population(pop))
        print(calculate_pairwise_distance_population(pop))
        children = list()
        for idx, i in enumerate(range(mu)):
            # check if this parent is the best solution ever seen
            # print(i)

            parent1 = best
            # parent1 = pop[random.randint(0,len(pop)-1)]
            # parent1 = pop[idx]
            # children.append(parent1)
            # children.append(best)
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    parent2 = pop[random.randint(0, len(pop) - 1)]
                    child, child_2 = breed_by_crossover(parent1, parent2)
                    child = child + randn(len(bounds)) * step_size
                    # child = child + np.random.randn(len(parent1)) * np.random.uniform(low=0.001, high=0.01)
                    # x_ti = population[np.random.choice(list(range(len(population))))]
                    # x_ti = selected_pop[np.random.choice(list(range(len(selected_pop))))]
                    # x_si = best
                    # child = x_si + np.random.uniform(low=0, high=1) * randn(len(bounds))
                    # child = x_si + (np.random.uniform(low=0, high=1) * (x_si-x_ti))
                children.append(child)
        # replace population with children
        population = children
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
        population = children
        print(len(population))
        best_fitness.append(best_eval)
        # step_size *= decay_rate
    return [best, best_eval], best_fitness, geno_divers


def qa_es_comma(sampler, objective, bounds, n_iter, step_size, mu, lam):
    # sampler = DWaveCliqueSampler(solver=dict(qpu=True))
    decay_rate = 1
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = []
    geno_divers = []
    best_fitness = []
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
        # rank scores in ascending order
        # ranks = argsort(argsort(scores))
        # select the indexes for the top mu ranked solutions
        # selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        selected_pop, selected_scores = select_individuals_quantum(population, scores, mu, sampler)
        # ranked_selected = np.argsort(np.argsort(selected_scores))
        # selected_pop, selected_scores = np.array(selected_pop)[ranked_selected], np.array(selected_scores)[ranked_selected]
        # selected_pop, selected_scores = selected_pop[:mu], selected_scores[:mu]
        # create children from parents
        # print(len(selected_pop))
        pop = selected_pop
        geno_divers.append(calculate_pairwise_distance_population(pop))
        print(calculate_pairwise_distance_population(pop))
        children = []
        for idx, i in enumerate(range(mu)):
            # check if this parent is the best solution ever seen
            # print(i)
            # if i == mu:
            #    parent1 = best
            # parent1 = pop[random.randint(0,len(pop)-1)]
            # else:
            # parent1 = pop[idx-1]
            # parent1 = pop[random.randint(0,len(pop)-1)]
            parent1 = pop[idx]
            # children.append(parent1)
            # children.append(i)
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    parent2 = pop[random.randint(0, len(pop) - 1)]
                    child, child_2 = breed_by_crossover(parent1, parent2)
                    # child = pop[random.randint(0,len(pop)-1)]
                    child = child + randn(len(bounds)) * step_size
                    # child = child + np.random.randn(len(parent1)) * np.random.uniform(low=0.001, high=0.01)
                    # x_ti = population[np.random.choice(list(range(len(population))))]
                    # x_ti = selected_pop[np.random.choice(list(range(len(selected_pop))))]
                    # x_si = best
                    # child = x_si + np.random.uniform(low=0, high=1) * randn(len(bounds))
                    # child = x_si + (np.random.uniform(low=0, high=1) * (x_si-x_ti))
                children.append(child)
        # replace population with children
        population = children
        print(len(population))
        best_fitness.append(best_eval)
        # step_size *= decay_rate
    return [best, best_eval], best_fitness, geno_divers


def qa_es_plus_mean(objective, bounds, n_iter, step_size, mu, lam, pop_size, decay_rate):
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        for idx, i in enumerate(scores):
            if i < best_eval:
                best, best_eval = population[idx], i
        # rank scores in ascending order
        # print(scores)
        # print(argsort(scores))
        # ranks = argsort(argsort(scores))

        # print(ranks)
        # select the indexes for the top mu ranked solutions
        # selected = [population[i] for i,_ in enumerate(ranks) if ranks[i] < mu]
        selected_pop, selected_scores = select_individuals_quantum(population, scores, mu)
        # print(selected)
        # create children from parents
        mean_selected = np.array(selected_pop).mean(axis=0)
        print("mean: ", mean_selected)
        print("best score: ", best_eval)
        print("mean eval: ", objective(mean_selected))
        # print(mean_selected)
        children = []
        children.append(mean_selected)
        for _ in range(pop_size - 1):
            child = None
            while child is None or not in_bounds(child, bounds):
                child = mean_selected + (randn(len(bounds)) * (step_size * decay_rate))
            children.append(child)
        # replace population with children
        population = children
    return [best, best_eval]


if __name__ == "__main__":
    from optproblems.continuous import Weierstrass, Schaffer7, LunacekTwoRastrigins, Ackley, Rastrigin, Branin, \
        FletcherPowell, GoldsteinPrice, Griewank, Vincent, Himmelblau, SixHumpCamelback
    import pickle

    # from mpl_toolkits.mplot3d import Axes3D
    # %matplotlib inline
    chrom_size = 50
    n_experiments = 20
    n_iter = 100
    step_size = 0.9

    mu = 50
    lam = 100

    fs = [Ackley, Rastrigin, FletcherPowell, Griewank, Vincent, LunacekTwoRastrigins, Weierstrass, Schaffer7]
    fs2 = [Branin, GoldsteinPrice, Himmelblau, SixHumpCamelback]

    experiment_results = {}

    for func in fs:
        try:
            f = func(num_variables=chrom_size, name=str(func.__name__))

        except:
            f = func()
        # step_size = 0.1

        # r_min, r_max = 0., 10.
        try:

            r_min, r_max = f.min_bounds[0], f.max_bounds[0]

        except:
            r_min, r_max = -5, 5.

        experiment_results[func.__name__] = {}

        bounds = asarray([[r_min, r_max] for i in range(chrom_size)])

        qa_fitness_all2 = []
        qa_geno_all2 = []
        qa_c_fitness_all2 = []
        qa_c_geno_all2 = []
        es_c_fitness_all = []
        es_c_geno_all = []
        es_p_fitness_all = []
        es_p_geno_all = []
        for s in range(n_experiments):
            try:
                [best, score], qa_best_fitness, qa_geno = qa_es_plus(sampler, f, bounds, n_iter, step_size, mu, lam)
                # best, score = qa_es_plus_mean(objective, bounds, n_iter, step_size, mu, lam, 80, 0.6)
                print('Done!')
                print('f(%s) = %f' % (best, score))
                qa_fitness_all2.append(qa_best_fitness)
                qa_geno_all2.append(qa_geno)

                [best, score], qa_c_best_fitness, qa_c_geno = qa_es_comma(sampler, f, bounds, n_iter, step_size, mu,
                                                                          lam)
                # best, score = qa_es_plus_mean(objective, bounds, n_iter, step_size, mu, lam, 80, 0.6)
                print('Done!')
                print('f(%s) = %f' % (best, score))
                qa_c_fitness_all2.append(qa_c_best_fitness)
                qa_c_geno_all2.append(qa_c_geno)

                [best, score], es_c_best_fitness, es_c_geno = es_comma(f, bounds, n_iter, step_size, mu, lam,
                                                                       decay_rate)
                # best, score = es_comma_mean(objective, bounds, n_iter, step_size, mu, lam, 80, 0.6)
                print('Done!')
                print('f(%s) = %f' % (best, score))
                es_c_fitness_all.append(es_c_best_fitness)
                es_c_geno_all.append(es_c_geno)

                [best, score], es_p_best_fitness, es_p_geno = es_plus(f, bounds, n_iter, step_size, mu, lam, decay_rate)
                print('Done!')
                print('f(%s) = %f' % (best, score))
                es_p_fitness_all.append(es_p_best_fitness)
                es_p_geno_all.append(es_p_geno)
            except:
                pass

        experiment_results[func.__name__]["qa_p"] = [qa_fitness_all2, qa_geno_all2]
        experiment_results[func.__name__]["qa_c"] = [qa_c_fitness_all2, qa_c_geno_all2]
        experiment_results[func.__name__]["es_p"] = [es_p_fitness_all, es_p_geno_all]
        experiment_results[func.__name__]["es_c"] = [es_c_fitness_all, es_c_geno_all]



