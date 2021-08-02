import collections
from typing import Counter
import numpy as np
import random
import math


def initial_population(pop_size=20):
    population = np.random.randint(high=7, low=0, size=(pop_size, 8))
    return population

def fitness_function(population):
    solved = False
    solution = None
    best = None
    best_fitness = 0

    pop_num_collisions = collisions(population)
    fitness_scores = [(28-collision)/28 for collision in pop_num_collisions] #normalize the collisions (highest:1, lowest:0)
    avg_fitness = sum(fitness_scores)/len(fitness_scores)

    
    for i in range(len(fitness_scores)):
        #Track the best for the population
        if fitness_scores[i] > best_fitness:
            best_fitness = fitness_scores[i]
            best = population[i]

        #Check if an optimal solution was found
        if fitness_scores[i] == 1:
            print('SCORE!!!!!!!!!!!!!!!!!!!!!!!!!')
            solved = True
            solution = population[i]

    return fitness_scores, avg_fitness, best_fitness, best, solved, solution

def selection_crossover_mutation(population, fitness, avg_fitness, new_pop_size, mutation_chance):

    # If the fitness of an individual is less than the average remove them from the population
    for i in range(len(population)):
        if fitness[i] < avg_fitness:
            np.delete(population, i)
            np.delete(fitness, i)

    
    new_pop = []
    temp = [np.exp(score) for score in fitness]
    probs = [score/np.sum(temp) for score in temp] #softmax probabilities for selection

    for _ in range(new_pop_size):
        #Select
        parent1 = population[np.random.choice(np.arange(len(population)), p=probs)]
        parent2 = population[np.random.choice(np.arange(len(population)), p=probs)]

        #Crossover
        splice = np.random.randint(low=0, high=7)

        child = np.concatenate((parent1[:splice], parent2[splice:]))

        #Mutate
        for q in range(len(child)): 
            mutate = np.random.choice([True, False], p=[mutation_chance, 1-mutation_chance])
            if mutate:
                mutation = np.random.randint(low=0, high=7)
                child[q] = mutation # change row of queen at index=q to row=mutation

        new_pop.append(child)
    return new_pop


# HELPER METHODS

def transform(pop):
    total_x = []
    for p in pop:
        x = []
        for i in range(len(p)):
            x.append([p[i], i]) #pairs the row with its index number
        total_x.append(x) 
    return total_x

def collisions(pop):
    total_collisions = []
    t_pop = transform(pop)
    horizontal_collisions = right(pop)
    diagonal_collisions = diagonals(t_pop)
    for coll in zip(horizontal_collisions, diagonal_collisions):
        total_collisions.append(sum(coll))
    return total_collisions

def diagonals(pop):
    counts = []
    for p in pop:
        count = 0
        for i in range(len(p)):
            #ur for going up and right, dr for going down and right
            row_ur = p[i][0] 
            col_ur = p[i][1]
            row_dr = p[i][0]
            col_dr = p[i][1] 
            for j in range(i, len(p)):
                #if out of bounds don't check
                if row_ur >= 0:
                    if (col_ur==p[j][1] and row_ur==p[j][0]):
                        count+=1
                if row_dr < 8:
                    if (col_dr==p[j][1] and row_dr==p[j][0]):
                        count+=1
                #take a step in diagonal direction
                row_dr += 1
                col_dr += 1
                row_ur -= 1
                col_ur += 1
        
        counts.append(count-16)
    return counts

def right(pop):
    pop_counts = []
    for p in pop:
        counts = Counter(p)
        p_counts = 0
        for q in counts:
            count = counts[q]-1
            p_counts += (count * (count+1))/2  #summation over queen collisions
        pop_counts.append(p_counts) 
    return pop_counts
 
def run():
    #Initialize 
    initial_pop_size = 50
    mutation_rate = 0.05 #Mutation=0 causes the generations to get stuck until variation is introduced
    gen = 0
    best_fitness = 0
    best = None
    avg_fitness = 0
    pop = initial_population(pop_size=initial_pop_size)

    #Loop through the generations
    while avg_fitness < 1.0:

        pop_size = len(pop)
        fitness, avg_fitness, potential_best_fitness, potential_best, solved, solution = fitness_function(pop)
        pop = selection_crossover_mutation(pop, fitness, avg_fitness, pop_size, mutation_rate)

        #Track the best individual
        if potential_best_fitness > best_fitness:
            best_fitness = potential_best_fitness
            best = potential_best

        #If best possible score achieved then stop
        if solved:
            print('Best Fitness Achieved: {} | Individual: {}'.format(round(best_fitness, 3), best))
            break

        #Introduce variation to get around convergence
        if (gen%500 == 0 and gen != 0):
            pop = np.concatenate((pop, initial_population(int(pop_size/2))), axis=0)

        #Decrease mutations so useful schemas are passed on more frequently (schema - sequence of numbers)
        if (gen%100 == 0):
            mutation_rate *= .75

        print('Generation: {} | Avg Score: {} | Pop Size: {}'.format(gen, round(avg_fitness, 3), pop_size))

        if gen == 2500:
            print('Best Fitness Achieved: {} | Individual: {}'.format(round(best_fitness, 3), best))
            break
        
        gen+=1

run()