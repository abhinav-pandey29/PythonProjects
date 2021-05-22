import numpy as np


def crossover(num_cities, parent_1, parent_2):
    crossover_pos = np.random.randint(0, num_cities+1)
    offspring_A, offspring_B = parent_1[:
                                        crossover_pos], parent_2[:crossover_pos]
    for i in range(num_cities):
        if parent_2[i] not in offspring_A:
            offspring_A = np.append(offspring_A, parent_2[i])
        if parent_1[i] not in offspring_B:
            offspring_B = np.append(offspring_B, parent_1[i])
    return offspring_A, offspring_B


def mutate(num_cities, individual, mutation_rate):
    if np.random.random() < mutation_rate:
        idx_1, idx_2 = np.random.rand(num_cities).argsort()[-2:]
        (individual[idx_1], individual[idx_2]) = (
            individual[idx_2], individual[idx_1])
    return individual
