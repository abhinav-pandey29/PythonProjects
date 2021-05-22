from datetime import datetime
import numpy as np
from .evolution_utils import crossover, mutate
from .utils import calc_travel_distance


class EvolutionarySalesman:

    def __init__(self, cities, population_size,
                 num_parents_per_gen, mutation_rate,
                 num_generations, crossover=crossover, mutate=mutate):
        self.cities = cities
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_parents_per_gen = num_parents_per_gen
        self.mutation_rate = mutation_rate

        self.crossover = crossover  # TODO: Create parameterized Crossover class
        self.mutate = mutate  # TODO: Create parameterized Mutation class

        self.num_cities = len(self.cities)
        self.best_score = float('inf')
        self.best_individual = list(range(self.num_cities))
        self.num_children_per_gen = self.population_size - self.num_parents_per_gen

    # Initialise population
    def initialize_population(self):
        order = list(range(self.num_cities))
        population = np.array([np.random.permutation(order)
                              for _ in range(self.population_size)])
        return population

    # Fitness
    def travel_distance(self, order):
        return calc_travel_distance(self.cities, order)

    # Reproduction
    def population_reproduction(self, best_individuals):
        children = []
        while len(children) < self.num_children_per_gen:
            for i in range(self.num_parents_per_gen):
                parent_1, parent_2 = best_individuals[i], best_individuals[(
                    i+1) % self.num_parents_per_gen]
                offspring_A, offspring_B = self.crossover(
                    self.num_cities, parent_1, parent_2)
                children.append(offspring_A)
                children.append(offspring_B)
                if len(children) >= self.num_children_per_gen:
                    break
        children = [self.mutate(self.num_cities, child, self.mutation_rate)
                    for child in children]
        return children

    # Simulation
    def run(self, image_dir=None):
        champ = self.best_individual
        gen_best_scores = []
        overall_best_scores = []

        old_population = self.initialize_population()
        for gen in range(self.num_generations):

            # Calculate Fitness
            pop_fitness = [self.travel_distance(
                individual) for individual in old_population]

            # Select Parent Pool
            pop_heirarchy = np.argsort(pop_fitness)
            best_individuals = old_population[pop_heirarchy][:self.num_parents_per_gen]
            weakest_individuals = old_population[pop_heirarchy][-int(
                0.2 * self.population_size):]
            parent_pool = np.append(
                best_individuals, weakest_individuals, axis=0)
            champ = best_individuals[0]
            champ_score = self.travel_distance(champ)
            if champ_score < self.best_score:
                self.best_score = champ_score
                self.best_individual = champ

            gen_best_scores.append(champ_score)
            overall_best_scores.append(self.best_score)

            # Reproduce
            children = self.population_reproduction(parent_pool)

            # Calculate Fitness
            children_scores = [self.travel_distance(
                child) for child in children]
            best_children_ids = np.argsort(children_scores)[
                :(self.num_children_per_gen)]
            best_children = [children[i] for i in best_children_ids]

            # Survival of fittest
            population = np.append(parent_pool, best_children, axis=0)
            np.random.shuffle(population)
            old_population = population

            yield {
                'timestamp': datetime.timestamp(datetime.now()),
                'generation': gen,
                'gen_scores': pop_fitness,
                'num_individuals': len(pop_fitness),
                'generation_best': {
                    'solution': champ.tolist(),
                    'score': champ_score
                },
                'overall_best': {
                    'solution': self.best_individual.tolist(),
                    'score': self.best_score
                }
            }


'''
Plotting Code (if required) -

try:
    fig, ax = plt.subplots(1, 3, figsize=(23, 10))
    ax[0].scatter(self.cities[:, 0],
                    self.cities[:, 1], c='red')
    ax[1].plot(self.cities[self.best_individual, 0],
                self.cities[self.best_individual, 1], 'b-')
    ax[1].scatter(self.cities[:, 0],
                    self.cities[:, 1], c='red')
    ax[2].plot(np.arange(gen + 1), gen_best_scores,
                'b--', label='Generation Best')
    ax[2].scatter(np.arange(gen + 1),
                    gen_best_scores, c='blue')
    ax[2].plot(np.arange(gen + 1), overall_best_scores,
                'r-', label='Overall Best')
    ax[2].scatter(np.arange(gen + 1),
                    overall_best_scores, c='red')
    ax[0].set_title("City Locations")
    ax[1].set_title(
        "Generation {} (score = {})".format(gen, self.best_score))
    ax[2].set_title("Progression over Time")
    ax[2].set_xlabel("Generation")
    ax[2].set_ylabel("Travelled Distance / Loss")
    ax[2].legend()
except:
    pass
else:
    fig.savefig(f'{image_dir}/gen_{gen}.png')

'''
