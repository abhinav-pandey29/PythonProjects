import numpy as np
from salesmen import EvolutionarySalesman
import json

cities = np.load('data/cities_10.npy')

results_filename = './EvolutionarySalesman_10Cities.json'

salesman = EvolutionarySalesman(
    cities=cities, population_size=5000, mutation_rate=0.05,
    num_parents_per_gen=2500, num_generations=120
)

if __name__ == '__main__':

    results_generator = salesman.run()

    data = [res for res in results_generator]

    with open(results_filename, 'w') as f:
        json.dump(data, f, indent=4)