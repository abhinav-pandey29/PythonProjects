from datetime import datetime
import itertools
import numpy as np
from .utils import calc_travel_distance


class BruteSalesman:

    result_columns = ['timestamp', 'iteration', 'travel_path',
                      'iteration_score', 'overall_best_path', 'overall_best_score']

    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(self.cities)

        self.best_solution = list(range(self.num_cities))
        self.best_score = self.travel_distance(self.best_solution)

        # Fitness
    def travel_distance(self, order):
        return calc_travel_distance(self.cities, order)

    # Brute-Force Algorithm
    def run(self, image_dir=None):
        best_scores = []
        iter_scores = []

        for i, path in enumerate(itertools.permutations(range(self.num_cities))):
            path = list(path)
            distance = self.travel_distance(path)
            if distance < self.best_score:
                self.best_solution = path
                self.best_score = distance

            best_scores.append(self.best_score)
            iter_scores.append(distance)

            yield [
                datetime.timestamp(datetime.now()),
                i, path, distance, self.best_solution, self.best_score
            ]


'''
Plotting Code (if required) -

try:
    fig, ax = plt.subplots(1, 3, figsize=(23, 10))
    ax[0].scatter(self.cities[:, 0],
                    self.cities[:, 1], c='red')
    ax[1].plot(self.cities[self.best_solution, 0],
                self.cities[self.best_solution, 1], 'b-')
    ax[1].scatter(self.cities[:, 0],
                    self.cities[:, 1], c='red')
    ax[2].plot(np.arange(i + 1), iter_scores,
                'b--', label='Iteration Score')
    ax[2].scatter(np.arange(i + 1),
                    iter_scores, c='blue')
    ax[2].plot(np.arange(i + 1), best_scores,
                'r-', label='Overall Best')
    ax[2].scatter(np.arange(i + 1),
                    best_scores, c='red')
    ax[0].set_title("City Locations")
    ax[1].set_title(
        "Iteration {} (score = {})".format(i, self.best_score))
    ax[2].set_title("Progression over Time")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Travelled Distance / Loss")
    ax[2].legend()
except:
    pass
else:
    fig.savefig(f'{image_dir}/gen_{i}.png')
    # plt.close('all')

'''
