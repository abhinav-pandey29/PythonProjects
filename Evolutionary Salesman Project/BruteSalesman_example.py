import numpy as np
from salesmen import BruteSalesman
import json
import pandas as pd

import salesmen

cities = np.load('data/cities_10.npy')

results_filename = './BruteSalesman_10Cities.csv'

salesman = BruteSalesman(cities=cities)

if __name__ == '__main__':

    results_generator = salesman.run()

    with open(results_filename, 'w') as f:
        f.write("")
        f.close()

    new_data_list = []
    for i, data in enumerate(results_generator):
        new_data_list.append(data)
        if i % 2000 == 0:
            print(f"Iteration {i} reached. Best score yet: {data[-1]}")
            try:
                existing_data = pd.read_csv(results_filename)
            except:
                existing_data = pd.DataFrame()

            new_data = pd.DataFrame(
                new_data_list, columns=salesman.result_columns)
            new_data = pd.concat((existing_data, new_data), axis=0)
            new_data.to_csv(results_filename, index=False)

            new_data_list = []

