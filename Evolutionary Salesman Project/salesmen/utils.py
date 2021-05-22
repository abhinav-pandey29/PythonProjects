import numpy as np


def calc_travel_distance(cities, order):
    total_distance = 0
    travel_path = cities[order]
    for i in range(len(order)-1):
        city_current = travel_path[i]
        city_next = travel_path[i+1]
        distance = np.sum((city_current - city_next)**2)
        total_distance += distance
    return total_distance