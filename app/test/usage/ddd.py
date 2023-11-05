from scipy.stats.mstats import winsorize
import numpy as np


a = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
])

new_data = winsorize(a, (0.1, 0.2), axis=1)
print(new_data)

