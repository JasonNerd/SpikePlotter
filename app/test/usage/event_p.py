# file_name:      event_p.py
# create_time:    2023/11/2-19:28

import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([[1, 0, 1, 1],
                   [1, 1, 0, 0],
                   [2, 1, 1, 0],
                   [0, 2, 2, 0]])
bin_edges = np.array([-1, 0, 1, 2, 3])

raster_list = [np.where(epoch > 0)[0]+bin_edges[0] for epoch in epochs]

fig, ax = plt.subplots()
# lineoffsets can be an array, however cannot be None or [], default is 1
ax.eventplot(raster_list, linelengths=0.8, colors='#6d6c6b')
plt.show()
print((np.array([2., 4, 3, 7])-1.34)*0.06)


