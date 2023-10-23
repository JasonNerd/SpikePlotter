# file_name:      nwbRW.py
# create_time:    2023/10/19-21:53

import matplotlib.pyplot as plt
import numpy as np

from pynwb import NWBHDF5IO

nwb_file_path = "D:/datasets/SpikeData/nwb_sample/sub-P17HMH_ses-20080501_ecephys_image.nwb"

io = NWBHDF5IO(nwb_file_path, mode="r", load_namespaces=True)
nwb_file = io.read()

print(nwb_file)
