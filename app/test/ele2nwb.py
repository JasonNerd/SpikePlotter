# file_name:      ele2nwb.py
# create_time:    2023/10/22-21:25

import neo
import os
from pynwb import NWBHDF5IO
from neo.io import NWBIO
def get_files_nwb():
    root = "D:/datasets/SpikeData/nwb_sample/"
    fls = os.listdir(root)
    fls = [root + f for f in fls]
    return fls


if __name__ == '__main__':
    file_list = get_files_nwb()
    f = file_list[2]
    # io = neo.io.NWBIO(f)
    io2 = NWBHDF5IO(f, mode="r", load_namespaces=True)
    nwb_file = io2.read()
    for item in nwb_file.children:
        print(item)
    # print(nwb_file)
    # print(nwb_file.units)
    # blk = io.read_block()
    # seg = blk.segments
    # print(type(seg))



