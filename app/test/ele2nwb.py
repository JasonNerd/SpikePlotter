# file_name:      ele2nwb.py
# create_time:    2023/10/22-21:25

import neo
import os

def get_files_nwb():
    root = "D:/datasets/SpikeData/nwb_sample/"
    fls = os.listdir(root)
    fls = [root + f for f in fls]
    return fls


if __name__ == '__main__':
    file_list = get_files_nwb()
    f = file_list[2]
    io = neo.io.NWBIO(f)
    blk = io.read_block()
    seg = blk.segments
    print(type(seg))



