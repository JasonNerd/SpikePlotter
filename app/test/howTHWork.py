# file_name:      howTHWork.py
# create_time:    2023/10/25-9:44
"""
time_histogram 可以接收很多类型的参数, 这是如何做到的呢?
"""
import elephant
from neo import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import time_histogram
# elephant.conversion.BinnedSpikeTrain
# 1. 先生成一些 spike-train
spa = SpikeTrain([0.12, 0.15, 0.23, 1.7, 3.8, 7.2, 7.33, 7.5, 7.62, 9.2]*ms, 10*ms)
spb = SpikeTrain([0.12, 2.15, 2.23, 2.7, 2.8, 8.2, 9.33, 9.5, 9.62]*ms, 10*ms)
spc = SpikeTrain([1.12, 1.15, 1.23, 1.7, 1.8, 4.2, 4.33, 4.5, 8.62, 9.2]*ms, 10*ms)
spList = [spa, spb, spc]

# 2. 生成 time_histogram
th = time_histogram(spList, bin_size=0.5*ms)
print(th.flatten())
print("... ...")

def adder(a, b):
    """
    add a and b, return the result

    Parameters
    ----------
    a: int or float
    b: int or float

    Returns
    -------
    int or float
        a number representing the result of a+b

    Examples
    --------
    >>> adder(2, 3.5)
    >>> 5.5
    """
    return a+b


if __name__ == '__main__':
    print(adder(3, 7.2))
