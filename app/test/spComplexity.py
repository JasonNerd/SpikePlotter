# file_name:      spComplexity.py
# create_time:    2023/10/24-20:29

import neo
import quantities as pq
from elephant.statistics import Complexity


if __name__ == '__main__':
    sampling_rate = 1 / pq.ms
    st1 = neo.SpikeTrain([1, 4, 5, 8, 9] * pq.ms, t_stop=10.0 * pq.ms)
    st2 = neo.SpikeTrain([1, 2, 4, 6, 8] * pq.ms, t_stop=10.0 * pq.ms)
    sts = [st1, st2]
    cpx = Complexity(sts, sampling_rate=sampling_rate, spread=1)
    print(cpx.complexity_histogram)
    print(cpx.time_histogram.flatten())



