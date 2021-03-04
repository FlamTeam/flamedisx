import numpy as np
import pandas as pd
import wimprates as wr
import flamedisx as fd


def test_j2000_conversion():
    j2000_times = np.linspace(0., 10000., 100)

    # convert j2000 -> event_time -> j2000
    test_times = wr.j2000(fd.j2000_to_event_time(j2000_times))

    np.testing.assert_array_almost_equal(j2000_times,
                                         test_times,
                                         decimal=6)
