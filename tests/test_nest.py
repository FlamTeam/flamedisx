import numpy as np
import pandas as pd

from .test_source import dummy_data


def test_nest_source():
    # Just import and initialize without data
    import flamedisx.nest as fd_nest
    fd_nest.nestERSource()

    # Initialize with data
    df_test = dummy_data()
    s = fd_nest.nestERSource(df_test, batch_size=2)

    # Simulate events
    d_sim = s.simulate(1000)
    assert isinstance(d_sim, pd.DataFrame)
    assert len(d_sim) > 0

    # Mu estimation (based on simulation)
    s.estimate_mu()

    # Differential rate
    dr = s.differential_rate(s.data_tensor[0])
    assert len(dr) == len(df_test)
    assert min(dr) > 0

    # NEST sources have been checked to match NEST.
    # If there are any changes that cause them to output different values,
    # we have to check they still match.
    # This test prevents any such changes from passing unless someone
    # manually updates the values below.
    np.testing.assert_almost_equal(dr.numpy(), [0.00723391, 0.01815869])
