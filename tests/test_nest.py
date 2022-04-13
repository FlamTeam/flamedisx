import numpy as np
import pandas as pd

def dummy_data():
    return pd.DataFrame(
        [dict(s1=43., s2=3574., drift_time=65710.930108,
              x=0.82062, y=6.028471, z=44.785994, r=6.084068, theta=1.435504,
              event_time=1579784955000000000),
         dict(s1=45., s2=2308., drift_time=55682.601147,
              x=-13.22201, y=14.981184, z=-46.265423, r=19.981427, theta=2.293900,
              event_time=1579784956000000000)])


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
    np.testing.assert_allclose(
        dr.numpy(),
        [0.00287317, 0.00692382],
        # For some reason, we get different values on different machines
        rtol=5e-3)
