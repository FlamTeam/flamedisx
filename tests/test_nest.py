import numpy as np
import pandas as pd

def dummy_data():
    return pd.DataFrame(
        [dict(s1=43., s2=3574., drift_time=65710.928173,
              x=0.82062, y=6.028471, z=44.785994, r=6.084068, theta=1.435504,
              event_time=1579784955000000000),
         dict(s1=45., s2=2308., drift_time=276544.954682,
              x=-13.22201, y=14.981184, z=13.6827, r=19.981427, theta=2.293900,
              event_time=1579784956000000000)])


def test_nest_source():
    # Just import and initialize without data
    import flamedisx.nest as fd_nest
    fd_nest.nestERSource(energy_min=8, energy_max=8, num_energies=1)

    # Initialize with data
    df_test = dummy_data()
    s = fd_nest.nestERSource(df_test, energy_min=8, energy_max=8, num_energies=1, batch_size=2)

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
        [1.9269697e-05, 4.2966261e-05],
        # For some reason, we get different values on different machines
        rtol=5e-3)
