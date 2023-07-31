import flamedisx as fd
import numpy as np
import pandas as pd
from scipy import interpolate

from multihist import Histdd


def test_template():
    # Create a template source from 1e6 ERSource events
    s = fd.ERSource()
    d = s.simulate(int(1e6))
    mh = Histdd(d['s1'], d['s2'], bins=30, axis_names=['s1', 's2'])
    st = fd.TemplateSource(mh, events_per_bin=True, batch_size=int(1e3))

    # Test differential rate equals the result from a simple lookup
    # (in the differential rate histogram, i.e. divide hist by bin volumes)
    d = st.simulate(1020)
    st.set_data(d)
    assert np.allclose(
        st.batched_differential_rate(),
        (mh / mh.bin_volumes()).lookup(d['s1'], d['s2']))

    # Simulate data from the template source,
    # ensuring the distribution remains similar
    # (Could set the seed here, but it shouldn't depend on it anyway)
    d2 = st.simulate(int(1e6))
    mh2 = Histdd(d2, axis_names=['s1', 's2'], bins=mh.bin_edges)
    assert np.abs(np.mean(
        (mh2.histogram - mh.histogram)
        /(mh.histogram + mh2.histogram + 0.1))) < 0.1

    # Total events should equal the histogram sum
    assert np.isclose(st.estimate_mu().numpy(), mh.n)


def test_template_interpolation():
    """Test linear interpolation of a template source"""

    # Simple 2d histogram
    mh = Histdd.from_histogram(
        histogram=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        bin_edges=[np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3])],
        axis_names=('x', 'y'),
    )

    # Events / points to interpolate
    data = pd.DataFrame({'x': [0.5, 1.5, 3.5], 'y': [0.5, 0.5, 2.5]})

    # Interpolate using flamedisx
    s = fd.TemplateSource(mh, interp_2d=True)
    s.set_data(data)
    dr_flamedisx = s.batched_differential_rate()

    # Interpolate using scipy
    # RegularGridInterpolator expects (n_points, n_dims) array
    z = np.stack([data['x'].values, data['y'].values]).T
    dr_itp = interpolate.RegularGridInterpolator(
        points=(mh.bin_centers(0), mh.bin_centers(1)),
        values=mh.histogram,
        method='linear')(z)
    assert np.allclose(dr_flamedisx, dr_itp)
