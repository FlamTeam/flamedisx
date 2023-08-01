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
    """Test inter-bin interpolation in a single template source"""

    # Simple 2d histogram
    mh = Histdd.from_histogram(
        histogram=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        bin_edges=[np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3])],
        axis_names=('x', 'y'),
    )

    # Events / points to interpolate
    data = pd.DataFrame({'x': [0.5, 1.2, 3.42], 'y': [0.5, 0.6, 2.3]})

    # Interpolate using flamedisx
    s = fd.TemplateSource(mh, interpolate=True)
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

    # With interpolation turned off, flamedisx just looks up the diff rates
    s = fd.TemplateSource(mh, interpolate=False)
    s.set_data(data)
    dr_flamedisx_noitp = s.batched_differential_rate()
    assert np.allclose(dr_flamedisx_noitp, mh.lookup(data['x'], data['y']))


def test_multi_template():
    """Test linear interpolation of multiple templates"""

    # Differential rate histograms, with values offset by constants.
    offsets = np.array([12, 14.3, 18.4, 3.1])
    mhs = []
    for offset in offsets:
        mhs.append(Histdd.from_histogram(
            histogram=offset + np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            bin_edges=[np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3])],
            axis_names=('x', 'y'),
        ))

    # Observable events
    data = pd.DataFrame({'x': [0.5, 1.5, 2.5], 'y': [0.5, 0.5, 2.5]})
    # diff rate values at offset=0 will be [1, 4, 9]
    base_diffrate = np.array([1, 4, 9])

    s = fd.MultiTemplateSource(
        params_and_templates=[
            ({'a': 0., 'b': 0.}, mhs[0]),
            ({'a': 0., 'b': 1.}, mhs[1]),
            ({'a': 1., 'b': 0.}, mhs[2]),
            ({'a': 1., 'b': 1.}, mhs[3]),
        ],
        interpolate=False)
    s.set_data(data)

    ##
    # Test differential rate interpolation
    ##

    # Default values are those of the first template
    np.testing.assert_allclose(
        s.batched_differential_rate(),
        base_diffrate + offsets[0])

    # at (a=0, b=0.5), get average of offset 0 and 1
    np.testing.assert_allclose(
        s.batched_differential_rate(b=0.5),
        base_diffrate + (offsets[0] + offsets[1])/2)

    # at (a=0.5, b=0), get average of offset 0 and 2
    np.testing.assert_allclose(
        s.batched_differential_rate(a=0.5),
        base_diffrate + (offsets[0] + offsets[2])/2)

    # at (a=0.5, b=0.5), get average of all offsets
    np.testing.assert_allclose(
        s.batched_differential_rate(a=0.5, b=0.5),
        base_diffrate + offsets.mean())

    ##
    # Test mu interpolation
    ##
    np.testing.assert_allclose(
        s.estimate_mu(),
        s._templates[0].mu)

    np.testing.assert_allclose(
        s.estimate_mu(a=0, b=0.5),
        (s._templates[0].mu + s._templates[1].mu)/2)

    np.testing.assert_allclose(
        s.estimate_mu(a=0.5, b=0),
        (s._templates[0].mu + s._templates[2].mu)/2)
