import flamedisx as fd
import numpy as np

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
    # TODO: should set seed...
    d2 = st.simulate(int(1e6))
    mh2 = Histdd(d2, axis_names=['s1', 's2'], bins=mh.bin_edges)
    assert np.abs(np.mean(
        (mh2.histogram - mh.histogram)
        /(mh.histogram + mh2.histogram + 0.1))) < 0.1

    # Total events should equal the histogram sum
    assert np.isclose(st.estimate_mu().numpy(), mh.n)
