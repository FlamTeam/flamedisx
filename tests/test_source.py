from datetime import timedelta
import warnings

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from wimprates import j2000

from multihist import Histdd

import flamedisx as fd
quanta_types = ('photon', 'electron')

o = tf.newaxis


def np_lookup_axis1(x, indices, fill_value=0):
    """Return values of x at indices along axis 1,
    returning fill_value for out-of-range indices"""
    d = indices
    imax = x.shape[1]
    mask = d >= imax
    d[mask] = 0
    result = np.take_along_axis(
            x,
            d.reshape(len(d), -1), axis=1
        ).reshape(d.shape)
    result[mask] = fill_value
    return result


n_events = 2


def dummy_data():
    return pd.DataFrame(
        [dict(s1=56., s2=2905., drift_time=143465.,
              x=2., y=0.4, z=-20, r=2.1, theta=0.1,
              event_time=1579784955000000000),
         dict(s1=23, s2=1080., drift_time=445622.,
              x=1.12, y=0.35, z=-59., r=1., theta=0.3,
              event_time=1579784956000000000)])


@pytest.fixture(params=["ER", "NR", "ER_spatial", "WIMP"])
def xes(request):
    # warnings.filterwarnings("error")
    data = dummy_data()
    if request.param == 'ER':
        x = fd.ERSource(data.copy(), batch_size=2, max_sigma=8)
    elif request.param == 'NR':
        x = fd.NRSource(data.copy(), batch_size=2, max_sigma=8)
    elif request.param == 'WIMP':
        x = fd.WIMPSource(data.copy(), batch_size=2, max_sigma=8)
    elif request.param == 'ER_spatial':
        nbins = 100
        r = np.linspace(0, 47.9, nbins + 1)
        z = np.linspace(-97.6, 0, nbins + 1)
        theta = np.linspace(0, 2 * np.pi, nbins + 1)

        # Construct histogram corresponding to a uniform source
        # (since the tests expect this)
        # number of events ~ bin volume ~ r
        h = Histdd(bins=[r, theta, z], axis_names=['r', 'theta', 'z'])
        h.histogram = h.histogram * 0 + h.bin_centers('r')[:, None, None]

        class ERSpatial(fd.SpatialRateERSource):
            spatial_hist = h

        x = ERSpatial(data.copy(), batch_size=2, max_sigma=8)
    return x


def test_test(xes):
    # All size tests written assuming we have just one batch: due to variable
    # tensor sizes! (determined per batch)
    assert xes.n_batches == 1


def test_fetch(xes):
    data_tensor = xes.data_tensor[0]    # ??
    assert data_tensor is not None
    print(data_tensor.shape)
    np.testing.assert_almost_equal(
        xes._fetch('s1', data_tensor),
        xes.data['s1'].values)


def test_gimme(xes: fd.ERSource):
    x = xes.gimme('photoelectron_gain_mean',
                  data_tensor=None, ptensor=None)
    assert isinstance(x, tf.Tensor)
    assert x.dtype == fd.float_type()

    y = xes.gimme('photoelectron_gain_mean',
                  data_tensor=None, ptensor=None, numpy_out=True)
    assert isinstance(y, np.ndarray)
    if fd.float_type() == tf.float32:
        assert y.dtype == np.float32
    else:
        assert y.dtype == np.float64

    np.testing.assert_array_equal(x.numpy(), y)

    # This assumes photoelectron_gain_mean is a scalar, not a function
    np.testing.assert_equal(
        y,
        xes.photoelectron_gain_mean * np.ones(n_events))

    data_tensor = xes.data_tensor[0]
    assert data_tensor is not None
    print(data_tensor.shape)
    z = xes.gimme('photoelectron_gain_mean',
                  data_tensor=data_tensor, ptensor=None)
    assert isinstance(z, tf.Tensor)
    assert z.dtype == fd.float_type()
    assert tf.reduce_all(tf.equal(x, z))


def test_simulate(xes: fd.ERSource):
    """Test the simulator with and without fix_truth"""
    n_ev = int(1e3)

    # Test simulate with number of events
    simd = xes.simulate(n_ev)
    assert len(simd) <= n_ev

    # Test n_events etc. have not been changed
    assert xes.n_events == 2
    assert len(xes.data) == 2
    assert xes.n_batches == 1

    # Test simulate with fix_truth DataFrame
    fix_truth_df = simd.iloc[:1].copy()
    simd = xes.simulate(n_ev, fix_truth=fix_truth_df)

    # Check if all 'x' values are the same
    assert len(set(simd['x'].values)) == 1
    assert simd['x'].values[0] == fix_truth_df['x'].values[0]

    # Test simulate with fix_truth dict
    e_test = 5.
    fix_truth = dict(energy=e_test)
    simd = xes.simulate(n_ev, fix_truth=fix_truth)

    # Check if all energies are the same fixed value
    assert 'energy' in simd
    assert len(set(simd['energy'].values)) == 1
    assert simd['energy'].values[0] == e_test


def test_bounds(xes: fd.ERSource):
    """Test bounds on nq_produced and _detected"""
    data = xes.data
    ##
    for qn in quanta_types:
        for p in ('produced', 'detected'):
            print(qn + '_' + p)
            np.testing.assert_array_less(
                data['%ss_%s_min' % (qn, p)].values,
                data['%ss_%s_mle' % (qn, p)].values + 1e-5)

            np.testing.assert_array_less(
                data['%ss_%s_mle' % (qn, p)].values,
                data['%ss_%s_max' % (qn, p)].values + 1e-5)


def test_domains(xes: fd.ERSource):
    n_det, n_prod = xes.cross_domains('electrons_detected', 'electrons_produced',
                                      xes.data_tensor[0])
    n_det = n_det.numpy()
    n_prod = n_prod.numpy()

    assert (n_det.shape == n_prod.shape
            == (n_events,
                max(xes.dimsizes['electrons_detected']),
                max(xes.dimsizes['electrons_produced'])))

    np.testing.assert_equal(
        np.amin(n_det, axis=(1, 2)),
        np.floor(xes.data['electrons_detected_min']))

    np.testing.assert_equal(
        np.amin(n_prod, axis=(1, 2)),
        np.floor(xes.data['electrons_produced_min']))


def test_domain_detected(xes: fd.ERSource):
    dd = xes.domain('photons_detected').numpy()
    np.testing.assert_equal(
        dd.min(axis=1),
        np.floor(xes.data['photons_detected_min']).values)


def test_detector_response(xes: fd.ERSource):
    data_tensor, ptensor = xes.data_tensor[0], xes.ptensor_from_kwargs()

    for block in fd.MakeS1, fd.MakeS2:

        r = block(xes).compute(
            data_tensor, ptensor,
            **xes._domain_dict(block.dimensions, data_tensor))
        r = r.numpy()

        quanta_name = block.quanta_name
        assert r.shape == \
               (n_events, max(xes.dimsizes[quanta_name + 's_detected']), 1)
        r = r[:, :, 0]

        # r is p(S1 | detected electrons) as a function of detected electrons
        # so the sum over r isn't meaningful (as long as we're frequentists)

        # Maximum likelihood est. of detected quanta is correct
        max_is = r.argmax(axis=1)
        domain = xes.domain(quanta_name + 's_detected').numpy()
        found_mle = np_lookup_axis1(domain, max_is)
        np.testing.assert_array_less(
            np.abs(xes.data[quanta_name + 's_detected_mle'] - found_mle),
            0.5)


def test_detection_prob(xes: fd.ERSource):
    data_tensor, ptensor = xes.data_tensor[0], xes.ptensor_from_kwargs()
    block = fd.DetectElectrons
    r = block(xes).compute(
            data_tensor, ptensor,
            **xes._domain_dict(block.dimensions, data_tensor)).numpy()

    assert r.shape == (n_events,
                       max(xes.dimsizes['electrons_produced']),
                       max(xes.dimsizes['electrons_detected']))

    # Test below was written assuming a (batch, detected, produced) tensor
    r = np.transpose(r, [0, 2, 1])
    # We need to weight the block by the electrons detected steppings
    steps = xes._fetch('electrons_detected_steps', data_tensor=data_tensor)
    step_mul = tf.repeat(steps[:,o], tf.shape(r)[1], axis=1)
    step_mul = tf.repeat(step_mul[:,:,o],
    tf.shape(r)[2], axis=2)
    r *= step_mul.numpy()

    # Sum of probability over detected electrons must be
    #  A) in [0, 1] for any value of electrons_produced
    #     (it would be 1 everywhere if we considered
    #      infinitely many electrons_detected values)
    # TODO: this holds to 1e-4... is that enough?
    rs = r.sum(axis=1)
    np.testing.assert_array_less(rs, 1 + 1e-4)
    np.testing.assert_array_less(1 - rs, 1 + 1e-4)

    # B) 1 at the MLE of electrons_produced,
    #    where all reasonably probable electrons_detected values
    #    should be probed
    # Account for stepping when finding the mle
    mle_is = np.round(
        (xes.data['electrons_produced_mle']
        - xes.data['electrons_produced_min']) /
        xes.data['electrons_produced_steps']).values.astype(np.int)
    np.testing.assert_almost_equal(
        np_lookup_axis1(rs, mle_is),
        np.ones(n_events),
        decimal=2)


def test_estimate_mu(xes: fd.ERSource):
    xes.estimate_mu()


def test_underscore_diff_rate(xes: fd.ERSource):

    x = xes._differential_rate(data_tensor=xes.data_tensor[0],
                               ptensor=xes.ptensor_from_kwargs())
    assert isinstance(x, tf.Tensor)
    assert x.dtype == fd.float_type()

    y = xes._differential_rate(data_tensor=xes.data_tensor[0],
                               ptensor=xes.ptensor_from_kwargs(elife=100e3))
    np.testing.assert_array_less(-fd.tf_to_np(tf.abs(x - y)), 0)


def test_diff_rate_grad(xes):
    # Test low-level version
    ptensor = xes.ptensor_from_kwargs()
    dr = xes._differential_rate(xes.data_tensor[0], ptensor)
    dr = dr.numpy()
    assert dr.shape == (xes.n_events,)

    # Test eager/wrapped version
    dr2 = xes.differential_rate(xes.data_tensor[0], autograph=False)
    dr2 = dr2.numpy()
    np.testing.assert_almost_equal(dr, dr2)

    # Test traced version
    # TODO: currently small discrepancy due to float32/float64!
    # Maybe due to weird events / poor bounds est
    # Check with real data
    dr3 = xes.differential_rate(xes.data_tensor[0], autograph=True)
    dr3 = dr3.numpy()
    np.testing.assert_almost_equal(dr, dr3, decimal=4)


def test_set_data(xes: fd.ERSource):
    assert xes.n_batches == 1
    assert xes.n_padding == 0
    assert xes.batch_size == 2

    x = xes.batched_differential_rate()
    assert x.shape == (2,)

    data1 = xes.data
    data2 = pd.concat([data1.copy(),
                       data1.iloc[:1].copy()])
    data2['s1'] *= 0.9
    data3 = pd.concat([data2, data2.iloc[:1]])

    # Setting temporarily
    with xes._set_temporarily(data2):
        np.testing.assert_array_equal(xes.data['s1'], data2['s1'])
    np.testing.assert_array_equal(xes.data['s1'], data1['s1'])

    # Setting defaults temporarily (see PR #110)
    with xes._set_temporarily(data2, elife=100e3):
        pass
    assert xes.defaults['elife'] == fd.ERSource().defaults['elife']

    # Setting for real
    xes.set_data(data2)
    assert xes.data is not data1
    np.testing.assert_array_equal(
        xes.data['s1'].values,
        data3['s1'].values)

    np.testing.assert_almost_equal(
        xes._fetch('s1', data_tensor=xes.data_tensor[0]).numpy(),
        data2['s1'].values[:2].astype('float32'))

    # Test batching stuff has been updated
    assert xes.n_batches == 2
    assert xes.n_padding == 1
    assert xes.batch_size == 2

    x = xes.batched_differential_rate()
    assert x.shape == (3,)


def test_clip(xes):
    if not isinstance(xes, fd.WIMPSource):
        return

    t_bad = pd.to_datetime(xes.t_start) - timedelta(days=2)
    with pytest.raises(fd.InvalidEventTimes):
        xes.simulate(10, fix_truth=dict(event_time=t_bad))

    t_good = pd.to_datetime(xes.t_start) + timedelta(days=2)
    assert xes.model_blocks[0].clip_j2000_times(j2000(t_good)) == j2000(t_good)
    xes.simulate(10, fix_truth=dict(event_time=t_good))


def test_config(xes):
    # Test the use of config files to set source attributes
    xes.set_defaults(config='example')
    assert xes.defaults['elife'].numpy() == 987_654.
    assert xes.photon_detection_eff == 0.11

    # Trying to set unused attributes should trigger a warnings
    with warnings.catch_warnings(record=True) as w:
        xes.set_defaults(grumbl=42)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)

    # Blocks should also see the attributes changed
    b = [_b for _b in xes.model_blocks
         if isinstance(_b, fd.DetectPhotons)][0]
    assert b.photon_detection_eff == 0.11
