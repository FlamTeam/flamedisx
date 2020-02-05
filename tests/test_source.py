import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from multihist import Histdd

import flamedisx as fd
from flamedisx.er_nr_base import quanta_types


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

@pytest.fixture(params=["ER", "NR", "ER_spatial", "WIMP"])
def xes(request):
    # warnings.filterwarnings("error")
    data = pd.DataFrame([dict(s1=56., s2=2905., drift_time=143465.,
                              x=2., y=0.4, z=-20, r=2.1, theta=0.1,
                              event_time=1579784955000000000),
                         dict(s1=23, s2=1080., drift_time=445622.,
                              x=1.12, y=0.35, z=-59., r=1., theta=0.3,
                              event_time=1579784956000000000)])
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

        # Construct PDF histogram
        h = Histdd(bins=[r, theta, z], axis_names=['r', 'theta', 'z'])
        h.histogram = np.ones((nbins, nbins, nbins))

        # Calculate bin volumes for cylindrical coords (r dr dtheta)
        r_c, _, _ = h.bin_centers()
        bin_volumes = h.bin_volumes() * r_c[:, np.newaxis, np.newaxis]

        # Convert to events per bin histogram
        h.histogram *= bin_volumes

        class ERSpatial(fd.ERSource):
            spatial_rate_hist = h
            spatial_rate_bin_volumes = bin_volumes

        x = ERSpatial(data.copy(), batch_size=2, max_sigma=8)
    return x


def test_fetch(xes):
    data_tensor = xes.data_tensor[0]    # ??
    assert data_tensor is not None
    print(data_tensor.shape)
    np.testing.assert_almost_equal(
        xes._fetch('s1', data_tensor),
        xes.data['s1'].values)


def test_gimme(xes: fd.ERSource):
    x = xes.gimme('photon_gain_mean', data_tensor=None, ptensor=None)
    assert isinstance(x, tf.Tensor)
    assert x.dtype == fd.float_type()

    y = xes.gimme('photon_gain_mean', data_tensor=None, ptensor=None, numpy_out=True)
    assert isinstance(y, np.ndarray)
    if fd.float_type() == tf.float32:
        assert y.dtype == np.float32
    else:
        assert y.dtype == np.float64

    np.testing.assert_array_equal(x.numpy(), y)

    np.testing.assert_equal(
        y,
        xes.photon_gain_mean * np.ones(n_events))

    data_tensor = xes.data_tensor[0]
    assert data_tensor is not None
    print(data_tensor.shape)
    z = xes.gimme('photon_gain_mean', data_tensor=data_tensor, ptensor=None)
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
                data['%s_%s_min' % (qn, p)].values,
                data['%s_%s_mle' % (qn, p)].values + 1e-5)

            np.testing.assert_array_less(
                data['%s_%s_mle' % (qn, p)].values,
                data['%s_%s_max' % (qn, p)].values + 1e-5)


def test_nphnel(xes: fd.ERSource):
    """Test (nph, nel) rate matrix"""
    r = xes.rate_nphnel(xes.data_tensor[0],
                        xes.ptensor_from_kwargs()).numpy()
    assert r.shape == (n_events,
                       xes.dimsizes['photon_produced'],
                       xes.dimsizes['electron_produced'])


def test_domains(xes: fd.ERSource):
    n_det, n_prod = xes.cross_domains('electron_detected', 'electron_produced',
                                      xes.data_tensor[0])
    n_det = n_det.numpy()
    n_prod = n_prod.numpy()

    assert (n_det.shape == n_prod.shape
            == (n_events,
                xes.dimsizes['electron_detected'],
                xes.dimsizes['electron_produced']))

    np.testing.assert_equal(
        np.amin(n_det, axis=(1, 2)),
        np.floor(xes.data['electron_detected_min']))

    np.testing.assert_equal(
        np.amin(n_prod, axis=(1, 2)),
        np.floor(xes.data['electron_produced_min']))


def test_domain_detected(xes: fd.ERSource):
    dd = xes.domain('photon_detected').numpy()
    np.testing.assert_equal(
        dd.min(axis=1),
        np.floor(xes.data['photon_detected_min']).values)


def test_detector_response(xes: fd.ERSource):
    r = xes.detector_response('photon', xes.data_tensor[0], xes.ptensor_from_kwargs()).numpy()
    assert r.shape == (n_events, xes.dimsizes['photon_detected'])

    # r is p(S1 | detected quanta) as a function of detected quanta
    # so the sum over r isn't meaningful (as long as we're frequentists)

    # Maximum likelihood est. of detected quanta is correct
    max_is = r.argmax(axis=1)
    domain = xes.domain('photon_detected').numpy()
    found_mle = np_lookup_axis1(domain, max_is)
    np.testing.assert_array_less(
        np.abs(xes.data['photon_detected_mle'] - found_mle),
        0.5)


def test_detection_prob(xes: fd.ERSource):
    r = xes.detection_p('electron', xes.data_tensor[0], xes.ptensor_from_kwargs()).numpy()
    assert r.shape == (n_events,
                       xes.dimsizes['electron_detected'],
                       xes.dimsizes['electron_produced'])

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
    mle_is = np.round(
        xes.data['electron_produced_mle']
        - xes.data['electron_produced_min']).values.astype(np.int)
    np.testing.assert_almost_equal(
        np_lookup_axis1(rs, mle_is),
        np.ones(n_events),
        decimal=2)


def test_estimate_mu(xes: fd.ERSource):
    xes.estimate_mu()


def test_underscore_diff_rate(xes: fd.ERSource):

    x = xes._differential_rate(data_tensor=xes.data_tensor[0], ptensor=xes.ptensor_from_kwargs())
    assert isinstance(x, tf.Tensor)
    assert x.dtype == fd.float_type()

    y = xes._differential_rate(data_tensor=xes.data_tensor[0], ptensor=xes.ptensor_from_kwargs(elife=100e3))
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
    data2['s1'] *= 1.3
    data3 = pd.concat([data2, data2.iloc[:1]])

    # Setting temporarily
    with xes._set_temporarily(data2):
        np.testing.assert_array_equal(xes.data['s1'], data2['s1'])
    np.testing.assert_array_equal(xes.data['s1'], data1['s1'])

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

