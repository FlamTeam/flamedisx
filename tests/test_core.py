import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import flamedisx as fd
from flamedisx.source import quanta_types


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

@pytest.fixture(params=["ER", "NR"])
def xes(request):
    # warnings.filterwarnings("error")
    data = pd.DataFrame([dict(s1=20., s2=3000., drift_time=20.,
                              x=0., y=0, z=-5., r=0., theta=0),
                         dict(s1=2.4, s2=400., drift_time=500.,
                              x=0., y=0., z=-50., r=0., theta=0.)])
    if request.param == 'ER':
        x = fd.ERSource(data.copy(), n_batches=2, max_sigma=5)
    else:
        x = fd.NRSource(data.copy(), n_batches=2, max_sigma=5)
    return x


def test_simulate(xes: fd.ERSource):
    """Test the simulator doesn't crash"""
    xes.simulate(data=xes.data, energies=np.linspace(0., 100., int(1e3)))


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


def test_gimme(xes: fd.ERSource):
    x = xes.gimme('photon_gain_mean')
    assert isinstance(x, tf.Tensor)
    assert x.dtype == tf.float32

    y = xes.gimme('photon_gain_mean', numpy_out=True)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32

    np.testing.assert_array_equal(x.numpy(), y)

    np.testing.assert_equal(
        y,
        xes.photon_gain_mean * np.ones(n_events))


def test_nphnel(xes: fd.ERSource):
    """Test (nph, nel) rate matrix"""
    r = xes.rate_nphnel().numpy()
    assert r.shape == (n_events,
                       xes.dimsizes['photon_produced'],
                       xes.dimsizes['electron_produced'])


def test_domains(xes: fd.ERSource):
    n_det, n_prod = xes.cross_domains('electron_detected', 'electron_produced')
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
    r = xes.detector_response('photon').numpy()
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
    r = xes.detection_p('electron').numpy()
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
        decimal=4)


def test_estimate_mu(xes: fd.ERSource):
    xes.estimate_mu(xes.data)


def test_diff_rate(xes: fd.ERSource):
    """Test differential_rate give the same answer
    whether it is batched or not"""

    # Need very high sigma for this
    # so extending the bounds due to not-batching does not
    # matter anymore
    y = xes.differential_rate(i_batch=None)
    y2 = np.concatenate([
        fd.tf_to_np(xes.differential_rate(i_batch=batch_i))
        for batch_i in range(xes.n_batches)])
    np.testing.assert_array_equal(y.numpy(), y2)

def test_inference(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        data=xes.data)

    # Test eager version
    y1 = lf.log_likelihood(fd.np_to_tf(np.array([200e3,])))

    # # Test graph version
    # print("GRAPH MODE TEST NOW")
    # y2 = lf.log_likelihood(fd.np_to_tf(np.array([200e3, ])))
    # np.testing.assert_array_equal(y1, y1)
    #
    # # TODO: test fit and hessian
