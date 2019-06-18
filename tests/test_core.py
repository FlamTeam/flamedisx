import warnings

import numpy as np
import pandas as pd
import pytest

from flamedisx import ERSource, NRSource
from flamedisx.core import quanta_types


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


@pytest.fixture(params=["ER", "NR"])
def xes(request):
    warnings.filterwarnings("error")
    data = pd.DataFrame([dict(s1=20, s2=3000, drift_time=20),
                         dict(s1=2.4, s2=400, drift_time=500)])
    if request.param == 'ER':
        x = ERSource()
    else:
        x = NRSource()
    x.set_data(data)
    return x


def test_likelihood(xes: ERSource):
    """Test that the likelihood doesn't crash"""
    xes.likelihood()


def test_simulate(xes: ERSource):
    """Test the simulator doesn't crash"""
    xes.simulate(energies=np.linspace(0., 100., int(1e3)))


def test_bounds(xes: ERSource):
    """Test bounds on nq_produced and _detected"""
    data = xes.data
    ##
    for qn in quanta_types:
        for p in ('produced', 'detected'):
            print(qn + '_' + p)
            np.testing.assert_array_less(
                data['%s_%s_min' % (qn, p)].values,
                data['%s_%s_mle' % (qn, p)].values)

            np.testing.assert_array_less(
                data['%s_%s_mle' % (qn, p)].values,
                data['%s_%s_max' % (qn, p)].values)


def test_gimme(xes: ERSource):
    np.testing.assert_equal(
        xes.gimme('photon_gain_mean').numpy(),
        xes.photon_gain_mean * np.ones(xes.n_evts))


def test_nphnel(xes: ERSource):
    """Test (nph, nel) rate matrix"""
    r = xes.rate_nphnel().numpy()
    assert r.shape == (xes.n_evts,
                       xes._dimsize('photon_produced'),
                       xes._dimsize('electron_produced'))


def test_domains(xes: ERSource):
    n_det, n_prod = xes.cross_domains('electron_detected', 'electron_produced')
    n_det = n_det.numpy()
    n_prod = n_prod.numpy()

    assert (n_det.shape == n_prod.shape
            == (xes.n_evts,
                xes._dimsize('electron_detected'),
                xes._dimsize('electron_produced')))

    np.testing.assert_equal(
        np.amin(n_det, axis=(1, 2)),
        np.floor(xes.data['electron_detected_min']))

    np.testing.assert_equal(
        np.amin(n_prod, axis=(1, 2)),
        np.floor(xes.data['electron_produced_min']))


def test_domain_detected(xes: ERSource):
    dd = xes.domain('photon_detected').numpy()
    np.testing.assert_equal(
        dd.min(axis=1),
        np.floor(xes.data['photon_detected_min']).values)


def test_detector_response(xes: ERSource):
    r = xes.detector_response('photon').numpy()
    assert r.shape == (xes.n_evts, xes._dimsize('photon_detected'))

    # r is p(S1 | detected quanta) as a function of detected quanta
    # so the sum over r isn't meaningful (as long as we're frequentists)

    # Maximum likelihood est. of detected quanta is correct
    max_is = r.argmax(axis=1)
    domain = xes.domain('photon_detected').numpy()
    found_mle = np_lookup_axis1(domain, max_is)
    np.testing.assert_array_less(
        np.abs(xes.data['photon_detected_mle'] - found_mle),
        0.5)


def test_detection_prob(xes: ERSource):
    r = xes.detection_p('electron').numpy()
    assert r.shape == (xes.n_evts,
                       xes._dimsize('electron_detected'),
                       xes._dimsize('electron_produced'))

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
        np.ones(xes.n_evts),
        decimal=4)
