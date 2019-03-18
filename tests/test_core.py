import warnings

import numpy as np
import pandas as pd
import pytest

from flamedisx import XenonSource
from flamedisx.core import quanta_types, _lookup_axis1


@pytest.fixture
def xes():
    warnings.filterwarnings("error")
    data = pd.DataFrame([dict(s1=20, s2=3000, drift_time=20),
                         dict(s1=2.4, s2=400, drift_time=500)])
    x = XenonSource()
    x.set_data(data)
    return x


def test_likelihood(xes: XenonSource):
    """Test that the likelihood doesn't crash"""
    xes.likelihood()


def test_bounds(xes: XenonSource):
    """Test bounds on nq_produced and _detected"""
    data = xes.data
    ##
    for qn in quanta_types:
        for p in ('produced', 'detected'):
            np.testing.assert_array_less(
                data['%s_%s_min' % (qn, p)].values,
                data['%s_%s_mle' % (qn, p)].values)

            np.testing.assert_array_less(
                data['%s_%s_mle' % (qn, p)].values,
                data['%s_%s_max' % (qn, p)].values)


def test_gimme(xes: XenonSource):
    np.testing.assert_equal(
        xes.gimme('photon_gain_mean'),
        xes.photon_gain_mean() * np.ones(xes.n_evts))


def test_nphnel(xes: XenonSource):
    """Test (nph, nel) rate matrix"""
    r = xes.rate_nphnel()
    assert r.shape == (xes.n_evts,
                       xes._dimsize('photon_produced'),
                       xes._dimsize('electron_produced'))


def test_domains(xes: XenonSource):
    n_det, n_prod = xes.cross_domains('electron_detected', 'electron_produced')

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


def test_domain_detected(xes: XenonSource):
    dd = xes.domain('photon_detected')
    np.testing.assert_equal(
        dd.min(axis=1),
        np.floor(xes.data['photon_detected_min']).values)


def test_detector_response(xes: XenonSource):
    r = xes.detector_response('photon')
    assert r.shape == (xes.n_evts, xes._dimsize('photon_detected'))

    # r is p(S1 | detected quanta) as a function of detected quanta
    # so the sum over r isn't meaningful (as long as we're frequentists)
    # Nonetheless, it sums to one... (well, not exactly,
    # since we're considering a finite domain)
    # only for this specific model, e.g. try changing 0.5 to 0.7
    # in detector_response.
    # (on the other hand changing norm -> t with 3 dof still works...)
    # I wonder why...
    np.testing.assert_almost_equal(
        r.sum(axis=1),
        np.ones(xes.n_evts), decimal=3)

    # Maximum likelihood est. of detected quanta is correct
    max_is = r.argmax(axis=1)
    domain = xes.domain('photon_detected')
    found_mle = _lookup_axis1(domain, max_is)
    np.testing.assert_array_less(
        np.abs(xes.data['photon_detected_mle'] - found_mle),
        0.5)


def test_detection_prob(xes: XenonSource):
    r = xes.detection_p('electron')
    assert r.shape == (xes.n_evts,
                       xes._dimsize('electron_detected'),
                       xes._dimsize('electron_produced'))

    # Sum of probability over detected electrons must be
    #  A) in [0, 1] for any value of electrons_produced
    #     (it would be 1 everywhere if we considered
    #      infinitely many electrons_detected values)
    rs = r.sum(axis=1)
    np.testing.assert_array_less(rs, 1 + 1e-5)
    np.testing.assert_array_less(1 - rs, 1 + 1e-5)

    # B) 1 at the MLE of electrons_produced,
    #    where all reasonably probable electrons_detected values
    #    should be probed
    mle_is = np.round(
        xes.data['electron_produced_mle']
        - xes.data['electron_produced_min']).values.astype(np.int)
    np.testing.assert_almost_equal(
        _lookup_axis1(rs, mle_is),
        np.ones(xes.n_evts),
        decimal=4)
