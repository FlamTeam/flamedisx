import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interpn

import flamedisx as fd


def mu_func(x=0, y=0):
    return 42 + x + 1.2 * y + 0.5 * x * y


class MuTestSource(fd.Source):
    """Source for testing mu estimation"""

    model_functions = ('our_mu',)

    def mu_before_efficiencies(self, **params):
        return self.our_mu(**params)

    def estimate_mu(self, n_trials=None, **params):
        # Let this source have perfect efficiency
        # so we don't need MC trials
        return self.mu_before_efficiencies(**params)

    def _differential_rate(self, data_tensor, ptensor):
        # Assume constant diff rates
        return tf.ones(len(data_tensor))

    def our_mu(self, x=0, y=0):
        return mu_func(x, y)

# Combining two cross interpolators
# (the same as ordinary cross interpolation)
double_cross = fd.CombinedMu.from_estimators(dict(
    x=fd.CrossInterpolatedMu,
    y=fd.CrossInterpolatedMu))

ll_options = dict(
    sources=dict(bla=MuTestSource),
    data=pd.DataFrame([]),
    x=(-1, 1),
    y=(-1, 1),
    batch_size=1)


def test_cross_interpolation():
    ll = fd.LogLikelihood(**ll_options, mu_estimators=fd.CrossInterpolatedMu)

    # Interpolate one variable
    mu_est = -ll(x=0.5)
    base_mu = mu_func(0)
    assert np.isclose(mu_est, base_mu * np.interp(
        0.5,
        [-1, 1],
        [mu_func(-1)/base_mu, mu_func(1)/base_mu]))

    # Interpolate both variables
    mu_est = -ll(x=0.5, y=0.3)
    assert np.isclose(mu_est, (
        base_mu
        * np.interp(0.5, [-1, 1], [mu_func(-1)/base_mu, mu_func(1)/base_mu])
        * np.interp(0.3, [-1, 1], [mu_func(0, -1)/base_mu, mu_func(0, 1)/base_mu])))


def test_double_cross():
    # Combining cross interpolators is the same as doing
    # one big cross interpolation
    # (because cross interpolators and CombinedEstimator both
    #  multiply independently estimated relative changes)
    ll_1 = fd.LogLikelihood(**ll_options, mu_estimators=double_cross)
    ll_2 = fd.LogLikelihood(**ll_options, mu_estimators=fd.CrossInterpolatedMu)

    assert np.isclose(ll_1(x=0.3, y=-0.2), ll_2(x=0.3, y=-0.2))


def test_constant_mu():
    # On a source with constant mu, all mu estimation methods
    # should give the same results.

    class ConstantMuSource(MuTestSource):
        def our_mu(self, x=1, y=1):
            return 42

    test_estimators = [
        fd.ConstantMu,
        fd.CrossInterpolatedMu,
        double_cross
    ]

    opts = {**ll_options}
    opts['sources'] = dict(bla=ConstantMuSource)

    for mu_est in test_estimators:
        print(f"Testing {mu_est}...")
        assert fd.LogLikelihood(**opts, mu_estimators=mu_est)() == -42
        print(f"Testing {mu_est} succeeded!")


def test_grid_interpolation():
    ll = fd.LogLikelihood(
        **ll_options,
        mu_estimators=fd.GridInterpolatedMu)

    # Interpolate both variables
    mu_est = -ll(x=0.5, y=0.3)

    points = ([-1, 0, 1], [-1, 0, 1])
    mu_grid = mu_func(*np.meshgrid(*points, indexing='ij'))

    assert np.isclose(mu_est, interpn(points, mu_grid, np.array([0.5, 0.3])))

    # Check that ordering doesn't change the result
    mu_est_xy = ll.mu_estimators['bla'](**{'x': 0.5,
                                           'y': 0.3}).numpy()
    mu_est_yx = ll.mu_estimators['bla'](**{'y': 0.3,
                                           'x': 0.5}).numpy()

    assert np.isclose(mu_est_xy, mu_est_yx)

    # Check that we agree exactly at one of the grid corners
    mu_est_corner = -ll(x=-1, y=-1)

    assert np.isclose(mu_est_corner, mu_func(-1, -1))
