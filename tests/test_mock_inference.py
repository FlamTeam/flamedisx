import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from iminuit import Minuit

import flamedisx as fd
from flamedisx.likelihood import DEFAULT_DSETNAME
from flamedisx.inference import (ScipyObjective,
                                 TensorFlowObjective,
                                 MinuitObjective)


n_events = 2

class MockObjective:
    def _inner_fun_and_grad(self, params):
        # Return -2lnL and -2lnL gradient
        # Use simple parabola with NaN bounds for rate multipliers
        # Set true parameters in self.truths[param_name] before
        ll = 0
        grads_to_return = [p for p in params if p not in self.fix.keys()]
        n_grads = len(grads_to_return)
        ll_grad_dict = {k: 0 for k in grads_to_return}

        for k, v in params.items():
            v = v.numpy()
            if k.endswith('_rate_multiplier') and v <= 0.:
                ll = float('nan')
                ll_grad = [float('nan')] * n_grads
                break
            # Add parabola
            ll += (v - self.truths[k]) ** 2
            ll_grad_dict[k] += 2 * (v - self.truths[k])
        return (tf.constant(ll, dtype=fd.float_type()),
                tf.constant(list(ll_grad_dict.values()), dtype=fd.float_type()))


class MockScipyObjective(MockObjective, ScipyObjective):
    pass


class MockTensorFlowObjective(MockObjective, TensorFlowObjective):
    pass


class MockMinuitObjective(MockObjective, MinuitObjective):
    pass


def test_mock_bestfit_tf():
    # Test bestfit (including hessian)
    arg_names = ['er_rate_multiplier', 'elife']
    mock_opt = MockTensorFlowObjective(lf=None,
                                       arg_names=arg_names,
                                       fix=dict(),
                                       autograph=True,
                                       nan_val=float('nan'))

    truths = dict(er_rate_multiplier=2.,
                  elife=2.)
    truth_test = np.array([truths[k] for k in arg_names])
    mock_opt.truths = truths
    x_guess = np.array([1., 1.])  # optmizer start

    res = mock_opt.minimize(x_guess,
                            get_lowlevel_result=False,
                            # We haven't defined a mock grad2
                            use_hessian=False,
                            llr_tolerance=0.1)

    res_test = np.array([res[k] for k in arg_names])

    np.testing.assert_array_almost_equal(truth_test,
                                         res_test, decimal=3)


def test_mock_bestfit_minuit():
    # Test bestfit (including hessian)
    arg_names = ['er_rate_multiplier', 'elife']
    mock_opt = MockMinuitObjective(lf=None,
                                   arg_names=arg_names,
                                   fix=dict(),
                                   autograph=True,
                                   nan_val=float('nan'))

    truths = dict(er_rate_multiplier=2.,
                  elife=2.)
    truth_test = np.array([truths[k] for k in arg_names])
    mock_opt.truths = truths
    x_guess = np.array([1., 1.])

    res = mock_opt.minimize(x_guess,
                            get_lowlevel_result=False,
                            use_hessian=True,
                            llr_tolerance=0.001)

    res_test = np.array([res[k] for k in arg_names])

    np.testing.assert_array_almost_equal(truth_test,
                                         res_test, decimal=3)


def test_mock_bestfit_scipy():
    # Test bestfit (including hessian)
    arg_names = ['er_rate_multiplier', 'elife']
    mock_opt = MockScipyObjective(lf=None,
                                  arg_names=arg_names,
                                  fix=dict(),
                                  autograph=True,
                                  nan_val=float('nan'))

    truths = dict(er_rate_multiplier=2.,
                  elife=2.)
    truth_test = np.array([truths[k] for k in arg_names])
    mock_opt.truths = truths
    x_guess = np.array([1., 1.])

    res = mock_opt.minimize(x_guess,
                            get_lowlevel_result=False,
                            use_hessian=True,
                            llr_tolerance=0.1)

    res_test = np.array([res[k] for k in arg_names])

    np.testing.assert_array_almost_equal(truth_test,
                                         res_test, decimal=3)
