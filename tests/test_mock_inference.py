import numpy as np
import pytest

import flamedisx as fd

n_events = 2

arg_names = ['er_rate_multplier', 'elife']
truth_test = np.array([2., 3.])


@pytest.fixture(params=list(fd.SUPPORTED_OPTIMIZERS.values()))
def mock_objective(request):

    class MockObjective(request.param):
        def __init__(self, *, truths, **kwargs):
            self.arg_names = list(truths.keys())
            self.truths = truths
            super().__init__(**kwargs)

        def _inner_fun_and_grad(self, params):
            # Return -2lnL and -2lnL gradient
            # Use simple parabola with NaN bounds for rate multipliers
            # Set true parameters in self.truths[param_name] before
            ll = 0
            grads_to_return = [p for p in params if p not in self.fix.keys()]
            n_grads = len(grads_to_return)
            ll_grad_dict = {k: 0 for k in grads_to_return}

            for k, v in params.items():
                if k.endswith('_rate_multiplier') and v <= 0.:
                    ll = float('nan')
                    ll_grad = [float('nan')] * n_grads
                    break
                # Add parabola
                ll += (v - self.truths[k]) ** 2
                ll_grad_dict[k] += 2 * (v - self.truths[k])
            return ll, np.array(list(ll_grad_dict.values()))

    truths = {k: truth_test[i] for i, k in enumerate(arg_names)}
    guess = {k: 1. for k in arg_names}

    return MockObjective(
        lf=None,
        truths=truths,
        guess=guess,
        fix=dict(),
        nan_val=float('nan'),
        get_lowlevel_result=False,
        # We haven't defined a mock grad2
        use_hessian=False)


def test_mock_bestfit_tf(mock_objective):
    # Test bestfit (including hessian)
    res = mock_objective.minimize()
    res_test = np.array([res[k]
                         for k in mock_objective.arg_names])

    np.testing.assert_array_almost_equal(truth_test,
                                         res_test, decimal=3)
