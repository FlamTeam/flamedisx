import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from iminuit import Minuit

import flamedisx as fd
from flamedisx.likelihood import DEFAULT_DSETNAME


n_events = 2

@pytest.fixture(params=["ER", "NR"])
def xes(request):
    # warnings.filterwarnings("error")
    data = pd.DataFrame([dict(s1=56., s2=2905., drift_time=143465.,
                              x=2., y=0.4, z=-20, r=2.1, theta=0.1),
                         dict(s1=23, s2=1080., drift_time=445622.,
                              x=1.12, y=0.35, z=-59., r=1., theta=0.3)])
    if request.param == 'ER':
        x = fd.ERSource(data.copy(), batch_size=2, max_sigma=8)
    else:
        x = fd.NRSource(data.copy(), batch_size=2, max_sigma=8)
    return x


def test_one_parameter_interval(xes):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)

    guess = lf.guess()
    # Set reasonable rate
    # Evaluate the likelihood curve around the minimum
    xs_er = np.linspace(0.001, 0.004, 20)  # ER source range
    xs_nr = np.linspace(0.04, 0.1, 20)  # NR source range
    xs = list(xs_er) + list(xs_nr)
    ys = np.array([-lf(er_rate_multiplier=x) for x in xs])
    guess['er_rate_multiplier'] = xs[np.argmin(ys)]
    assert len(guess) == 2

    # TODO remove equality tests below, limit stuck at bounds

    # First find global best so we can check intervals
    bestfit = lf.bestfit(guess, optimizer='scipy')

    ul = lf.one_parameter_interval('er_rate_multiplier', guess,
                                   confidence_level=0.9, kind='upper')
    assert ul >= bestfit['er_rate_multiplier']

    ll = lf.one_parameter_interval('er_rate_multiplier', guess,
                                   confidence_level=0.9, kind='lower')
    assert ll <= bestfit['er_rate_multiplier']

    ll, ul = lf.one_parameter_interval('er_rate_multiplier', guess,
                                       confidence_level=0.9, kind='central')
    assert (ll <= bestfit['er_rate_multiplier']
            and ul >= bestfit['er_rate_multiplier'])

    # Test fixed parameter
    fix = dict(elife=bestfit['elife'])

    ul = lf.one_parameter_interval('er_rate_multiplier', guess, fix=fix,
                                   confidence_level=0.9, kind='upper')


def test_bestfit_tf(xes):
    # Test bestfit (including hessian)
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)

    guess = lf.guess()
    # Set reasonable rate
    # Evaluate the likelihood curve around the minimum
    xs_er = np.linspace(0.001, 0.004, 20)  # ER source range
    xs_nr = np.linspace(0.04, 0.1, 20)  # NR source range
    xs = list(xs_er) + list(xs_nr)
    ys = np.array([-lf(er_rate_multiplier=x) for x in xs])
    guess['er_rate_multiplier'] = xs[np.argmin(ys)]
    assert len(guess) == 2

    bestfit = lf.bestfit(guess, optimizer='tfp', use_hessian=True)
    assert isinstance(bestfit, dict)
    assert len(bestfit) == 2
    assert bestfit['er_rate_multiplier'].dtype == np.float32


def test_bestfit_minuit(xes):
    # Test bestfit (including hessian)
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)

    guess = lf.guess()
    # Set reasonable rate
    # Evaluate the likelihood curve around the minimum
    xs_er = np.linspace(0.001, 0.004, 20)  # ER source range
    xs_nr = np.linspace(0.04, 0.1, 20)  # NR source range
    xs = list(xs_er) + list(xs_nr)
    ys = np.array([-lf(er_rate_multiplier=x) for x in xs])
    guess['er_rate_multiplier'] = xs[np.argmin(ys)]
    assert len(guess) == 2

    bestfit = lf.bestfit(guess, optimizer='minuit',
                         return_errors=True,
                         error=(0.0001, 1000))
    assert isinstance(bestfit[0], dict)
    assert len(bestfit[0]) == 2


def test_bestfit_scipy(xes):
    # Test bestfit (including hessian)
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)

    guess = lf.guess()
    # Set reasonable rate
    # Evaluate the likelihood curve around the minimum
    xs_er = np.linspace(0.001, 0.004, 20)  # ER source range
    xs_nr = np.linspace(0.04, 0.1, 20)  # NR source range
    xs = list(xs_er) + list(xs_nr)
    ys = np.array([-lf(er_rate_multiplier=x) for x in xs])
    guess['er_rate_multiplier'] = xs[np.argmin(ys)]
    assert len(guess) == 2

    bestfit = lf.bestfit(guess, optimizer='scipy')
    assert isinstance(bestfit, dict)
    assert len(bestfit) == 2
