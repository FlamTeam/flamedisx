import numpy as np

import flamedisx as fd
from test_source import xes   # Yes, it is used through pytest magic


n_events = 2


def test_one_parameter_interval_nonlincontr(xes):
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

    # First find global best so we can check intervals
    bestfit = lf.bestfit(guess,
                         optimizer='scipy')

    ul = lf.limit('er_rate_multiplier', bestfit,
                  optimizer='nlin',
                  confidence_level=0.9, kind='upper')
    assert ul > bestfit['er_rate_multiplier']

    ll = lf.limit('er_rate_multiplier', bestfit,
                  optimizer='nlin',
                  confidence_level=0.9, kind='lower')
    assert ll < bestfit['er_rate_multiplier']

    ll, ul = lf.limit('er_rate_multiplier', bestfit,
                      optimizer='nlin',
                      confidence_level=0.9, kind='central')
    assert ll < bestfit['er_rate_multiplier'] < ul

    # Test fixed parameter
    fix = dict(elife=bestfit['elife'])

    ul = lf.limit('er_rate_multiplier', bestfit, fix=fix,
                  optimizer='nlin',
                  confidence_level=0.9, kind='upper')
    assert bestfit['er_rate_multiplier'] < ul


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

    # First find global best so we can check intervals
    bestfit = lf.bestfit(guess,
                         optimizer='scipy')

    ul = lf.limit('er_rate_multiplier', bestfit,
                  confidence_level=0.9, kind='upper')
    assert ul > bestfit['er_rate_multiplier']

    ll = lf.limit('er_rate_multiplier', bestfit,
                  confidence_level=0.9, kind='lower')
    assert ll < bestfit['er_rate_multiplier']

    ll, ul = lf.limit('er_rate_multiplier', bestfit,
                      confidence_level=0.9, kind='central')
    assert ll < bestfit['er_rate_multiplier'] < ul

    # Test fixed parameter
    fix = dict(elife=bestfit['elife'])

    ul = lf.limit('er_rate_multiplier', bestfit, fix=fix,
                  confidence_level=0.9, kind='upper')
    assert bestfit['er_rate_multiplier'] < ul


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
                         use_hessian=False)
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
