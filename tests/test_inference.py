import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from iminuit import Minuit

import flamedisx as fd
from flamedisx.inference import DEFAULT_DSETNAME


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


def test_inference(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        data=xes.data)

    ##
    # Test non-autograph version
    ##
    x, x_grad = lf._log_likelihood(i_batch=tf.constant(0),
                                   dsetname=DEFAULT_DSETNAME,
                                   autograph=False,
                                   elife=tf.constant(200e3))
    assert isinstance(x, tf.Tensor)
    assert x.dtype == fd.float_type()
    assert x.numpy() < 0

    assert isinstance(x_grad, tf.Tensor)
    assert x_grad.dtype == fd.float_type()
    assert x_grad.numpy().shape == (1,)

    # Test a different parameter gives a different likelihood
    x2, x2_grad = lf._log_likelihood(i_batch=tf.constant(0),
                                     dsetname=DEFAULT_DSETNAME,
                                     autograph=False,
                                     elife=tf.constant(300e3))
    assert (x - x2).numpy() != 0
    assert (x_grad - x2_grad).numpy().sum() !=0

    ##
    # Test batching
    # ##
    l1 = lf.log_likelihood(autograph=False)
    l2 = lf(autograph=False)
    lf.log_likelihood(elife=tf.constant(200e3), autograph=False)


def test_multisource(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)
    l1 = lf.log_likelihood(er_rate_multiplier=2.)

    lf2 = fd.LogLikelihood(
        sources=dict(er=xes.__class__, er2=xes.__class__),
        elife=(100e3, 500e3, 5),
        data=xes.data)
    # Prevent jitter from mu interpolator simulation to fail test
    itp = lf.mu_itps['er']
    lf2.mu_itps = dict(er=itp, er2=itp)
    assert lf2.log_likelihood()[0].numpy() == l1[0].numpy()


def test_multisource_er_nr(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__, nr=fd.NRSource),
        elife=(100e3, 500e3, 5),
        data=xes.data)

    lf()


def test_columnsource(xes: fd.ERSource):
    class myColumnSource(fd.ColumnSource):
        column = "diffrate"
        mu = 3.14

    xes.data['diffrate'] = 5.

    lf = fd.LogLikelihood(
        sources=dict(muur=myColumnSource),
        data=xes.data)

    np.testing.assert_almost_equal(lf(), -3.14 + len(xes.data) * np.log(5.))


def test_multi_dset(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        data=xes.data.copy())
    ll1 = lf()

    lf2 = fd.LogLikelihood(
        sources=dict(data1=dict(er1=fd.ERSource),
                     data2=dict(er2=fd.ERSource)),
        data=dict(data1=xes.data.copy(),
                  data2=xes.data.copy()))

    # Fix interpolator nondeterminism
    itp = lf.mu_itps['er']
    lf2.mu_itps = dict(er1=itp, er2=itp)

    ll2 = lf2()

    np.testing.assert_almost_equal(2 * ll1, ll2)


def test_set_data(xes: fd.ERSource):
    data1 = xes.data
    data2 = pd.concat([data1.copy(), data1.iloc[:1].copy()])
    data2['s1'] *= 1.3

    data3 = pd.concat([data2, data2.iloc[:1]])

    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)
    data3.reset_index(drop=True, inplace=True)

    lf = fd.LogLikelihood(
        sources=dict(data1=dict(er1=fd.ERSource),
                     data2=dict(er2=fd.ERSource)),
        data=dict(data1=data1,
                  data2=data2))

    def internal_data(sname, col):
        series = lf.sources[sname].data[col]
        n_padding = lf.sources[sname].n_padding
        return series.iloc[:len(series)-n_padding]

    # Test S1 columns are the same (DFs are annotated)
    # Here we don't have any padding since batch_size is n_events
    pd.testing.assert_series_equal(internal_data('er1', 's1'), data1['s1'])
    pd.testing.assert_series_equal(internal_data('er2', 's1'), data2['s1'])

    # Set new data for only one dataset
    lf.set_data(dict(data1=data2))

    # Test S1 columns are the same (DFs are annotated)
    # Here we might have padding
    pd.testing.assert_series_equal(internal_data('er1', 's1'), data2['s1'])
    pd.testing.assert_series_equal(internal_data('er2', 's1'), data2['s1'])

    # Set new data for both datasets
    lf.set_data(dict(data1=data1,
                     data2=data3))

    # Test S1 columns are the same (DFs are annotated)
    pd.testing.assert_series_equal(internal_data('er1', 's1'), data1['s1'])
    pd.testing.assert_series_equal(internal_data('er2', 's1'), data3['s1'])

    # Test padding for smaller dsets
    lf.set_data(dict(data2=data1))

    pd.testing.assert_series_equal(internal_data('er2', 's1'), data1['s1'])

def test_constraint(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        data=xes.data.copy())
    ll1 = lf()

    lf2 = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        log_constraint=lambda **kwargs: 100.,
        data=xes.data.copy())

    # Fix interpolator nondeterminism
    itp = lf.mu_itps['er']
    lf2.mu_itps = dict(er=itp)

    ll2 = lf2()

    np.testing.assert_almost_equal(ll1 + 100., ll2)


def test_hessian(xes: fd.ERSource):
    # Test the hessian at the guess position
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)

    guess = lf.guess()
    assert len(guess) == 2

    inv_hess = lf.inverse_hessian(guess)
    inv_hess_np = inv_hess.numpy()
    assert inv_hess_np.shape == (2, 2)
    assert inv_hess.dtype == fd.float_type()
    # Check symmetry of hessian
    # The hessian is explicitly symmetrized before being passed to
    # the optimizer in bestfit
    a = inv_hess_np[0, 1]
    b = inv_hess_np[1, 0]
    assert abs(a - b)/(a+b) < 1e-3


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

    bestfit = lf.bestfit(guess, optimizer=tfp.optimizer.bfgs_minimize, use_hessian=True)
    assert isinstance(bestfit[0], dict)
    assert len(bestfit[0]) == 2
    assert bestfit[0]['er_rate_multiplier'].dtype == np.float32


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

    bestfit = lf.bestfit(guess, optimizer=Minuit.from_array_func, error = (0.0001,1000))
    assert isinstance(bestfit[0], dict)
    assert len(bestfit[0]) == 2
    
