import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import flamedisx as fd
from flamedisx.likelihood import DEFAULT_DSETNAME


n_events = 2


@pytest.fixture(params=["ER", "NR"])
def xes(request):
    # warnings.filterwarnings("error")
    data = pd.DataFrame([dict(s1=56., s2=2905., drift_time=143465.,
                              x=2., y=0.4, z=-20, r=2.1, theta=0.1,
                              event_time=15e17),
                         dict(s1=23, s2=1080., drift_time=445622.,
                              x=1.12, y=0.35, z=-59., r=1., theta=0.3,
                              event_time=15e17)])
    if request.param == 'ER':
        x = fd.ERSource(data.copy(), batch_size=2, max_sigma=8)
    else:
        x = fd.NRSource(data.copy(), batch_size=2, max_sigma=8)
    return x


def test_wimp_source(xes):
    # test KeyError 't' issue, because of add_extra_columns bug
    lf = fd.LogLikelihood(sources=dict(er=fd.ERSource,
                                       wimp=fd.WIMPSource),
                          free_rates=('er', 'wimp'))

    d = lf.simulate(er_rate_multiplier=1.0,
                    wimp_rate_multiplier=0.)
    lf.set_data(d)


def test_inference(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        data=xes.data)

    # Test single-batch likelihood
    x, x_grad, _ = lf._log_likelihood(i_batch=tf.constant(0),
                                      dsetname=DEFAULT_DSETNAME,
                                      data_tensor=lf.data_tensors[DEFAULT_DSETNAME][0],
                                      batch_info=lf.batch_info,
                                      elife=tf.constant(200e3))
    assert isinstance(x, tf.Tensor)
    assert x.dtype == fd.float_type()
    assert x.numpy() < 0

    assert isinstance(x_grad, tf.Tensor)
    assert x_grad.dtype == fd.float_type()
    assert x_grad.numpy().shape == (1,)

    # Test a different parameter gives a different likelihood
    x2, x2_grad, _ = lf._log_likelihood(i_batch=tf.constant(0),
                                        dsetname=DEFAULT_DSETNAME,
                                        data_tensor=lf.data_tensors[DEFAULT_DSETNAME][0],
                                        batch_info=lf.batch_info,
                                        elife=tf.constant(300e3))
    assert (x - x2).numpy() != 0
    assert (x_grad - x2_grad).numpy().sum() != 0

    # Test batching
    l1 = lf.log_likelihood()
    l2 = lf()
    lf.log_likelihood(elife=tf.constant(200e3))


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

    np.testing.assert_allclose(lf2.log_likelihood()[0],
                               l1[0],
                               rtol=1e-6)


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


def test_no_dset(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        data=None)

    lf2 = fd.LogLikelihood(
        sources=dict(data1=dict(er1=fd.ERSource),
                     data2=dict(er2=fd.ERSource)),
        data=dict(data1=None,
                  data2=None))


def test_set_data_on_no_dset(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        data=None,
        batch_size=4)
    # The batch_size can be at most 2 * len(data) or padding wont work
    # which is why it is set explicitly in this test with only 2 events
    # Usually when constructing the likelihood with a very small dataset
    # the batch_size is set accordingly, but in this test with data=None
    # that is not possible (an assert has been put in Source._init_padding)

    lf.set_data(xes.data.copy())

    assert lf.sources['er'].batch_size == 4
    assert lf.sources['er'].n_batches == 1
    assert lf.sources['er'].n_padding == 2
    ll1 = lf()

    lf2 = fd.LogLikelihood(
        sources=dict(data1=dict(er1=fd.ERSource),
                     data2=dict(er2=fd.ERSource)),
        data=dict(data1=None,
                  data2=None),
        batch_size=4)

    lf2.set_data(dict(data1=xes.data.copy(),
                      data2=xes.data.copy()))
    ll2 = lf2()


def test_retrace_set_data(xes: fd.ERSource):
    # Test issue #53
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        data=xes.data.copy())
    ll1 = lf()

    new_data = xes.data.copy()
    new_data['s2'] *= 2
    lf.set_data(new_data)

    ll2 = lf()

    # issue 53 would not have retraced ll so lf() would be unchanged
    assert not ll1 == ll2


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

    np.testing.assert_almost_equal(2 * ll1, ll2, decimal=2)


def test_simulate(xes):
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource),
        data=None)

    events = lf.simulate()
    events = lf.simulate(er_rate_multiplier=2.)
    events = lf.simulate(fix_truth=dict(x=0., y=0., z=-50.))


def test_simulate_column(xes):
    # Test for issue #47, check if not crashing since ColumnSource has no
    # simulator
    lf = fd.LogLikelihood(
        sources=dict(er=fd.ERSource,
                     muur=fd.ColumnSource),
        data=None)

    events = lf.simulate()
    events = lf.simulate(er_rate_multiplier=2.)
    events = lf.simulate(fix_truth=dict(x=0., y=0., z=-50.))


def test_set_data(xes: fd.ERSource):
    data1 = xes.data
    data2 = pd.concat([data1.copy(), data1.iloc[:1].copy()])
    data2['s1'] *= 0.7

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


def test_hessian_rateonly(xes: fd.ERSource):

    class Bla(xes.__class__):
        """ER source with slightly different elife
        to prevent a singular matrix
        """
        @staticmethod
        def electron_detection_eff(drift_time, *,
                                   different_elife=333e3,
                                   extraction_eff=0.96):
            return extraction_eff * tf.exp(-drift_time / different_elife)

    # Test the hessian at the guess position
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__,  er2=Bla),
        free_rates=['er', 'er2'],
        data=xes.data)

    guess = lf.guess()
    assert len(guess) == 2

    print(guess)
    print(lf.log_likelihood(second_order=True, **guess))

    inv_hess = lf.inverse_hessian(guess)
    assert inv_hess.shape == (2, 2)
    assert inv_hess.dtype == np.float64
    # Check symmetry of hessian
    # The hessian is explicitly symmetrized before being passed to
    # the optimizer in bestfit
    a = inv_hess[0, 1]
    b = inv_hess[1, 0]
    assert abs(a - b)/(a+b) < 1e-3


def test_hessian_rate_and_shape(xes: fd.ERSource):
    lf = fd.LogLikelihood(
        sources=dict(er=xes.__class__),
        elife=(100e3, 500e3, 5),
        free_rates='er',
        data=xes.data)

    guess = lf.guess()
    assert len(guess) == 2

    print(guess)
    print(lf.log_likelihood(second_order=True, **guess))

    inv_hess = lf.inverse_hessian(guess)
    assert inv_hess.shape == (2, 2)
    assert inv_hess.dtype == np.float64
    a = inv_hess[0, 1]
    b = inv_hess[1, 0]
    assert abs(a - b)/(a+b) < 1e-3
