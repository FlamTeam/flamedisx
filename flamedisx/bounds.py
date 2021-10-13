import typing as ty

import numpy as np
import pandas as pd
from scipy import spatial
from scipy import stats

import flamedisx as fd
export, __all__ = fd.exporter()


def bayes_bounds(df, in_dim, bounds_prob, bound, bound_type, supports, **kwargs):
    assert (bound == 'upper' or 'lower' or 'mle'), "bound argumment must be upper, lower or mle"
    assert (bound_type == 'binomial' or 'normal'), "bound_type must be binomial or normal"

    if bound_type == 'binomial':
        cdfs =  bayes_bounds_binomial(supports, **kwargs)

    elif bound_type == 'normal':
        cdfs =  bayes_bounds_normal(supports, **kwargs)

    if bound == 'lower':
        lower_lims = [support[np.where(cdf < bounds_prob)[0][-1]]
                      if len(np.where(cdf < bounds_prob)[0]) > 0
                      else support[0]
                      for support, cdf in zip(supports, cdfs)]
        df[in_dim + '_min'] = lower_lims

    elif bound == 'upper':
        upper_lims = [support[np.where(cdf > 1. - bounds_prob)[0][0]]
                      if len(np.where(cdf > 1. - bounds_prob)[0]) > 0
                      else support[-1]
                      for support, cdf in zip(supports, cdfs)]
        df[in_dim + '_max'] = upper_lims

    elif bound == 'mle':
        mles = [support[np.argmin(np.abs(cdf - 0.5))] for support, cdf in zip(supports, cdfs)]
        df[in_dim + '_mle'] = mles


def bayes_bounds_binomial(supports, rvs_binom, ns_binom, ps_binom, prior_data=None):
    """Calculate bounds on a block using a binomial distribution.

    :param supports: Values of block 'input' dimension over which the PMF/CMF used to find the bounds
    will be calculated, for each event in the dataframe
    :param rvs_binom: Variable the block uses as the 'object' of the binomial calculation;
    must be the same shape as supports
    :param ns_binom: Variable the block uses as the number of trials of the binomial calculation;
    must be the same shape as supports
    :param ps_binom: Variable the block uses as the success probability of the binomial calculation;
    must be the same shape as supports
    :param prior_data: FILL THIS IN
    """
    assert (np.shape(rvs_binom) == np.shape(ns_binom) == np.shape(ps_binom) == np.shape(supports)), \
        "Shapes of suports, rvs_binom, ns_binom and ps_binom must be equal"

    if prior_data is not None:
        prior_hist = np.histogram(prior_data)
        prior_pdf = stats.rv_histogram(prior_hist)
    def prior(x):
        if prior_data is None:
            return 1
        elif np.sum(prior_pdf.pdf(x)) == 0:
            return 1
        else:
            return prior_pdf.pdf(x)

    pdfs = [stats.binom.pmf(rv_binom, n_binom, p_binom) * prior(support)
            for rv_binom, n_binom, p_binom, support in zip(rvs_binom, ns_binom, ps_binom, supports)]
    pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
    cdfs = [np.cumsum(pdf) for pdf in pdfs]

    return cdfs


def bayes_bounds_normal(supports, rvs_normal, mus_normal, sigmas_normal):
    """Calculate bounds on a block using a normal distribution.
    Note that we do not account for continuity corrections here.

    :param supports: Values of block 'input' dimension over which the PMF/CMF used to find the bounds
    will be calculated, for each event in the dataframe
    :param rvs_normal: Variable the block uses as the 'object' of the normal calculation;
    must be the same shape as supports
    :param mus_normal: Variable the block uses as the mean of the normal calculation;
    must be the same shape as supports
    :param sigmas_normal: Variable the block uses as the standard deviation of the normal calculation;
    must be the same shape as supports
    """
    assert (np.shape(rvs_normal) == np.shape(mus_normal) == np.shape(sigmas_normal) == np.shape(supports)), \
        "Shapes of supports, rvs_normal, mus_normal and sigmas_normal must be equal"

    pdfs = [stats.norm.pdf(rv_normal, mu_normal, sigma_normal)
            for rv_normal, mu_normal, sigma_normal in zip(rvs_normal, mus_normal, sigmas_normal)]
    pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
    cdfs = [np.cumsum(pdf) for pdf in pdfs]

    return cdfs


def scale_observables(df, observables):
    """"""
    observables_scaled = []
    for x in observables:
        if np.std(df[x]) < 1e-10:
            observables_scaled.append(df[x] - df[x].mean())
        else:
            observables_scaled.append((df[x] - df[x].mean()) / df[x].std())

    return observables_scaled


def kd_bounds(source, df, kd_tree_observables: ty.Tuple[str], initial_dimension: str):
    """"""
    source.MC_reservoir = MC_data = source.simulate(int(1e6), keep_padding=True)

    df_full = pd.concat([MC_data, df])
    data_full = np.array(list(zip(*scale_observables(df_full, kd_tree_observables))))

    data = data_full[len(MC_data)::]
    data_MC = data_full[0:len(MC_data)]

    tree = spatial.KDTree(data_MC)

    dist, ind = tree.query(data[::], k=100)
    for i in range(len(source.data)):
        df.at[i, initial_dimension + '_min'] = min(MC_data[initial_dimension].iloc[ind[i]])
        df.at[i, initial_dimension + '_max'] = max(MC_data[initial_dimension].iloc[ind[i]])

    dist, ind = tree.query(data[::], k=1)
    for i in range(len(source.data)):
        df.at[i, initial_dimension + '_mle'] = MC_data[initial_dimension].iloc[ind[i]]
