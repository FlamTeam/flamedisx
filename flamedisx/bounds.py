import numpy as np
from scipy import stats

import flamedisx as fd
export, __all__ = fd.exporter()


def bayes_bounds(df, in_dim, bounds_prob, bound, bound_type, supports, **kwargs):
    """Calculate bounds on a block using an inversion of Bayes theorem, with a flat prior.

    :param df: dataframe that the bounds will be added to
    :param in_dim: hidden variable dimension we are calculating the bounds for
    :param bounds_prob: set the value of the CDF of the posterior for the 'in' hidden
    variable that we will use to define the bounds
    :param bound: upper bound, lower bound or mle
    :param bound_type: distribution used in going from the block's 'in' hidden variable
    to 'out' hidden variable
    :param supports: sensible support for the 'in' hidden variable that we want bounds
    for, that the CDF of the posterior will be evaluated along
    """
    assert (bound in ('upper', 'lower', 'mle')), "bound argumment must be upper, lower or mle"
    assert (bound_type in ('binomial', 'normal')), "bound_type must be binomial or normal"

    if bound_type == 'binomial':
        cdfs = bayes_bounds_binomial(supports, **kwargs)

    elif bound_type == 'normal':
        cdfs = bayes_bounds_normal(supports, **kwargs)

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


def bayes_bounds_priors(source, batch, df, in_dim, bounds_prob, bound, bound_type, supports, **kwargs):
    """Calculate bounds on a block using an inversion of Bayes theorem, with a calculated prior.

    :param source: the source calling the function, for access to the priors
    :param batch: the batch of events that bounds are being computed for, to access the correct priors
    :param df: dataframe that the bounds will be added to
    :param in_dim: hidden variable dimension we are calculating the bounds for
    :param bounds_prob: set the value of the CDF of the posterior for the 'in' hidden
    variable that we will use to define the bounds
    :param bound: upper bound, lower bound or mle
    :param bound_type: distribution used in going from the block's 'in' hidden variable
    to 'out' hidden variable
    :param supports: sensible support for the 'in' hidden variable that we want bounds
    for, that the CDF of the posterior will be evaluated along
    """
    assert (bound in ('upper', 'lower',  'mle')), "bound argumment must be upper or lower"
    assert (bound_type in ('binomial',)), "bound_type must be binomial"

    if bound == 'upper':
        prior_pdfs = source.prior_PDFs_UB[batch]
    elif bound == 'lower':
        prior_pdfs = source.prior_PDFs_LB[batch]

    # We will calculate bounds with the prior and also with a flat prior. Take
    # the tightest set of bounds at the end
    cdfs_prior = bayes_bounds_binomial(supports, prior_pdf=prior_pdfs[in_dim], **kwargs)
    cdfs_no_prior = bayes_bounds_binomial(supports, **kwargs)

    if bound == 'lower':
        lower_lims_prior = [support[np.where(cdf < bounds_prob)[0][-1]]
                            if len(np.where(cdf < bounds_prob)[0]) > 0
                            else support[0]
                            for support, cdf in zip(supports, cdfs_prior)]
        lower_lims_no_prior = [support[np.where(cdf < bounds_prob)[0][-1]]
                               if len(np.where(cdf < bounds_prob)[0]) > 0
                               else support[0]
                               for support, cdf in zip(supports, cdfs_no_prior)]
        df.loc[batch * source.batch_size:(batch + 1) * source.batch_size - 1, in_dim + '_min'] = \
            max(lower_lims_prior, lower_lims_no_prior)

    elif bound == 'upper':
        upper_lims_prior = [support[np.where(cdf > 1. - bounds_prob)[0][0]]
                            if len(np.where(cdf > 1. - bounds_prob)[0]) > 0
                            else support[-1]
                            for support, cdf in zip(supports, cdfs_prior)]
        upper_lims_no_prior = [support[np.where(cdf > 1. - bounds_prob)[0][0]]
                               if len(np.where(cdf > 1. - bounds_prob)[0]) > 0
                               else support[-1]
                               for support, cdf in zip(supports, cdfs_no_prior)]
        df.loc[batch * source.batch_size:(batch + 1) * source.batch_size - 1, in_dim + '_max'] = \
            min(upper_lims_prior, upper_lims_no_prior)


def get_priors(source, reservoir, prior_dims,
               prior_data_cols, filter_data_cols,
               filter_dims_min, filter_dims_max):
    """Obtain priors on certain hidden variable dimensions, to obtain more
    accurate Bayes bounds. Separate priors calculated for estimating upper and
    lower bounds.

    :param reservoir: MC reservoir filtered for prior estimation
    :param prior_dims: tuple of dimensions we are obtaining priors for
    :param prior_data_cols: column numbers in reservoir corresponding to prior_dims
    :param filter_data_cols: column numbers in reservoir corresponding to dimensions
    we are filtering reservoir by to obtain the priors
    :param filter_dims_min: lower bounds of the dimensions we are filtering by, for
    obtaining lower bound priors
    :param filter_dims_max: upper bounds of the dimensions we are filtering by, for
    obtaining upper bound priors
    """
    prior_dict = {}

    for prior_dim, prior_data_col in zip(prior_dims, prior_data_cols):
        prior_data_filter = [True] * len(reservoir[:, 0])

        for filter_data_col, filter_dim_min in zip(filter_data_cols, filter_dims_min):
            prior_data_filter = prior_data_filter * (reservoir[:, filter_data_col] >= filter_dim_min)

        prior_data = reservoir[:, prior_data_col][prior_data_filter]
        prior_hist = np.histogram(prior_data)
        prior_pdf = stats.rv_histogram(prior_hist)
        prior_dict[prior_dim] = prior_pdf

    source.prior_PDFs_LB += (prior_dict,)

    prior_dict = {}

    for prior_dim, prior_data_col in zip(prior_dims, prior_data_cols):
        prior_data_filter = [True] * len(reservoir[:, 0])

        for filter_data_col, filter_dim_max in zip(filter_data_cols, filter_dims_max):
            prior_data_filter = prior_data_filter * (reservoir[:, filter_data_col] <= filter_dim_max)

        prior_data = reservoir[:, prior_data_col][prior_data_filter]
        prior_hist = np.histogram(prior_data)
        prior_pdf = stats.rv_histogram(prior_hist)
        prior_dict[prior_dim] = prior_pdf

    source.prior_PDFs_UB += (prior_dict,)


def bayes_bounds_binomial(supports, rvs_binom, ns_binom, ps_binom, prior_pdf=None):
    """Calculate bounds on a block using a binomial distribution.

    :param supports: Values of block 'in' dimension over which the PMF/CMF used to find the bounds
    will be calculated, for each event in the dataframe
    :param rvs_binom: Variable the block uses as the 'object' of the binomial calculation;
    must be the same shape as supports
    :param ns_binom: Variable the block uses as the number of trials of the binomial calculation;
    must be the same shape as supports
    :param ps_binom: Variable the block uses as the success probability of the binomial calculation;
    must be the same shape as supports
    :param prior_pdf: if we are using a non-flat prior, pass in the PDF to be used
    """
    assert (np.shape(rvs_binom) == np.shape(ns_binom) == np.shape(ps_binom) == np.shape(supports)), \
        "Shapes of suports, rvs_binom, ns_binom and ps_binom must be equal"

    def prior(x):
        if prior_pdf is None:
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

    :param supports: Values of block 'in' dimension over which the PMF/CMF used to find the bounds
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
