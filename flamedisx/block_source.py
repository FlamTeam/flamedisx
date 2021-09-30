import typing as ty

import numpy as np
import tensorflow as tf
import pandas as pd
from scipy import stats
import scipy.special as sp

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis


@export
class Block:
    """One part of a BlockSource model.

    For example, P(electrons_detected | electrons_produced).
    """
    #: Source the block belongs to
    source: fd.Source = None

    #: Names of dimensions of the block's compute result
    dimensions: ty.Tuple[str]

    #: Additional dimensions used in the block computation.
    #: Label True if they represent an internally contracted hidden variable;
    #: these will be added to inner_dimensions so domain tensors are calculated
    #: automatically.
    #: Label False otherwise; these will be added to bonus_dimensions. Thus,
    #: any additional domain tensors utilising them will need calculating via
    #: the block overriding _domain_dict_bonus())
    extra_dimensions: ty.Tuple[ty.Tuple[str, bool]] = tuple()

    #: Blocks whose result this block expects as an extra keyword
    #: argument to compute. Specify as ((block_dims, argument_name), ...),
    #: where block_dims is the dimensions-tuple of the block, and argument_name
    #: the expected name of the compute keyword argument.
    depends_on: ty.Tuple[ty.Tuple[ty.Tuple[str], str]] = tuple()

    #: Names of model functions defined in this block
    model_functions: ty.Tuple[str] = tuple()

    #: Names of model functions that take an additional first argument
    #: ('bonus arg') defined in this block; must be a subset of model_functions
    special_model_functions: ty.Tuple[str] = tuple()

    #: Names of columns this block expects to be array-valued
    array_columns: ty.Tuple[str] = tuple()

    #: Frozen model functions defined in this block
    frozen_model_functions: ty.Tuple[str] = tuple()

    #: Additional attributes this Block will furnish the source with.
    #: These can be overriden by Source attributes, just like model functions.
    model_attributes: ty.Tuple[str] = tuple()

    def __init__(self, source):
        self.source = source
        assert len(self.dimensions) in (1, 2), \
            "Blocks must output 1 or 2 dimensions"
        # Currently only support 1 extra_dimension per block
        assert len(self.extra_dimensions) <= 1, \
            f"{self} has >1 extra dimension!"

    # Redirect all model attribute queries to the linked source,
    # as soon as one exists, and has the relevant attribute.
    # We need __getattribute__, not __getattr__, to catch access
    # to attributes that exist.
    def __getattribute__(self, name):
        # Can't use self.something here without recursion...
        linked_attributes = (
            super().__getattribute__('model_functions')
            + super().__getattribute__('model_attributes'))
        source = super().__getattribute__('source')

        if (name in linked_attributes and hasattr(source, name)):
            return getattr(source, name)
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if (name in (self.model_functions + self.model_attributes)
                and self.source is not None):
            setattr(self.source, name, value)
        else:
            super().__setattr__(name, value)

    def setup(self):
        """Do any necessary initialization.

        Called after the block's attributes have been properly overriden
        by source attributes, if specified.
        """

    def gimme(self, *args, **kwargs):
        """Shorthand for self.source.gimme, see docs there"""
        return self.source.gimme(*args, **kwargs)

    def gimme_numpy(self, *args, **kwargs):
        """Shorthand for self.source.gimme_numpy"""
        return self.source.gimme_numpy(*args, **kwargs)

    def compute(self, data_tensor, ptensor, **kwargs):
        if len(self.extra_dimensions) == 0:
            kwargs.update(self.source._domain_dict(
                self.dimensions, data_tensor))
        else:
            if self.extra_dimensions[0][1] is True:
                kwargs.update(self.source._domain_dict_extra(
                    self.dimensions, self.extra_dimensions[0][0], data_tensor))
            else:
                kwargs.update(self.source._domain_dict(
                    self.dimensions, data_tensor))
                kwargs.update(self._domain_dict_bonus(data_tensor))
        result = self._compute(data_tensor, ptensor, **kwargs)
        assert result.dtype == fd.float_type(), \
            f"{self}._compute returned tensor of wrong dtype!"
        assert len(result.shape) == len(self.dimensions) + 1, \
            f"{self}._compute returned tensor of wrong rank!"
        return result

    def simulate(self, d: pd.DataFrame):
        return_value = self._simulate(d)
        assert return_value is None, f"_simulate of {self} should return None"
        # Check necessary columns were actually added
        for dim in self.dimensions:
            assert dim in d.columns, f"_simulate of {self} must set {dim}"
            assert np.all(np.isfinite(d[dim].values)),\
                f"_simulate of {self} returned non-finite values of {dim}"

    def annotate(self, d: pd.DataFrame):
        """Add _min and _max for each dimension to d in-place"""
        return_value = self._annotate(d)
        assert return_value is None, f"_annotate of {self} should return None"

        for dim in self.dimensions:
            if dim in self.source.final_dimensions \
                    or dim in self.source.initial_dimensions:
                continue
            for bound in ('min', 'max'):
                colname = f'{dim}_{bound}'
                assert colname in d.columns, \
                    f" must set {colname}"
            assert np.all(d[f'{dim}_min'].values
                          <= d[f'{dim}_max'].values), \
                f"_annotate of {self} set misordered bounds"

    def bayes_bounds_skew_normal(self, df, in_dim, supports, rvs_skew_normal, mus_skew_normal,
                                 sigmas_skew_normal, alphas_skew_normal, bound, prior_data):
        """
        """
        assert (bound == 'upper' or 'lower' or 'mle'), "bound argumment must be upper, lower or mle"
        assert (np.shape(rvs_skew_normal) == np.shape(mus_skew_normal) \
            == np.shape(sigmas_skew_normal) == np.shape(supports)), \
            "Shapes of supports, rvs_skew_normal, mus_skew_normal and sigmas_skew_normal must be equal"

        def skew_normal(x, mu, sigma, alpha):
            with np.errstate(invalid='ignore', divide='ignore'):
                return (1 / sigma) * np.exp(-0.5 * (x - mu)**2 / sigma**2) \
                    * (1 + sp.erf(alpha * (x - mu) / (np.sqrt(2) * sigma)))

        prior_hist = np.histogram(prior_data)
        prior_pdf = stats.rv_histogram(prior_hist)
        def prior(x):
            if np.sum(prior_pdf.pdf(x)) == 0:
                return 1
            else:
                return prior_pdf.pdf(x)

        pdfs = [skew_normal(rv_skew_normal, mu_skew_normal, sigma_skew_normal, alpha_skew_normal) \
                * prior(support)
                for rv_skew_normal, mu_skew_normal, sigma_skew_normal, alpha_skew_normal, support
                in zip(rvs_skew_normal, mus_skew_normal, sigmas_skew_normal, alphas_skew_normal, supports)]
        pdfs = [pdf / np.sum(pdf) for pdf in pdfs]
        cdfs = [np.cumsum(pdf) for pdf in pdfs]

        if bound == 'lower':
            lower_lims = [support[np.where(cdf < self.source.bounds_prob)[0][-1]]
                          if len(np.where(cdf < self.source.bounds_prob)[0]) > 0
                          else support[0]
                          for support, cdf in zip(supports, cdfs)]
            df[in_dim + '_min'] = lower_lims

        elif bound == 'upper':
            upper_lims = [support[np.where(cdf > 1. - self.source.bounds_prob)[0][0]]
                          if len(np.where(cdf > 1. - self.source.bounds_prob)[0]) > 0
                          else support[-1]
                          for support, cdf in zip(supports, cdfs)]
            df[in_dim + '_max'] = upper_lims

        elif bound == 'mle':
            mles = [support[np.argmin(np.abs(cdf - 0.5))] for support, cdf in zip(supports, cdfs)]
            df[in_dim + '_mle'] = mles

    def check_data(self):
        pass

    def _compute(self, data_tensor, ptensor, **kwargs):
        """Return (n_batch_events, ...dimensions...) tensor"""
        raise NotImplementedError

    def _simulate(self, d):
        """Simulate extra columns in place.

        Use the p_accepted column to modify acceptances; do not remove
        events here.
        """
        raise NotImplementedError

    def _annotate(self, d):
        """Add _min and _max for each dimension to d in-place"""
        raise NotImplementedError

    def _domain_dict_bonus(self, d):
        """Calculate any additional intenal tensors arising from the use of
        bonus_dimensions in a block. Override within block"""
        raise NotImplementedError

    def _calculate_dimsizes_special(self):
        """Re-calculate dimension size and steps differently for any
        dimensions; will need to override _calculate_dimsizes_special()
        within a block"""
        pass


@export
class FirstBlock(Block):
    """The first Block of a source. This is usually an energy spectrum"""

    def _simulate(self, d):
        raise RuntimeError("FirstBlock's shouldn't simulate")

    def _annotate(self, d):
        # First block can omit annotate
        pass

    def domain(self, data_tensor):
        """Return dictionary mapping dimension -> domain"""
        raise NotImplementedError

    def random_truth(self, n_events, fix_truth=None, **params):
        raise NotImplementedError

    def mu_before_efficiencies(self, **params):
        raise NotImplementedError

    def validate_fix_truth(self, d):
        raise NotImplementedError


@export
class BlockModelSource(fd.Source):
    """Source whose model is split over different Blocks
    """

    #: Blocks the source is built from.
    #: simulate will be called from first to last, annotate from last to first.
    model_blocks: tuple

    #: Dimensions provided by the first block
    initial_dimensions: tuple

    #: Additional dimensions used in the block computation;
    #: see Block.extra_dimensions for info
    extra_dimensions = ()

    def __init__(self, *args, **kwargs):
        if isinstance(self.model_blocks[0], FirstBlock):
            # Blocks have already been instantiated
            return
        if not issubclass(self.model_blocks[0], FirstBlock):
            raise RuntimeError("The first block must inherit from FirstBlock")
        for b in self.model_blocks[1:]:
            if issubclass(b, FirstBlock):
                raise RuntimeError("Only the first block can be a FirstBlock")

        # Collect attributes from the different blocks in this dictionary:
        collected = {k: [] for k in (
            'dimensions',
            'extra_dimensions',
            'model_functions',
            'special_model_functions',
            'model_attributes',
            'frozen_model_functions',
            'array_columns')}

        # Instantiate the blocks.
        self.model_blocks = tuple([b(self) for b in self.model_blocks])

        for b in self.model_blocks:
            # Maybe someome forgot a comma in a tuple specification
            for k in collected:
                if not isinstance(getattr(b, k), tuple):
                    raise ValueError(
                        f"{k} in {b} should be a tuple, not a {type(k)}")

            # Call the setup method. This method is not really needed anymore;
            # blocks can simply override __init__ for setup, as long as they
            # call super().__init__ *first* (else self.source would not be set)
            b.setup()

            # Set the source attributes from the block's attributes.
            # From here on, block attributes become invisible and irrelevant,
            # since get/setattr on the blocks will redirect to the source.
            for x in set(b.model_functions + b.model_attributes):
                setattr(self, x, getattr(b, x))

            # Collect all information from the block.
            # We do this after the setup; the array columns field in
            # particular is nice to change in the block setup code
            for k in collected:
                collected[k] += getattr(b, k)

            # Someone might try to modify a source attribute after
            # instantiation, then be surprised when nothing happens
            # (because the same-named block attribute wasn't changed).
            # Sorry. I can't use properties to prevent writing, since
            # those are class-bound.

        # The source may declare additional frozen data methods
        collected['frozen_model_functions'] += self.frozen_model_functions

        # For array columns, the source may declare new columns
        # or override the length of the old columns
        for column, length in self.array_columns:
            for i, (_old_column, _) in enumerate(collected['array_columns']):
                if _old_column == column:
                    # Change length of existing column
                    collected['array_columns'][i] = (column, length)
                    break
            else:
                # New column
                collected['array_columns'] += [(column, length)]

        # Make the collected attributes available to the source
        for k, v in collected.items():
            if k == 'dimensions':
                # Dimensions is a special case, see below
                continue
            setattr(self, k, getattr(self, k) + tuple(set(v)))

        self.inner_dimensions = tuple(
            [d for d in collected['dimensions']
                if ((d not in self.final_dimensions)
                    and (d not in self.model_blocks[0].dimensions))]
            + [d[0] for d in collected['extra_dimensions']
                if d[1] is True])
        self.initial_dimensions = self.model_blocks[0].dimensions
        self.bonus_dimensions = tuple([
            d[0] for d in collected['extra_dimensions'] if d[1] is False])

        super().__init__(*args, **kwargs)

    @staticmethod
    def _find_block(blocks,
                    has_dim: ty.Union[list, tuple, set],
                    exclude: Block = None):
        """Find a block with a dimension in has_dim, other than the block in
        exclude. Return (dimensions, b), or raises BlockNotFoundError.
        """
        for dims, b in blocks.items():
            if b is exclude:
                continue
            for d in dims:
                if d in has_dim:
                    return dims, b
        raise BlockNotFoundError(f"No block with {has_dim} found!")

    def _differential_rate(self, data_tensor, ptensor):
        results = {}
        already_stepped = ()  # Avoid double-multiplying to account for stepping

        for b in self.model_blocks:
            b_dims = b.dimensions
            scaling_dims = b.dimensions + tuple([extra_dimension[0] for extra_dimension in b.extra_dimensions if extra_dimension[1] is True])

            # Gather extra compute arguments.
            kwargs = dict()
            for dependency_dims, dependency_name in b.depends_on:
                if dependency_dims not in results:
                    raise ValueError(
                        f"Block {b} depends on {dependency_dims}, but that has "
                        f"not yet been computed")
                kwargs[dependency_name] = results[dependency_dims]
                kwargs.update(self._domain_dict(dependency_dims, data_tensor))

            # Compute the block
            r = b.compute(data_tensor, ptensor, **kwargs)

            # Scale the block by stepped dimensions, if not already done in
            # another block
            for dim in scaling_dims:
                if (dim in self.inner_dimensions) and \
                        (dim not in self.no_step_dimensions) and \
                        (dim not in already_stepped):
                    steps = self._fetch(dim+'_steps', data_tensor=data_tensor)
                    step_mul = tf.repeat(steps[:, o], tf.shape(r)[1], axis=1)
                    step_mul = tf.repeat(step_mul[:, :, o],
                                         tf.shape(r)[2], axis=2)
                    r *= step_mul
                    already_stepped += (dim,)

            results[b_dims] = r

            # Try to matrix multiply with earlier blocks, until we cannot
            # do so anymore.
            try:
                while True:
                    b2_dims, r2 = self._find_block(
                        results, has_dim=b_dims, exclude=r)

                    new_dims, r = self.multiply_block_results(
                        b_dims, b2_dims, r, r2)
                    results[new_dims] = r
                    del results[b_dims]
                    del results[b2_dims]

            except BlockNotFoundError:
                continue

        # The result should have a tensor with only final dimensions
        result = None
        for dims, result in results.items():
            if all([d in self.final_dimensions for d in dims]):
                break
        if result is None:
            raise ValueError("Result was not computed!")
        return tf.reshape(tf.squeeze(result), (self.batch_size,))

    def multiply_block_results(self, b_dims, b2_dims, r, r2):
        """Return result of matrix-multiplying two block results
        :param b_dims: tuple, dimension specification of r
        :param b2_dims: tuple, dimension specification of r2
        :param r: tensor , first block result to be multiplier
        :param r2: tensor, second block result to be multiplied
        :return: (dimension specification, tensor) of results
        """
        shared_dim = set(b_dims).intersection(set(b2_dims))
        if len(shared_dim) != 1:
            raise ValueError(f"Expected one shared dimension, "
                             f"found {len(shared_dim)}!")
        shared_dim = list(shared_dim)[0]

        # Figure out dimensions of result
        new_dims = tuple([d for d in b_dims if d != shared_dim]
                         + [d for d in b2_dims if d != shared_dim])

        # TODO: can we generalize to rank > 2?
        assert len(b_dims) in [1, 2]
        assert len(b2_dims) in [1, 2]

        # Due to the batch dimension, tf.matmul requires that the
        # ranks of arguments to tf.matmul match exactly.
        dummy_axis = None
        if len(b_dims) == 1 and len(b2_dims) == 2:
            dummy_axis = 1
            r = r[:, o, :]
        elif len(b_dims) == 2 and len(b2_dims) == 1:
            dummy_axis = 2
            r2 = r2[:, :, o]
        elif len(b_dims) == 2 and len(b2_dims) == 2:
            # Ensure the shared dimension is in the right position
            # for matrix multiplication
            if b_dims.index(shared_dim) != 1:
                assert len(b_dims) == 2
                r = tf.transpose(r, [0, 2, 1])
            if b2_dims.index(shared_dim) != 0:
                assert len(b2_dims) == 2
                r2 = tf.transpose(r2, [0, 2, 1])
        else:
            raise ValueError(
                "Unsupported ranks {len(b_dims)}, {len(b2_dims)}")

        # Multiply the matching index to get a new block
        r = r @ r2
        if dummy_axis:
            r = tf.squeeze(r, dummy_axis)
        assert len(r.shape) == len(new_dims) + 1

        return (new_dims, r)

    def random_truth(self, n_events, fix_truth=None, **params):
        # First block provides the 'deep' truth (energies, positions, time)
        return self.model_blocks[0].random_truth(
            n_events, fix_truth=fix_truth, **params)

    def validate_fix_truth(self, fix_truth):
        return self.model_blocks[0].validate_fix_truth(fix_truth)

    def _check_data(self):
        super()._check_data()
        for b in self.model_blocks:
            b.check_data()

    def _simulate_response(self):
        # All blocks after the first help to simulate the response
        d = self.data
        d['p_accepted'] = 1.
        for b in self.model_blocks[1:]:
            b.simulate(d)
        return d.iloc[np.random.rand(len(d)) < d['p_accepted'].values].copy()

    def _annotate(self, _skip_bounds_computation=False):
        d = self.data
        fd.bounds.energy_bounds(source=self, df=d,
                                kd_tree_observables=('s1', 's2', 'r', 'z'), initial_dimension='energy')

        # By going in reverse order through the blocks, we can use the bounds
        # on hidden variables closer to the final signals (easy to compute)
        # for estimating the bounds on deeper hidden variables.
        for b in self.model_blocks[::-1]:
            b.annotate(d)

    def mu_before_efficiencies(self, **params):
        return self.model_blocks[0].mu_before_efficiencies(**params)

    def draw_positions(self, *args, **kwargs):
        # TODO: This is a kludge; allows one to reuse draw_positions
        # from the first block in a source-overriden random_truth function.
        return self.model_blocks[0].draw_positions(*args, **kwargs)

    def domain(self, x, data_tensor=None):
        if x in self.initial_dimensions:
            # Domain computation of the inner dimension is passed to the
            # first block
            return self.model_blocks[0].domain(data_tensor=data_tensor)[x]
        else:
            return super().domain(x, data_tensor=data_tensor)

    def _domain_dict(self, dimensions, data_tensor):
        if len(dimensions) == 1:
            return {dimensions[0]:
                    self.domain(dimensions[0], data_tensor)}
        assert len(dimensions) == 2
        return dict(zip(dimensions,
                        self.cross_domains(*dimensions,
                                           data_tensor=data_tensor)))

    def _domain_dict_extra(self, dimensions, internal_dimension, data_tensor):
        all_dimensions = list(dimensions) + list((internal_dimension,))
        assert len(all_dimensions) == 3
        return dict(zip(all_dimensions,
                        self.cross_domains_extra(*all_dimensions,
                                                 data_tensor=data_tensor)))

    def add_derived_observables(self, d):
        pass

    def calculate_dimsizes_special(self):
        """Custom calulcation of any dimension sizes and steps; override
        _calculate_dimsizes_special within block"""
        for b in self.model_blocks:
            b._calculate_dimsizes_special()


class BlockNotFoundError(Exception):
    pass
