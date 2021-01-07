import typing as ty

import numpy as np
import tensorflow as tf
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()

o = tf.newaxis


@export
class Block:
    """One part of a BlockSource model.

    For example, P(electrons_detected | electrons_produced).
    """
    dimensions: ty.Tuple[str]

    depends_on: ty.Tuple[str] = tuple()

    model_functions: ty.Tuple[str] = tuple()
    special_model_functions: ty.Tuple[str] = tuple()
    array_columns: ty.Tuple[str] = tuple()
    frozen_model_functions: ty.Tuple[str] = tuple()
    model_attributes: ty.Tuple[str] = tuple()

    def __init__(self, source):
        self.source = source
        assert len(self.dimensions) in (1, 2), \
            "Blocks must output 1 or 2 dimensions"

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
        kwargs.update(self.source._domain_dict(self.dimensions, data_tensor))
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

    model_blocks: tuple
    final_dimensions: tuple
    initial_dimensions: tuple

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

            attributes = set(b.model_functions
                             + b.special_model_functions
                             + b.model_attributes)

            for x in attributes:
                # If a source attribute was specified,
                # override the block's attribute.
                if hasattr(self, x):
                    setattr(b, x, getattr(self, x))

            # Now that the block is properly furnished with all attributes,
            # call the block setup.
            b.setup()

            # Set the source attributes from the block's attributes.
            # If we did this before b.setup(), the source and block attributes
            # could diverge, causing potential confusion.
            for x in attributes:
                setattr(self, x, getattr(b, x))

            # Collect all information from the block.
            # We also do this after the setup; the array columns field in
            # particular is nice to change in the block setup
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
            setattr(self, k, tuple(set(v)))

        self.inner_dimensions = tuple([
            d for d in collected['dimensions']
            if ((d not in self.final_dimensions)
                and (d not in self.model_blocks[0].dimensions))])
        self.initial_dimensions = self.model_blocks[0].dimensions

        super().__init__(*args, **kwargs)

    @staticmethod
    def _find_block(blocks,
                    has_dim: ty.Union[list, tuple, set],
                    exclude: Block = None):
        """Find a block with a dimension in allowed
        Return (dimensions, b), or
         raises BlockNotFoundError if no such block found.
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

        for b in self.model_blocks:
            b_dims = b.dimensions

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
            results[b_dims] = r = b.compute(data_tensor, ptensor, **kwargs)

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

    def add_derived_observables(self, d):
        pass


class BlockNotFoundError(Exception):
    pass
