import typing as ty

import numpy as np
import tensorflow as tf
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


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
    frozen_data_methods: ty.Tuple[str] = tuple()
    static_attributes: ty.Tuple[str] = tuple()

    def __init__(self, source):
        self.source = source
        assert len(self.dimensions) in (1, 2), \
            "Blocks must output 1 or 2 dimensions"

    def gimme(self, *args, **kwargs):
        """Shorthand for self.source.gimme, see docs there"""
        return self.source.gimme(*args, **kwargs)

    def gimme_numpy(self, *args, **kwargs):
        """Shorthand for self.source.gimme, see docs there"""
        return self.source.gimme_numpy(*args, **kwargs)

    def domain(self, data_tensor):
        """Return dictionary mapping dimension -> domain"""
        if len(self.dimensions) == 1:
            return {self.dimensions[0]:
                    self.source.domain(self.dimensions[0], data_tensor)}
        assert len(self.dimensions) == 2
        return dict(zip(self.dimensions,
                        self.source.cross_domains(*self.dimensions,
                                                  data_tensor=data_tensor)))

    def compute(self, data_tensor, ptensor, _pass_domain=True, **kwargs):
        if _pass_domain:
            kwargs.update(self.domain(data_tensor))
        return self._compute(data_tensor, ptensor, **kwargs)

    def simulate(self, d: pd.DataFrame):
        return_value = self._simulate(d)
        assert return_value is None, f"_simulate of {self} should return None"
        # Check necessary columns were actually added
        for dim in self.dimensions:
            assert dim in d.columns, f"_simulate of {self} must set {dim}"

    def annotate(self, d: pd.DataFrame, _do_checks=False):
        """Add _min and _max for each dimension to d in-place"""
        return_value = self._annotate(d)
        assert return_value is None, f"_annotate of {self} should return None"

        if not _do_checks:
            return
        for dim in self.dimensions:
            if dim in self.source.final_dimensions:
                continue
            for bound in ('min', 'max'):
                colname = f'{dim}_{bound}'
                assert colname in d.columns, \
                    f"_annotate of {self} must set {colname}"

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
        pass

    def _annotate(self, d):
        """Add _min and _max for each dimension to d in-place"""
        pass


@export
class BlockModelSource(fd.Source):
    """Source whose model is split over different Blocks
    """

    model_blocks: tuple
    final_dimensions: tuple

    def __init__(self, *args, **kwargs):
        # Collect attributes from the different blocks in this dictionary:
        collected = {k: [] for k in (
            'model_functions',
            'special_model_functions',
            'static_attributes',
            'frozen_data_methods',
            'array_columns')}

        # Instantiate the blocks
        self.model_blocks = tuple([b(self) for b in self.model_blocks])

        for b in self.model_blocks:
            _this_block = {}
            for k in collected:
                _this_block[k] = list(getattr(b, k))
                collected[k] += _this_block[k]

            for x in (_this_block['model_functions']
                      + _this_block['special_model_functions']
                      + _this_block['static_attributes']):
                # If a source attribute was specified,
                # override the block's attribute.
                if hasattr(self, x):
                    setattr(b, x, getattr(self, x))

                # Set the source attribute from the block's attribute
                setattr(self, x, getattr(b, x))

                # Someone might try to modify the source attribute after
                # instantiation, then be really surprised when nothing happens.
                # Sorry. I can't use properties to prevent writing, since
                # those are class-bound.

        # TODO: Ugly since source's conventions / naming is a bit divergent
        self.data_methods = tuple(collected['model_functions']
                                  + collected['special_model_functions'])
        self.special_data_methods = tuple(collected['special_model_functions'])
        self.frozen_data_methods = tuple(collected['frozen_data_methods'])
        self.frozen_data_methods = tuple(collected['frozen_data_methods'])
        self.array_columns = tuple(collected['array_columns'])
        super().__init__(*args, **kwargs)

    @staticmethod
    def _find_block(blocks,
                    has_dim: ty.Union[list, tuple, set],
                    exclude: Block = None):
        """Find a block with a dimension in allowed.
        Return ((dimensions, b), dimension found), or raises ValueError
        """
        for dims, b in blocks.items():
            if b is exclude:
                continue
            for d in dims:
                if d is has_dim:
                    return (dims, b), d
        raise ValueError(f"No block with {has_dim} found!")

    def _differential_rate(self, data_tensor, ptensor):
        # Calculate blocks that have no dependencies
        results = {b.dimensions(): b.compute(
                        data_tensor, ptensor,
                        _pass_domain=(b != self.model_blocks[0]))
                   for b in self.model_blocks}
        waiting_blocks = {b.dimensions(): b
                          for b in self.model_blocks
                          if b.dimensions() not in results}

        while True:
            # Can we compute a block waiting on dependencies?
            available = set(results.keys())
            for b_dims, b in waiting_blocks.items():
                required = set([x[0] for x in b.depends_on])
                if required - available:
                    # Don't have results of all dependencies yet
                    continue

                # Gather extra compute arguments.
                kwargs = dict()
                for dependency_dims, dependency_name in b.depends_on.items():
                    kwargs[dependency_name] = results[dependency_dims]

                    # We must also pass the dependency domain.
                    # But to call .domain(), we need to find the original Block
                    # that produced the result :-(
                    for _b in self.model_blocks:
                        if _b.dimensions == dependency_dims:
                            break
                    else:
                        raise RuntimeError("Can't happen")
                    kwargs.update(_b.domain(data_tensor))

                results[b_dims] = b.compute(data_tensor, ptensor, **kwargs)
                del waiting_blocks[b_dims]

                # 'continue' on the outer while loop.
                break
            if len(results) > len(available):
                continue

            # Find two blocks to matrix multiply.

            # Find a block with a final dimension (S1 or S2)
            (b1, b1_dims), observable_dim = self._find_block(
                results,
                has_dim=self.final_dimensions)

            # Does the block have all final dimensions? Then we are done!
            if all([obs in b1_dims for obs in self.final_dimensions]):
                result = tf.squeeze(b1).reshape((self.batch_size,))
                return result

            # Find a block containing one of the non-observable dimensions in b1
            (b2, b2_dims), inner_dim = self._find_block(
                results,
                has_dim=set(b1_dims) - {observable_dim},
                exclude=b1)
            other_b2_dim = (set(b2_dims) - {inner_dim}).pop()
            assert other_b2_dim != observable_dim

            # Blocks must be rank-3 tensors
            # TODO: if we relax this, what models would we be able to support?
            for b, dims in [(b1, b1_dims), (b2, b2_dims)]:
                assert len(dims) + 1 == len(b.shape) == 3, "Need rank-3 blocks"

            # Reorder dimensions for matmul
            if b1_dims.index(inner_dim) != 1:
                b1 = tf.transpose(b1, [0, 2, 1])
            if b2_dims.index(inner_dim) != 0:
                b2 = tf.transpose(b1, [0, 2, 1])

            # Multiply the matching index to get a new block
            b = b1 @ b2
            dims = (observable_dim, other_b2_dim)

            # Update the block dict
            del results[b1_dims]
            del results[b2_dims]
            results[dims] = b

    def random_truth(self, n_events, **params):
        # First block provides the 'deep' truth (energies, positions, time)
        return self.model_blocks[0].random_truth(n_events, **params)

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
            d = b.annotate(d,
                           _do_checks=(b != self.model_blocks[0]))

    def mu_before_efficiencies(self, **params):
        return self.model_blocks[0].mu_before_efficiencies(**params)
