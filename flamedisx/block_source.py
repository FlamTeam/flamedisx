import typing as ty

import numpy as np
import pandas as pd
import tensorflow as tf

import flamedisx as fd
export, __all__ = fd.exporter()


@export
class Block:
    """One part of a BlockSource model.

    For example, P(electrons_detected | electrons_produced).
    """
    dimensions: ty.Tuple[str]

    depends_on: ty.Typle[str] = tuple()
    model_functions: ty.Tuple[str] = tuple()

    def __init__(self, source):
        self.source = source
        assert len(self.dimensions) in (1, 2), \
            "Blocks must output 1 or 2 dimensions"

    def gimme(self, *args, **kwargs):
        """Shorthand for self.source.gimme, see docs there"""
        return self.source.gimme(*args, **kwargs)

    def domain(self, data_tensor):
        """Return dictionary mapping dimension -> domain"""
        if len(self.dimensions) == 1:
            return {self.dimensions[0]:
                        self.source.domain(self.dimensions[0], data_tensor)}
        return dict(zip(self.dimensions,
                        self.source.cross_domains(*self.dimensions,
                                                  data_tensor=data_tensor)))

    def compute(self, data_tensor, ptensor):
        kwargs = dict()
        # TODO: pass observable dim
        # observed = self._fetch(signal_name[quanta_type], data_tensor=data_tensor)
        # Pass domains
        kwargs.update(self.domain(data_tensor))
        # TODO: Pass results of dependencies and their domains
        # for block_name, block in self.dependencies.items():
        #     kwargs[block_name] = block.compute(data_tensor, ptensor)
        #     kwargs[block_name + '_domain'] = block.domain()
        return self._compute(data_tensor, ptensor, **kwargs)

    def simulate(self, d):
        # TODO: check necessary columns are present?
        return_value = self._simulate(d)
        assert return_value is None, f"_simulate of {self} should return None"
        # TODO: check necessary columns were actually added

    def annotate(self, d):
        """Add _min and _max for each dimension to d in-place"""
        # TODO: check necessary columns are present?
        return_value = self._annotate(d)
        assert return_value is None, f"_annotate of {self} should return None"
        # TODO: check necessary columns were actually added

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
    observables: tuple

    def __init__(self, *args, **kwargs):
        # TODO: Collect model functions from the different block
        # TODO: set/override static attributes for each block
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
        # Calculate individual blocks
        blocks = {b.dimensions(): b.compute(data_tensor, ptensor)
                  for b in self.model_blocks}

        # Matrix multiply until a scalar remains for each event.
        while len(blocks) > 1:

            # Find a block with an observable dimension (S1 or S2)
            (b1, b1_dims), observable_dim = self._find_block(
                blocks,
                has_dim=self.observables)

            # Find a block containing one of the non-observable dimensions in b1
            (b2, b2_dims), inner_dim = self._find_block(
                blocks,
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
            del blocks[b1_dims]
            del blocks[b2_dims]
            blocks[dims] = b

        # Get the last remaining block; return scalar differential rate
        # for each event in the batch.
        result = next(iter(blocks.values()))
        result = tf.squeeze(result).reshape((self.batch_size,))
        return result

    def random_truth(self, n_events, **params):
        # First block provides the 'deep' truth (energies, positions, time)
        return self.model_blocks[0].random_truth(n_events, **params)

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
            d = b.annotate(d)

    def mu_before_efficiencies(self, **params):
        raise NotImplementedError

    def random_truth(self, n_events, fix_truth=None, **params):
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        data = self.random_truth_observables(n_events)
        data = self._add_random_energies(data, n_events)

        if fix_truth is not None:
            # Override any keys with fixed values defined in fix_truth
            fix_truth = self.validate_fix_truth(fix_truth)
            for k, v in fix_truth.items():
                data[k] = v

        return pd.DataFrame(data)
