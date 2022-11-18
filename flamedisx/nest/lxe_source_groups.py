import typing as ty

import tensorflow as tf

from tqdm import tqdm

import flamedisx as fd
from .. import nest as fd_nest

export, __all__ = fd.exporter()
o = tf.newaxis


class BlockModelSourceGroup(fd.BlockModelSource):
    def batched_differential_rate(self, progress=True, **params):
        """Return numpy array with differential rate for all events.
        """
        progress = (lambda x: x) if not progress else tqdm
        y = []
        for i_batch in progress(range(self.n_batches)):
            q = self.data_tensor[i_batch]
            self.differential_rate(data_tensor=q, **params)
        #     y.append(fd.tf_to_np(self.differential_rate(data_tensor=q,
        #                                                 **params)))
        # print(y)

    # def _differential_rate_left(self, data_tensor, ptensor):

    @staticmethod
    def _find_block(blocks,
                    has_dim: ty.Union[list, tuple, set],
                    exclude: fd.block_source.Block = None):
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

    def _differential_rate_edges(self, data_tensor, ptensor, blocks, return_dims):
        results = {}
        already_stepped = ()  # Avoid double-multiplying to account for stepping

        for b in self.model_blocks:
            if b.__class__ not in blocks:
                continue

            b_dims = b.dimensions
            # These are the the dimensions we will do variable stepping over
            scaling_dims = b.dimensions + tuple([bonus_dimension[0] for
                                                bonus_dimension in b.bonus_dimensions
                                                if bonus_dimension[1] is True])

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
                if ((dim in self.inner_dimensions) or (dim in self.bonus_dimensions)) and \
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

        return({return_dims: results[return_dims]})

    def _differential_rate_central(self, data_tensor, ptensor, blocks, return_dims):
        results = {}
        already_stepped = ()  # Avoid double-multiplying to account for stepping

        for b in self.model_blocks:
            if b.__class__ not in blocks:
                continue

            b_dims = b.dimensions
            # These are the the dimensions we will do variable stepping over
            scaling_dims = b.dimensions + tuple([bonus_dimension[0] for
                                                bonus_dimension in b.bonus_dimensions
                                                if bonus_dimension[1] is True])

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
                if ((dim in self.inner_dimensions) or (dim in self.bonus_dimensions)) and \
                        (dim not in self.no_step_dimensions) and \
                        (dim not in already_stepped):
                    steps = self._fetch(dim+'_steps', data_tensor=data_tensor)
                    step_mul = tf.repeat(steps[o, :], tf.shape(r)[0], axis=0)
                    step_mul = tf.repeat(step_mul[:, :, o],
                                         tf.shape(r)[2], axis=2)
                    step_mul = tf.repeat(step_mul[:, :, :, o],
                                         tf.shape(r)[3], axis=3)
                    r *= step_mul
                    already_stepped += (dim,)

            results[b_dims] = r

        return({return_dims: results[return_dims]})

    def _differential_rate(self, data_tensor, ptensor):
        left = self._differential_rate_edges(data_tensor, ptensor, self.model_blocks_left, ('s1', 'photons_produced'))
        right = self._differential_rate_edges(data_tensor, ptensor, self.model_blocks_right, ('s2', 'electrons_produced'))

        centre = self._differential_rate_central(data_tensor, ptensor, self.model_blocks_centre, ('electrons_produced', 'photons_produced'))


class BlockNotFoundError(Exception):
    pass


@export
class nestNRSourceGroup(BlockModelSourceGroup, fd_nest.nestNRSource):
    model_blocks_left = (
        fd_nest.DetectPhotons,
        fd_nest.MakeS1Photoelectrons,
        fd_nest.DetectS1Photoelectrons,
        fd_nest.MakeS1)

    model_blocks_right = (
        fd_nest.DetectElectrons,
        fd_nest.MakeS2Photons,
        fd_nest.DetectS2Photons,
        fd_nest.MakeS2Photoelectrons,
        fd_nest.MakeS2)

    model_blocks_centre = (
        fd_nest.FixedShapeEnergySpectrumNR,
        fd_nest.SGMakePhotonsElectronsNR)
