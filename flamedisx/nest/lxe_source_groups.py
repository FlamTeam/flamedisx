import typing as ty

import tensorflow as tf
import numpy as np

from tqdm import tqdm

import flamedisx as fd
from .. import nest as fd_nest

export, __all__ = fd.exporter()
o = tf.newaxis


class BlockModelSourceGroup(fd.BlockModelSource):
    def batched_differential_rate(self, progress=True, quanta_tensor_dict=None,
                                  electrons_full_dict=None, photons_full_dict=None, **params):
        """Return probabilities of events in a dataset given mono-energetic deposists for all energies
        in the base_source stepped/trimmed spectrum for every batch.
        Can optionally pass arguments to read in pre-computed values for the central block.
        """
        progress = (lambda x: x) if not progress else tqdm
        energies_all = []
        results_all = []

        for i_batch in progress(range(self.n_batches)):
            q = self.data_tensor[i_batch]

            if quanta_tensor_dict is not None:
                quanta_tensors = []
                electrons_full = []
                photons_full = []

                # Grab the saved central block values for the relevant energies, along with the electron and
                # photon domains corresponding to the saved values
                for energy in fd.tf_to_np(self.model_blocks[0].domain(data_tensor=q)['energy'][0]):
                    quanta_tensors.append(quanta_tensor_dict[str(energy)])
                    electrons_full.append(electrons_full_dict[str(energy)])
                    photons_full.append(photons_full_dict[str(energy)])

                quanta_tensors = tf.ragged.stack(quanta_tensors)
                electrons_full = tf.ragged.stack(electrons_full)
                photons_full = tf.ragged.stack(photons_full)
            else:
                quanta_tensors = None
                electrons_full = None
                photons_full = None

            energies, results = self.differential_rate(data_tensor=q, quanta_tensors=quanta_tensors,
                                                       electrons_full=electrons_full, photons_full=photons_full,
                                                       **params)

            # Return the energies and probabilities we are evaluating
            energies_all.extend(fd.tf_to_np(energies))
            results_all.extend(np.transpose(fd.tf_to_np(results)))

        return energies_all, results_all

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

    def _differential_rate_edges(self, data_tensor, ptensor, blocks, return_dims, already_stepped):
        """Multiply and store the probabilities corresponding to the blocks left and right of
        the central block.
        """
        results = {}

        # Ensure we only compute the relevant blocks
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

        return ({return_dims: results[return_dims]}), already_stepped

    def _differential_rate_central(self, data_tensor, ptensor, blocks, return_dims, already_stepped,
                                   quanta_tensors=None, electrons_full=None, photons_full=None):
        """Store the probabilities of the central block for all energies in the base_source stepped/
        trimmed spectrum for every batch.
        Can optionally pass arguments to read in pre-computed values for this block.
        """
        results = {}

        # Ensure we only compute the relevant blocks
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

            # Compute the block, optionally reading in pre-computed values
            if b.__class__ in self.model_blocks_read_in:
                r = b.compute(data_tensor, ptensor, quanta_tensors=quanta_tensors,
                              electrons_full=electrons_full, photons_full=photons_full, **kwargs)
            else:
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

        return ({return_dims: results[return_dims]}), already_stepped

    def _differential_rate(self, data_tensor, ptensor):
        """"Computation of probabilities of a batch of events given mono-energetic deposists
        for all energies in the base_source stepped/trimmed spectrum for this batch.
        """
        already_stepped = ()  # Avoid double-multiplying when accounting for stepping

        # We compute the left and right block multiplications once
        left, already_stepped = self._differential_rate_edges(data_tensor, ptensor,
                                                              self.model_blocks_left,
                                                              ('s1', 'photons_produced'), already_stepped)
        right, already_stepped = self._differential_rate_edges(data_tensor, ptensor,
                                                               self.model_blocks_right,
                                                               ('s2', 'electrons_produced'), already_stepped)
        # We compute the central block for every energy in the base_source stepped/trimmed spectrum for this batch
        centre, _ = self._differential_rate_central(data_tensor, ptensor,
                                                    self.model_blocks_centre,
                                                    ('electrons_produced', 'photons_produced'), already_stepped)

        assert (len(left.keys()) == len(right.keys()) == len(centre.keys()) == 1)

        left_dims = next(iter(left))
        right_dims = next(iter(right))
        centre_dims = next(iter(centre))

        left_block = next(iter(left.items()))[1]
        right_block = next(iter(right.items()))[1]

        # We mutliply the pre-computed left/right block multiplication results with the central block
        # corresponding to each energy, and retur those energies/results

        results_collection = tf.TensorArray(fd.float_type(), size=0, dynamic_size=True)

        i = tf.constant(0)
        for centre_block in next(iter(centre.items()))[1]:
            left_centre_dims, r_left_centre = self.multiply_block_results(
                left_dims, centre_dims, left_block, centre_block)

            final_dims, r = self.multiply_block_results(
                left_centre_dims, right_dims, r_left_centre, right_block)

            results_collection = results_collection.write(i, tf.reshape(tf.squeeze(r), (self.batch_size,)))
            i += 1

        energies = self.model_blocks[0].domain(data_tensor=data_tensor)['energy']
        results = results_collection.stack()

        return energies, results

    def _differential_rate_read_in(self, data_tensor, ptensor,
                                   quanta_tensors, electrons_full, photons_full):
        """"Computation of probabilities of a batch of events given mono-energetic deposists
        for all energies in the base_source stepped/trimmed spectrum for this batch.
        We execute this function when reading in values for the central block.
        """
        already_stepped = ('ions_produced',)  # Avoid double-multiplying when accounting for stepping.
        # We have already accounted for the ion stepping

        # We compute the left and right block multiplications once
        left, already_stepped = self._differential_rate_edges(data_tensor, ptensor,
                                                              self.model_blocks_left,
                                                              ('s1', 'photons_produced'), already_stepped)
        right, already_stepped = self._differential_rate_edges(data_tensor, ptensor,
                                                               self.model_blocks_right,
                                                               ('s2', 'electrons_produced'), already_stepped)
        # We compute the central block for every energy in the base_source stepped/trimmed spectrum for this batch
        centre, _ = self._differential_rate_central(data_tensor, ptensor,
                                                    self.model_blocks_centre,
                                                    ('electrons_produced', 'photons_produced'), already_stepped,
                                                    quanta_tensors=quanta_tensors,
                                                    electrons_full=electrons_full, photons_full=photons_full)

        assert (len(left.keys()) == len(right.keys()) == len(centre.keys()) == 1)

        left_dims = next(iter(left))
        right_dims = next(iter(right))
        centre_dims = next(iter(centre))

        left_block = next(iter(left.items()))[1]
        right_block = next(iter(right.items()))[1]

        # We mutliply the pre-computed left/right block multiplication results with the central block
        # corresponding to each energy, and retur those energies/results

        results_collection = tf.TensorArray(fd.float_type(), size=0, dynamic_size=True)

        i = tf.constant(0)
        for centre_block in next(iter(centre.items()))[1]:
            left_centre_dims, r_left_centre = self.multiply_block_results(
                left_dims, centre_dims, left_block, centre_block)

            final_dims, r = self.multiply_block_results(
                left_centre_dims, right_dims, r_left_centre, right_block)

            results_collection = results_collection.write(i, tf.reshape(tf.squeeze(r), (self.batch_size,)))
            i += 1

        energies = self.model_blocks[0].domain(data_tensor=data_tensor)['energy']
        results = results_collection.stack()

        return energies, results


class BlockNotFoundError(Exception):
    pass


@export
class nestERSourceGroup(BlockModelSourceGroup, fd_nest.nestERSource):
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
        fd_nest.FixedShapeEnergySpectrumER,
        fd_nest.SGMakePhotonsElectronER)

    model_blocks_read_in = (
        fd_nest.SGMakePhotonsElectronER,)

    def __init__(self, caching=False, **kwargs):
        if caching is True:
            self.no_step_dimensions = self.no_step_dimensions + \
                ('photons_produced',
                 'electrons_produced')

        super().__init__(**kwargs)


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

    model_blocks_read_in = (
        fd_nest.SGMakePhotonsElectronsNR,)

    def __init__(self, caching=False, **kwargs):
        if caching is True:
            self.no_step_dimensions = self.no_step_dimensions + \
                ('photons_produced',
                 'electrons_produced')

        super().__init__(**kwargs)
