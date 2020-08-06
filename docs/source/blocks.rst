===============================
Customizing the flamedisx model
===============================



Common customization
---------------------
The flamedisx tutorial describes the most common case: customizations by changing a model function.

For example, if you want to change a source's energy spectrum, yield functions, detector response functions, etc., you can do that without knowing anything about the advanced topics described in this page.

Advanced customization
----------------------

Sometimes, you need to apply customizations that go beyond overriding a source's model functions. For example, suppose you want to change the distribution of S2 areas to a skewed Gaussian, or add an additional smearing at some stage in the process.

However, some changes look like they require fundamental changes, but actually do not. For example:

* Two binomial efficiencies following each other are mathematically identical to using a binomial with ``p = p1 * p2``;
* A binomial efficiency following a Poisson smearing is equivalent to changing the Poisson mean to ``mu = mu_orig * p``;
* Two Gaussian smearings following each other are equivalent to one, with  the variance added in quadrature ``s^2 = s1^2 + s2^2``.

If you are looking to add a non-parametric/non-physical background source -- e.g. for accidental coincidences, anomalous backgrounds, etc -- look at the :py:class:`~flamedisx.source.ColumnSource` instead. This is a simple model that does not use the block structure.

The rest of this page describes how to change the core structure of the flamedisx model. To do this, you first need to know a bit about how it works behind the scenes.

Blocks
-------
Flamedisx's sources are built from units called **blocks**. Each block takes care of one step in the computation, such as converting energies to generated quanta, or converting electrons to S2. A block brings all aspects of that computation together: deterministic PDF computation, random simulation, and hidden variable bound estimation.

Each block is represented by a :py:class:`~flamedisx.block_source.Block` class, which has three main methods:

* :py:meth:`~flamedisx.block_source.Block._compute` is a block's main method. This returns a factor in the deterministic differential rate computation; for example, the probability of seeing the observed S2 for a range of possible number of detected electrons.
* The :py:meth:`~flamedisx.block_source.Block._simulate` method performs a Monte Carlo simulation of the block's process. For example, drawing one possible S2 integral value, given a simulated event of a certain number of detected electrons.

  * Simulated events have a special `p_accepted` column, starting out at 1.0 for each event. Blocks can multiply this with probabilities for passing various selections. At the end of the simulation, a random number will be drawn to determine whether each event actually passes the selections.

* The :py:meth:`~flamedisx.block_source.Block._annotate` method estimates bounds for hidden variables used in the `_compute`. For example, it determines a plausible range of detected electrons given the observed S2.

Think of `_simulate` as going in the physical/causal direction, `_annotate` as going backwards, and `_compute` as a (usually) direction-independent description of the process.

Most blocks in a source can be computed independently of the other blocks. The results of different blocks are matrix-multiplied together, representing convolution over hidden variables (such as number of produced electrons).
 
A few blocks instead directly take the result of another block and turn it into something else in the compute step. For example, `MakeNRQuanta` takes in an energy spectrum and converts it into a spectrum vs. number of produced quanta in the nuclear recoil process.



Using blocks to build a Source
------------------------------

This is the easy part: inherit from :py:class:`~flamedisx.block_source.BlockModelSource` and specify the blocks you use in the `model_blocks` tuple. Flamedisx' :py:class:`~flamedisx.lxe_sources.ERSource` and :py:class:`~flamedisx.lxe_sources.NRSource` are both examples of this.

The source will operate as follows:
 * When **computing** differential rates, we run a tensorflow graph that includes the `_compute` of all the blocks. If a block has dependencies, it's compute will of course be later in the graph than that of its dependents.
  * During **simulation**, we run `_simulate` of the blocks in the order you specified in `model_blocks`, starting with the first block. This is usually the block that creates the energy spectrum.
  * When **setting data** (e.g. when you create the source), we run `_annotate` of the blocks in reverse order. This way, you can first estimate hidden variables close to observables, then use those estimates for guessing deeper hidden variables. For example, you can use the estimated number of detected electrons to estimate the number of produced electrons.

If your source needs an `__init__`, you may want to start by calling `self.build_source_from_blocks()`, so all the correct attributes, methods, etc. get copied over from blocks. The function is harmless if called more than once.  Likely you'll call `super().__init__(*args, **kwargs)` only after doing whatever it is you want to do, since `Source.__init__` will set the data.


Blocks in detail
-------------------

Besides the main three methods, blocks usually specify additional attributes that describe their behavior to the source.

Static attributes
=================
`model_attributes` is a tuple of strings of Block attributes that should be exposed in the source. Setting one of these attributes in the Source will override their value.

For example, the :py:class:`~flamedisx.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum` block has the `energies` and `rates_vs_energy` attributes to specify the the source's discretized energy spectrum. The `ERSource` and `NRSource` both use this block, so you can write::

    import flamedisx as fd
    import tensorflow as tf

    class MySource(fd.ERSource):
        """Flat ER spectrum from 0 to 5 keV"""
        energies = tf.linspace(0., 5., 100, dtype=fd.float_type())
        rate_vs_energy = tf.ones(100, dtype=fd.float_type())

to change the energy spectrum. This is simply another form of 'common customization', just like the more common model function overriding.

Do not try to change static attributes after a source is initialized. They are called static for a reason. (If you change them despite this warning, the change will not be propagated from the `Source` to the `Block`, and code in the `Block` will still see the old attribute and cause you a headache.)

You can find a string-tuple of all static attributes for a source in the `.model_attributes` attribute.


Model functions
=================

Just like `model_attributes` exposes attributes, `model_functions` and `special_model_functions` expose methods to the source. Each are a tuple of strings of method names.

In your block, you call model functions in different ways:
  * In `_compute`, call `self.gimme('your_model_function', data_tensor=data_tensor, ptensor=ptensor)`.
  * In `_simulate` and `_annotate`, call `self.gimme_numpy('your_model_function')`.

This takes care of several things:
  * Positional arguments are filled in with columns from the data;
  * Keyword arguments are filled in with inference parameters.
  * For `gimme_numpy`, you will get back a numpy array (rather than a TensorFlow tensor).
Never call a model function directly from your code!

`special_model_functions` take an extra positional argument when they are called. It's up to you what this represents; usually this is used to pass variables. The extra argument (called `bonus_arg` in flamedisx code) is passed as the first argument after `self`.

If a model function is 'special' in this way, you must list it in **both** model_functions and special_model_functions

As an example, the :py:class:`~flamedisx.lxe_blocks.quanta_generation.MakeNRQuanta` block exposes a :py:meth:`~flamedisx.lxe_blocks.quanta_generation.MakeNRQuanta.lindhard_l` model function that parametrizes the Lindhard process (nuclear recoil energy losses as heat) as a function of energy. Sources using this block can define a new `lindhard_l` method to override this. The modelling sections of the tutorial illustrate model function overriding in detail.

You can find string-tuples of all regular and special model functions for a source in the `.model_functions` attribute. (Special model functions are also listed here, and separately in `.special_model_functions`.)

Dimensions
=================

The `dimensions` tuple names the dimensions of the `_compute` output. Without this we wouldn't know how to combine the results of blocks. The batch/event dimension is not named.

For example:
  * For :py:class:`~flamedisx.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum`, this is `('deposited_energy',)`, since `_compute` outputs a one-dimensional array per event, the differential rate as a function of deposited energy.
  * For :py:class:`~flamedisx.lxe_blocks.quanta_generation.MakePhotonsElectronsBinomial`, this is `('electrons_produced', 'photons_produced')`, since it outputs a two-dimensional array per event, the differential rate as a function of the produced number of photons and electrons.

Dependencies
=================

Sometimes you can only compute a block once you know the result of another block. If so, specify this block in the `depends_on` tuple.

For example, `depends_on = ((('quanta_produced',), 'rate_vs_quanta'),)` means the block needs the result of some block with `dimensions = ('quanta_produced',)`. Depending on the source, this could be provided by :py:class:`~flamedisx.lxe_blocks.quanta_generation.MakeNRQuanta` or :py:class:`~flamedisx.lxe_blocks.quanta_generation.MakeERQuanta`.

The dependency result and its domain (i.e. the x-values corresponding to the y-values the block returned) will be passed to `_compute` as extra arguments. In the above example, `_compute` will get `quanta_produced` and `rate_vs_quanta` as extra arguments. The former is the domain, the latter the result.


Frozen functions and array columns
===================================

To be written -- see :py:class:`~flamedisx.lxe_sources.WIMPsource` for an example in the meantime.


The first block of a source
-----------------------------

This is usually the block specifying the energy spectrum. It is special in several ways. 

Some restrictions are relaxed:
  * It does not have a `_simulate` method.
  * `_annotate` can (but does not have to) be omitted. There is no need to estimate bounds for its dimension (deposited energy), as the block returns the full energy spectrum for each event.

Other restrictions are added:
  * You must inherit from `FirstBlock`, rather than `Block`
  * It must specify a `domain` method, returning a dictionary mapping its dimension (e.g. deposited_energy) to the range of values for which `_compute` returns results.
  * It must implement a `random_truth` method, taking `n_events` and a parameter dictionary, returning a dataframe with a number of simulated events.
  * It must implement a `mu_before_efficiencies` method, taking a parameter dictionary and returning the number of expected events directly from the spectrum (i.e. before any efficiencies) given these parameters.
  * It must specify a `validate_fix_truth` method, taking and returning a fixed truth specification.

See :py:class:`~flamedisx.lxe_blocks.energy_spectrum.FixedShapeEnergySpectrum` for an example and more details.
