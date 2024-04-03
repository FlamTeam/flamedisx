Flamedisx
==========

Fast likelihood analysis in more dimensions for xenon TPCs.

![Build Status](https://github.com/FlamTeam/flamedisx/actions/workflows/test_flamedisx.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/flamedisx/badge/?version=latest)](https://flamedisx.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/176141558.svg)](https://zenodo.org/badge/latestdoi/176141558)
[![ArXiv number](https://img.shields.io/badge/physics.ins--det-arXiv%3A2003.12483-%23B31B1B)](https://arxiv.org/abs/2003.12483)
[![Join the chat at https://gitter.im/AxFoundation/strax](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/FlamTeam/flamedisx)


Flamedisx aims to increase the practical number of dimensions and parameters in likelihoods for liquid-xenon (LXe) detectors, which are leading the field of direct dark matter detection.

Traditionally, particle physicists compute signal and background models by filling histogram 'templates' with high-statistics Monte Carlo (MC) simulations. However, the LXe model can also be computed with a series of (large) matrix multiplications, equivalent to the integral approximated by the MC simulation. Using TensorFlow makes this computation differentiable and GPU-scalable, so it can be used practically for fitting and statistical inference.

The result is a better sensitivity, since the likelihood can use all observables, and more robust fits, because using simultaneous correlated nuisance parameters no longer requires challenging interpolation and template morphing.



Getting started
---------------------------

To get started, [Launch our tutorial on Colaboratory](https://colab.research.google.com/github/FlamTeam/flamedisx-notebooks/blob/master/Tutorial.ipynb), or view it statically on [GitHub](https://github.com/FlamTeam/flamedisx-notebooks/blob/master/Tutorial.ipynb) or [ReadTheDocs](https://flamedisx.readthedocs.io/en/latest/tutorial.html).

Our [paper](https://arxiv.org/abs/2003.12483) gives a detailed description of Flamedisx, and compares Flamedisx quantitatively to traditional template-based methods.

If you want all the details, see the [Flamedisx Documentation](https://flamedisx.readthedocs.io) and our [Notebooks repository](https://github.com/FlamTeam/flamedisx-notebooks).



FlameNEST
-----------

[![arXiv](https://img.shields.io/badge/arXiv-2204.13621-b31b1b.svg)](https://arxiv.org/abs/2204.13621)

Since version 2.0.0, flamedisx includes an implementation of electronic and nuclear recoil models from the [Noble Element Simulation Technique](https://nest.physics.ucdavis.edu/). To use this, use sources from the ``fd.nest`` subpackage, e.g. ``fd.nest.ERSource``. See the [flameNEST paper](https://arxiv.org/abs/2204.13621) for a detailed description and validation.

As of April 2024, we implement NEST version 2.3.0, which was released November 2021 (see [#249](https://github.com/FlamTeam/flamedisx/pull/249)).
