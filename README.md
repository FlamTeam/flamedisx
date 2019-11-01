Flamedisx
==========

Fast likelihood analysis in more dimensions for xenon TPCs.

[![Build Status](https://travis-ci.org/FlamTeam/flamedisx.svg?branch=master)](https://travis-ci.org/FlamTeam/flamedisx)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3524865.svg)](https://doi.org/10.5281/zenodo.3524865)

By Jelle Aalbers, Bart Pelssers, and Cristian Antochi

Description
-------------

Flamedisx aims to increase the practical number of dimensions (e.g. s1, s2, x, 
y, z and time) and parameters (g1, g2, recombination model coefficients, 
electron lifetime, ...) in LXe TPC likelihoods.

Traditionally, we evaluate (the probability density functions used in) our likelihoods using histograms created from high-statistics MC simulations. We precompute these histograms for several parameter combinations, then interpolate between them during inference ("vertical template morphing" in collider physics jargon). The precomputation time is exponential in the number of likelihood/histogram dimensions *and* the number of parameters used.

Flamedisx instead computes the probability density directly at each observed event, without using MC integration (or approximating the model). The commonly used LXe emission model is simple enough that the integral equivalent to an MC simulation can be computed with a few matrix multiplications, at a speed of a few ms -- instead of a high-statistics MC simulation that takes O(minute) or more.

This has several advantages:
  - Each event has its "private" detector model computation at the observed (x, y, z, time), so making the likelihood time- and position dependent incurs no additional computational burden. 
  - The likelihood for a dataset takes O(seconds) to compute, so we can do this at each of optimizer's proposed points during inference. We thus remove the precomputation step exponential in the number of parameters -- and can thus fit a great deal more parameters.
  - Since the likelihood consists of deterministic matrix multiplications, it can be implemented in tensorflow. This enables automatic differentiation, which unlocks the gradient during minimizing, drastically reducing the number of needed interactions for a fit or profile likelihood.
  
Note this is under construction, so it probably has some bugs and little documentation.


