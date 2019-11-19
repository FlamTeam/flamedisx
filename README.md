Flamedisx
==========

Fast likelihood analysis in more dimensions for xenon TPCs.

[![Build Status](https://travis-ci.org/FlamTeam/flamedisx.svg?branch=master)](https://travis-ci.org/FlamTeam/flamedisx)
[![DOI](https://zenodo.org/badge/176141558.svg)](https://zenodo.org/badge/latestdoi/176141558)

By Jelle Aalbers, Bart Pelssers, and Cristian Antochi


Tutorial and documentation
---------------------------

See the [Tutorial](https://github.com/FlamTeam/flamedisx-notebooks/blob/master/Tutorial.ipynb) and other notebooks in our separate [notebooks repository](https://github.com/FlamTeam/flamedisx-notebooks).

Description
-------------

Flamedisx aims to increase the practical number of dimensions (e.g. s1, s2, x, 
y, z and time) and parameters (g1, g2, recombination model coefficients, 
electron lifetime, ...) in LXe TPC likelihoods.

Traditionally, we evaluate our signal and background models by filling histograms with high-statistics Monte Carlo simulations. However, the LXe emission model can be expressed so that  the integral equivalent to an MC simulation can be computed with a few matrix multiplications. Flamedisx uses this to compute the probability density directly at each observed event, without using MC integration. 

This has several advantages:
  - Each event has its "private" detector model computation at the observed (x, y, z, time), so it is easy and cheap to add  time- and position dependences to the likelihood.
  - Since the likelihood for a dataset takes O(seconds) to compute, we can do this at each of optimizer's proposed points during inference. We thus remove a histogram precomputation step exponential in the number of parameters, and can thus fit a great deal more parameters.
  - By implementing the signal model in tensorflow, the likelihood becomes differentiable. Using the gradient during fitting drastically reducing the number of needed interactions for a fit or profile likelihood.
  
Note this is still under construction / development, so it probably has some bugs and little documentation.


