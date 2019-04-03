flamedisx
==========

Fast likelihood analysis in many dimensions for xenon TPCs.


Description
-------------

Flamedisx aims to increase the practical number of dimensions (e.g. s1, s2, x, 
y, z and time) and parameters (g1, g2, recombination model coefficients, 
electron lifetime, ...) in LXe TPC likelihoods.

Traditionally, we evaluate (the probability density functions used in) our likelihoods using histograms created from high-statistics MC simulations. We precompute these for several parameter space combinations and interpolate between them. This precomputation time is exponential in the number of likelihood/histogram dimensions *and* the number of parameters used.

Flamedisx instead computes the probability density directly at each observed event, without either MC integration or approximation methods. Our emission model is simple enough that the integral equivalent to the MC simulation can be computed with a few matrix multiplications, at a speed of a few ms for each event. This is in the current numpy/CPU-bound prototype implementation (measured on my laptop); with GPUs this can almost certainly be accelerated. The current bottleck is computing the binomial distribution's PMF.

This gives you several advantages:
  - Since each event has its "private" detector model computation at the observed (x, y, z, time), making the likelihood time- and position dependent incurs no additional computational burden. 
  - Since the likelihood for a dataset takes O(seconds) to compute, we can compute it at each of optimizer's proposed points during minimization. We thus remove the precomputaion step that was exponential in the number of parameters.
  - Since the likelihood is now simply a set of deterministic matrix multiplications it can be implemented in tensorflow/pytorch. This gets you automatic differentiation, so you know the gradient during minimizing, which drastically reduces the number of iterations needed during fitting.
  
  
Current limitations 
-------------------

- This is under construction, so it probably doesn't work.
- Electronic recoils only (NRs are planned)


