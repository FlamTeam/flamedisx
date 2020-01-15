0.4.0 / 2019-01-15
-------------------
- Many changes to objectives and inference (#39, #40)
- Add tilt to objective for interval/limit searches
- one_parameter_interval -> limit and interval methods
- Optimizers use bounds
- Tolerance option homogenization (first pass)
- Auto-guess limits

0.3.1 / 2019-11-26
------------------
- Performance improvements and cleanup (#58)
- Improve one_parameter_interval arguments (#56)
- Add Tutorial output to flamedisx-notebooks (#56)
- Bugfixes (#57)

0.3.0 / 2019-11-19
------------------
- Split off notebook folder to flamedisx-notebooks
- Pass source specific parameters correctly (#51)
- Flexible event padding (#54)
- SciPy optimizer and optimizer settings (#54)
- one_parameter_interval (#54)
- Bugfixes (#46, #55, #51)
- Unify optimizers (#54)

0.2.2 / 2019-10-30
------------------
- Minuit optimizer (#40)
- Likelihood simulator (#43, #44)
- Updates to NRSource (#40)

0.2.1 / 2019-10-24
------------------
- Workaround for numerical errors (#38, #39)

0.2.0 / 2019-10-11
------------------
- Spatially dependent rates (#27)
- Time dependent energy spectra (#24)
- XENON1T SR1-like model / fixes (#22, #32)
- Switch optimizer to BFGS + Hessian (#19)
- Multiple source support (#14)
- Optimization (#13)
- Bugfixes / refactor (#18, #20, #21, #28, #30, #31, #35)

0.1.2 / 2019-07-24
-------------------
- Speedup ER computation, add tutorial (#11)
- Optimize lookup-axis1 (#10)

0.1.1 / 2019-07-21
-------------------
- 5x speedup for Hessian (#9)
- Fix pip install

0.1.0 / 2019-07-16
-------------------
- Batching (#7)
- Inference (#6)
- Ported to tensorflow / GPU support (#1, #2, #3, #5)

0.0.1 / 2019-03-17
------------------
- Initial numpy-based version
