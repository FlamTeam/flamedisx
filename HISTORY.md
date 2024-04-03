2.1.0 / 2024-04-03
------------------
- Mu estimation options, including grid interpolation (#222, #242, #285)
- Template sources and morphing (#241, #317, #318)
- NEST sources: update to NEST 2.3.0 (#249), add beta/gamma models (#252)
- Default sources: Reconstruction bias/smearing (#273)
- Non-integer dimensions (#258)
- Non-asymptotic inference (#269, #345)
- Reservoir source that caches rates (#247)
- Fixes and cleanups (#231, #270, #284, #327, #329)
- Tested on tensorflow 2.6.1 / numpy 1.26.4

2.0.0 / 2022-05-20
------------------
- FlameNEST models fully implemented (https://arxiv.org/abs/2204.13621)
- NEST models for pre-quanta processes (#205)
- Bayesian bounds estimation (#174)
- NEST source fixes (#152)
- Fix covariance used in `LogLikelihood.summary` (#176)
- Avoid calculating `produced_quanta = 0` probability (#181)
- `electron_loss` model function (#193)
- Add exposure parameter to WIMPEnergySpectrum (#223)
- Always reset data index (#225)
- XENON sources:
  - Wall events model (#143)
  - `double_pe_fraction` model function (#208)
  - Updates to config defaults (#209)
  - Spatially dependent drift field map (#221)
  - Configurable drift field, S2 AFT (#213, #218)

1.5.0 / 2021-06-29
------------------
- Variable stepping, support for high-energy models (#127)
- NEST models for post-quanta processes (#136)
- Configuration system (#140, #147)
- XENON1T: Fix S2 acceptance (#138) and unused imports (#128)
- Update block system documentation (#139)

1.4.1 / 2021-04-20
------------------
- Stabilize default optimizer with better parameter scaling (#114)
- XENONnT: Support reading data from private repository (#115)
- XENON1T: Variable elife (#118)
- XENON1T: Npz resource reading (#123)

1.4.0 / 2021-03-05
------------------
- Fix 'sticky defaults' bug (#110)
- Enable GitHub Actions and Dependabot (#109)
- Documentation updates (#92, [notebooks#3](https://github.com/FlamTeam/flamedisx-notebooks/pull/3))
- Likelihood `defaults` support, simulate argument fixes (#103)
- SpatialRateEnergySpectrum: Simplify API (#100) and fix draw_positions (#105)
- WIMPEnergySpectrum: Accept event times slightly out of range (#99)
- Do not round photons_detected_mle (#91)
- XENON1T: fix S2 acceptance (#97) and name reconstruction efficiency pivots (#102)

1.3.0 / 2020-08-25
------------------
- Block system (#81)
- Documentation (#81)
- Bugfixes (#83, #87, #89)

1.2.0 / 2020-07-21
------------------
- Access BBF data and XENON-utilities (#80)
- Double photoelectron emission modeling (#78)
- Optimization improvements (#76)
- Bugfix (#79)

1.1.0 / 2020-07-09
------------------
- Nonlinear constraint limit setting (experimental) (#70)
- Dimension scaling inside optimizers (#72)
- Auto-guess rate multipliers (#74)
- Python 3.8 builds (#73)
- Add sanity checks on input and guess (#69)

1.0.0 / 2020-03-26
------------------
- Fiducial volume specification (#64)
- Added default cS1 cut (#63)
- Cleanup and optimizations (#63, #64, #65)

0.5.0 / 2020-01-31
------------------
- Autographed Hessian; use Hessian in the optimizer (#62)
- Check for optimizer failures (#61)
- Trace single-batch likelihood, but use numpy thereafter (#61)
- Fix simulation/data discrepancy in recombination fluctuation
- Adjust optimizer defaults
- Option to use time-averaged WIMP spectra

0.4.0 / 2020-01-15
-------------------
- Many changes to objectives and inference (#59, #60)
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
