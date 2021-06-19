# Example flamedisx configuration, .py format
#
# You can use these files to store common detector defaults, or fit results.
# Load them by passing a path to the `config` argument of `Source.__init__`,
# or `Source.set_data`, or `Source.set_defaults`,
# or by setting 'config' key in the `defaults` argument of `Likelihood.__init__`
#
# Any parameter, model attribute, or model function with constant default
# can be overridden here.
#

elife = 987_654.
# er_pel_a = 15.0
# ...

photon_detection_eff = 0.11
# work = 0.0137
# ...
