import flamedisx as fd
import numpy as np
import typing as ty

export, __all__ = fd.exporter()

@export
class FrequentistUpperLimitRatesOnly():
    """NOTE: currently single dataset only

    Arguments:
        - xxx: yyy

    """

    def __init__(
            self,
            primary_source_name,
            sources: ty.Dict[str, fd.Source.__class__],
            arguments: ty.Dict[str, ty.Dict[str, ty.Union[int, float]]] = None,
            batch_size=100,
            max_sigma=None,
            max_sigma_outer=None,
            n_trials=None,
            defaults=None,
            ntoys=1):

        if arguments is None:
            arguments = dict()

        for key in sources.keys():
            if key not in arguments.keys():
                arguments[key] = dict()

        if defaults is None:
            defaults = dict()

        self.primary_source_name = primary_source_name

        # Create sources
        self.sources = {
            sname: sclass(**(arguments.get(sname)),
                          data=None,
                          max_sigma=max_sigma,
                          max_sigma_outer=max_sigma_outer,
                          batch_size=batch_size,
                          **defaults)
            for sname, sclass in sources.items()}

        assert self.primary_source_name in self.sources.keys(), 'Invalid primary source name'

        self.secondary_source_names = [source_name for source_name in self.sources.keys()
                                       if source_name != self.primary_source_name]

        # Create frozen source reservoir
        reservoir = fd.frozen_reservoir.make_event_reservoir(ntoys=ntoys, **self.sources)

        # Create likelihood
        self.log_likelihood = fd.LogLikelihood(sources={sname: fd.FrozenReservoirSource for sname, sclass in sources.items()},
                                               arguments = {sname: {'source_type': sclass, 'source_name': sname, 'reservoir': reservoir}
                                                            for sname, sclass in sources.items()},
                                               progress=False,
                                               batch_size=batch_size,
                                               free_rates=tuple([sname for sname in sources.keys()]))

        default_rm_bounds = {self.primary_source_name: (None, None)}
        for source_name in self.secondary_source_names:
            default_rm_bounds[source_name] = (None, None)

        self.log_likelihood.set_rate_multiplier_bounds(**default_rm_bounds)

    def toy_data(self):
        return self.log_likelihood.simulate()

    def set_data(self, data):
        self.log_likelihood.set_data(data)

    def test_statistic_tmu_tilde(self, mu_test):
        guess_dict = {f'{self.primary_source_name}_rate_multiplier': mu_test}

        for source_name in self.secondary_source_names:
            guess_dict[f'{source_name}_rate_multiplier'] = 1.

        bf_unconditional = self.log_likelihood.bestfit(guess=guess_dict)
