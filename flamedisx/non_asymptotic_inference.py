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

        # Create sources
        self.sources = {
            sname: sclass(**(arguments.get(sname)),
                          data=None,
                          max_sigma=max_sigma,
                          max_sigma_outer=max_sigma_outer,
                          batch_size=batch_size,
                          **defaults)
            for sname, sclass in sources.items()}

        reservoir = fd.frozen_reservoir.make_event_reservoir(ntoys=ntoys, **self.sources)

        self.log_likelihood = fd.LogLikelihood(sources={sname: fd.FrozenReservoirSource for sname, sclass in sources.items()},
                                               arguments = {sname: {'source_type': sclass, 'source_name': sname, 'reservoir': reservoir}
                                                            for sname, sclass in sources.items()},
                                               progress=False,
                                               batch_size=batch_size,
                                               free_rates=tuple([sname for sname in sources.keys()]))
