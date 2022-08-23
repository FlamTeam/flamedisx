import typing as ty
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


def make_event_reservoir(ntoys: int = None,
                         **sources):
    """Generate an annotated reservoir of events to be used in FrozenReservoirSource s.

    Arguments:
        - ntoys: number of toy MCs this reservoir will be used to generate (optional).
        - sources: pass in source instances to be used to build the reservoir, like
            'source1'=source1(args, kwargs), 'source2'=source2(args, kwargs), ...
    """
    default_ntoys = 1000

    assert len(sources) != 0, "Must pass at least one source instance to make_event_reservoir()"

    if ntoys is None:
        ntoys = default_ntoys

    dfs = []
    for sname, source in sources.items():
        n_simulate = int(ntoys * source.mu_before_efficiencies())

        sdata = source.simulate(n_simulate)
        sdata['source'] = sname
        dfs.append(sdata)

    data_reservoir = pd.concat(dfs, ignore_index=True)

    for sname, source in sources.items():
        source.set_data(data_reservoir)
        data_reservoir[f'{sname}_diff_rate'] = source.batched_differential_rate()

    return data_reservoir


@export
class FrozenReservoirSource(fd.ColumnSource):
    """Source that looks up precomputed differential rates in a column source,
    with the added ability to simulate.

    Arguments:
        - source_type: base flamedisx source class.
        - source_name: name given to the base source; must match the source name
            in the reservoir.
        - source_kwargs: any kwargs needed to instantiate the base source.
        - reservoir: dataframe of events with a column 'source' showing the
            {source_name} of the base source each event came from, and columns
            '{sorce_name}_diff_rate' with the differential rate of each event
            computed under all base sources that will have a FrozenReservoirSource used
            in the analysis.

    For other arguments, see flamedisx.source.Source
    """

    def __init__(self, source_type: fd.Source.__class__ = None, source_name: str = None,
                 source_kwargs: ty.Dict[str, ty.Union[int, float]] = None,
                 reservoir: pd.DataFrame = None,
                 *args, **kwargs):
        assert source_type is not None, "Must pass a source type to FrozenReservoirSource"
        assert source_name is not None, "Must pass a source name to FrozenReservoirSource"
        assert source_name in reservoir['source'].values, "The reservoir must contain events from this source type"

        if source_kwargs is None:
            source_kwargs = dict()

        self.source_name = source_name
        self.reservoir = reservoir
        source = source_type(**source_kwargs)

        self.column = f'{source_name}_diff_rate'
        self.mu = source.estimate_mu()

        super().__init__(*args, **kwargs)

    def random_truth(self, n_events, fix_truth=None, **params):
        if fix_truth is not None:
            raise NotImplementedError("FrozenReservoirSource does not yet support fix_truth")
        if len(params):
            raise NotImplementedError("FrozenReservoirSource does not yet support alternative parameters in simulate")

        return self.reservoir[self.reservoir['source'] == self.source_name].sample(n_events, replace=True)
