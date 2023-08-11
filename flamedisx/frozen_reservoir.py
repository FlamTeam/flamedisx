import typing as ty
import pandas as pd
import pickle as pkl
import numpy as np

import flamedisx as fd
export, __all__ = fd.exporter()


def make_event_reservoir(ntoys: int = None,
                         reservoir_output_name=None,
                         input_mus=None,
                         source_groups_dict=None,
                         quanta_tensor_dirs_dict=None,
                         save_comb=False,
                         read_comb=None,
                         **sources):
    """Generate an annotated reservoir of events to be used in FrozenReservoirSource s.

    Arguments:
        - ntoys: number of toy MCs this reservoir will be used to generate (optional).
        - sources: pass in source instances to be used to build the reservoir, like
            'source1'=source1(args, kwargs), 'source2'=source2(args, kwargs), ...
        - reservoir_output_name: if supplied, the filename the reservoir will be saved under.
        - input_mus: dictionary {sourcename: mu, ...} giving pre-computed mus for the sources.
    """
    default_ntoys = 1000

    assert len(sources) != 0, "Must pass at least one source instance to make_event_reservoir()"

    if ntoys is None:
        ntoys = default_ntoys

    if read_comb is not None:
        data_reservoir = pkl.load(open(read_comb, 'rb'))

        source_groups_already_calculated = []
        for _, source_group_class in source_groups_dict.items():
            assert isinstance(source_group_class, fd.nest.SourceGroup), "Must be using source groups here!"
            if source_group_class not in source_groups_already_calculated:
                if 'energies_diff_rates' in data_reservoir:
                    data_reservoir = data_reservoir.drop(columns=['energies_diff_rates'])
                data_reservoir = data_reservoir.rename(columns={f'{source_group_class.base_source.__class__.__name__}_energies_diff_rates': 'energies_diff_rates'})
                source_group_class.set_data(data_reservoir.copy())
                source_groups_already_calculated.append(source_group_class)

        for sname, source in sources.items():
            data_reservoir[f'{sname}_diff_rate'] = source_groups_dict[sname].get_diff_rate_source(source)

        cols = [c for c in data_reservoir.columns if (c[-9:] == 'diff_rate')]
        cols.append('source')
        data_reservoir = data_reservoir[cols]

        if reservoir_output_name is not None:
            data_reservoir.to_pickle(reservoir_output_name)

        return data_reservoir

    dfs = []
    for sname, source in sources.items():
        factor = 1.1
        while True:
            n_simulate = int(factor * ntoys * source.mu_before_efficiencies())
            sdata = source.simulate(n_simulate)
            sdata['source'] = sname

            if len(sdata) >= np.ceil(input_mus[sname]):
                dfs.append(sdata[0:int(np.ceil(input_mus[sname]))])
                break
            else:
                factor += 0.1

    data_reservoir = pd.concat(dfs, ignore_index=True)
    if 'ces_er_equivalent' in data_reservoir.columns:
        data_reservoir = data_reservoir.sort_values(by=['ces_er_equivalent'], ignore_index=True)

    if source_groups_dict is None:
        for sname, source in sources.items():
            source.set_data(data_reservoir)
            data_reservoir[f'{sname}_diff_rate'] = source.batched_differential_rate()
    else:
        source_groups_already_calculated = []

        for _, source_group_class in source_groups_dict.items():
            assert isinstance(source_group_class, fd.nest.SourceGroup), "Must be using source groups here!"
            if source_group_class not in source_groups_already_calculated:
                source_group_class.set_data(data_reservoir)

                if quanta_tensor_dirs_dict is None:
                    source_group_class.get_diff_rates()
                else:
                    source_group_class.get_diff_rates(read_in_dir=quanta_tensor_dirs_dict[source_group_class.base_source.__class__.__name__])

                source_groups_already_calculated.append(source_group_class)

                if save_comb:
                    data_reservoir[f'{source_group_class.base_source.__class__.__name__}_energies_diff_rates'] = \
                        source_group_class.base_source.data['energies_diff_rates']

        if not save_comb:
            for sname, source in sources.items():
                data_reservoir[f'{sname}_diff_rate'] = source_groups_dict[sname].get_diff_rate_source(source)

    if save_comb:
        cols = [c for c in data_reservoir.columns if (c[-19:] == 'energies_diff_rates')]
        cols.extend(['s1', 's2', 'x', 'y', 'z', 'r', 'theta', 'drift_time', 'event_time'])
    else:
        cols = [c for c in data_reservoir.columns if (c[-9:] == 'diff_rate')]
    cols.append('source')
    data_reservoir = data_reservoir[cols]

    if reservoir_output_name is not None:
        data_reservoir.to_pickle(reservoir_output_name)

    return data_reservoir


@export
class FrozenReservoirSource(fd.ColumnSource):
    """Source that looks up precomputed differential rates in a column source,
    with the added ability to simulate.

    Arguments:
        - source_type: base flamedisx source class. Pass if not passing input_mus
        - source_name: name given to the base source; must match the source name
            in the reservoir.
        - source_kwargs: any kwargs needed to instantiate the base source.
        - reservoir: dataframe of events with a column 'source' showing the
            {source_name} of the base source each event came from, and columns
            '{sorce_name}_diff_rate' with the differential rate of each event
            computed under all base sources that will have a FrozenReservoirSource used
            in the analysis.
        - input_mus: dictionary {sourcename: mu, ...} giving pre-computed mus for the sources.
        - rescale_diff_rates: if True, rate multipliers will correspond to expected counts after
            cuts.

    For other arguments, see flamedisx.source.Source
    """

    def __init__(self, source_type: fd.Source.__class__ = None, source_name: str = None,
                 source_kwargs: ty.Dict[str, ty.Union[int, float]] = None,
                 reservoir: pd.DataFrame = None,
                 input_mus=None,
                 rescale_diff_rates=False,
                 *args, **kwargs):
        if input_mus is None:
            assert source_type is not None, "Must pass a source type to FrozenReservoirSource, if not passing \
                input mus"
        assert source_name is not None, "Must pass a source name to FrozenReservoirSource"
        assert source_name in reservoir['source'].values, "The reservoir must contain events from this source type"

        if source_kwargs is None:
            source_kwargs = dict()

        self.source_name = source_name
        reservoir = reservoir.copy()
        reservoir = reservoir[reservoir['source'] == self.source_name]

        if rescale_diff_rates:
            assert input_mus is not None, "Must pass in input_mus if rescaling"
            for key, value in input_mus.items():
                reservoir[f'{key}_diff_rate'] = reservoir[f'{key}_diff_rate'] / value
            self.reservoir = reservoir
            self.mu = 1.

        else:
            self.reservoir = reservoir
            if input_mus is None:
                source = source_type(**source_kwargs)
                self.mu = source.estimate_mu()
            else:
                self.mu = input_mus[source_name]

        self.column = f'{source_name}_diff_rate'

        super().__init__(*args, **kwargs)

        self.ignore_acceptances = True

    def random_truth(self, n_events, fix_truth=None, **params):
        if fix_truth is not None:
            raise NotImplementedError("FrozenReservoirSource does not yet support fix_truth")
        if len(params):
            raise NotImplementedError("FrozenReservoirSource does not yet support alternative parameters in simulate")

        return self.reservoir.sample(n_events, replace=False)
