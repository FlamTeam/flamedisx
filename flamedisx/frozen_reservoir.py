import typing as ty
import pandas as pd
import pickle as pkl

import flamedisx as fd
export, __all__ = fd.exporter()


def make_event_reservoir(ntoys: int = None,
                         input_prefix='',
                         input_label=None,
                         reservoir_output_name=None,
                         max_rm_dict=None,
                         source_groups_dict=None,
                         quanta_tensor_dirs_dict=None,
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

    if max_rm_dict is None:
        max_rm_dict = dict()

    if input_label is not None:
        data_reservoir = pkl.load(open(f'{input_prefix}partial_toy_reservoir{input_label}.pkl', 'rb'))

        if source_groups_dict is None:
            for sname, source in sources.items():
                source.set_data(data_reservoir,
                                input_column_index=f'{input_prefix}{sname}_column_index{input_label}.pkl',
                                input_data_tensor=f'{input_prefix}{sname}_data_tensor{input_label}')
                data_reservoir[f'{sname}_diff_rate'] = source.batched_differential_rate()
        else:
            for _, source_group_class in source_groups_dict.items():
                assert isinstance(source_group_class, fd.nest.SourceGroup), "Must be using source groups here!"
                source_group_class.set_data(data_reservoir,
                                            input_column_index=f'{input_prefix}{source_group_class.base_source.__class__.__name__}_column_index{input_label}.pkl',
                                            input_data_tensor=f'{input_prefix}{source_group_class.base_source.__class__.__name__}_data_tensor{input_label}')
                if quanta_tensor_dirs_dict is None:
                    source_group_class.get_diff_rates()
                else:
                    source_group_class.get_diff_rates(read_in_dir=quanta_tensor_dirs_dict[source_group_class.base_source.__class__.__name__])
            for sname, source in sources.items():
                data_reservoir[f'{sname}_diff_rate'] = source_groups_dict[sname].get_diff_rate_source(source,
                                                                                                      input_column_index=f'{input_prefix}{sname}_column_index{input_label}.pkl',
                                                                                                      input_data_tensor=f'{input_prefix}{sname}_data_tensor{input_label}')

        if reservoir_output_name is not None:
            data_reservoir.to_pickle(reservoir_output_name)

        return data_reservoir

    dfs = []
    for sname, source in sources.items():
        if sname in max_rm_dict.keys():
            max_rm = max_rm_dict[sname]
        else:
            max_rm = 1.
        n_simulate = int(max_rm * ntoys * source.mu_before_efficiencies())

        sdata = source.simulate(n_simulate)
        sdata['source'] = sname
        dfs.append(sdata)

    data_reservoir = pd.concat(dfs, ignore_index=True)
    data_reservoir = data_reservoir.sort_values(by=['ces_er_equivalent'], ignore_index=True)

    if source_groups_dict is None:
        for sname, source in sources.items():
            source.set_data(data_reservoir)
            data_reservoir[f'{sname}_diff_rate'] = source.batched_differential_rate()
    else:
        for _, source_group_class in source_groups_dict.items():
            assert isinstance(source_group_class, fd.nest.SourceGroup), "Must be using source groups here!"
            source_group_class.set_data(data_reservoir)
            if quanta_tensor_dirs_dict is None:
                source_group_class.get_diff_rates()
            else:
                source_group_class.get_diff_rates(read_in_dir=quanta_tensor_dirs_dict[source_group_class.base_source.__class__.__name__])
        for sname, source in sources.items():
            data_reservoir[f'{sname}_diff_rate'] = source_groups_dict[sname].get_diff_rate_source(source)

    if reservoir_output_name is not None:
        data_reservoir.to_pickle(reservoir_output_name)

    return data_reservoir


def make_event_reservoir_no_compute(ntoys: int = None,
                                    output_prefix='',
                                    output_label='',
                                    max_rm_dict=None,
                                    source_groups_dict=None,
                                    **sources):
    """Generate data tensor and event reservoir without differetial rates, to be used to
    generate the full reservoir for a FrozenReservoirSource. This could be useful for
    pre-computation over a large numbers of CPUs, if the data tensor is particulalry complex
    or annotation takes a long time. It can then be quickly read in on a GPU, to allow one
    to maximise the utility of the  GPU computation time they are afforded.

    Arguments:
        - ntoys: number of toy MCs the reservoir will be used to generate (optional).
        - sources: pass in source instances to be used to build the reservoir, like
            'source1'=source1(args, kwargs), 'source2'=source2(args, kwargs), ...
    """
    default_ntoys = 1000

    assert len(sources) != 0, "Must pass at least one source instance to event_reservoir_data_tensor()"

    if ntoys is None:
        ntoys = default_ntoys

    if max_rm_dict is None:
        max_rm_dict = dict()

    dfs = []
    for sname, source in sources.items():
        if sname in max_rm_dict.keys():
            max_rm = max_rm_dict[sname]
        else:
            max_rm = 1.
        n_simulate = int(max_rm * ntoys * source.mu_before_efficiencies())

        sdata = source.simulate(n_simulate)
        sdata['source'] = sname
        dfs.append(sdata)

    data_reservoir = pd.concat(dfs, ignore_index=True)
    data_reservoir = data_reservoir.sort_values(by=['ces_er_equivalent'], ignore_index=True)

    data_reservoir.to_pickle(f'{output_prefix}partial_toy_reservoir{output_label}.pkl')

    if source_groups_dict is None:
        for sname, source in sources.items():
            source.set_data(data_reservoir, output_data_tensor=f'{output_prefix}{sname}_data_tensor{output_label}')
            pkl.dump(source.column_index, open(f'{output_prefix}{sname}_column_index{output_label}.pkl', 'wb'))
    else:
        for _, source_group_class in source_groups_dict.items():
            assert isinstance(source_group_class, fd.nest.SourceGroup), "Must be using source groups here!"
            source_group_class.set_data(data_reservoir,
                                        output_data_tensor=f'{output_prefix}{source_group_class.base_source.__class__.__name__}_data_tensor{output_label}')
            pkl.dump(source_group_class.base_source.column_index,
                     open(f'{output_prefix}{source_group_class.base_source.__class__.__name__}_column_index{output_label}.pkl', 'wb'))
            for sname, source in sources.items():
                if isinstance(source_groups_dict[sname].base_source, source_group_class.base_source.__class__):
                    source.set_data(source_group_class.base_source.data, data_is_annotated=True,
                                    output_data_tensor=f'{output_prefix}{sname}_data_tensor{output_label}')
                    pkl.dump(source.column_index,
                             open(f'{output_prefix}{sname}_column_index{output_label}.pkl', 'wb'))


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
                 input_mu=None,
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
        if input_mu is None:
            self.mu = source.estimate_mu()
        else:
            self.mu = input_mu

        super().__init__(*args, **kwargs)

    def random_truth(self, n_events, fix_truth=None, **params):
        if fix_truth is not None:
            raise NotImplementedError("FrozenReservoirSource does not yet support fix_truth")
        if len(params):
            raise NotImplementedError("FrozenReservoirSource does not yet support alternative parameters in simulate")

        return self.reservoir[self.reservoir['source'] == self.source_name].sample(n_events, replace=True)
