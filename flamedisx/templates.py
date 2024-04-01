import random
import string

from multihist import Histdd
from multihist import Hist1d
import pandas as pd
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d

import numpy as np
from scipy import stats

import flamedisx as fd

export, __all__ = fd.exporter()


@export
class TemplateSource(fd.ColumnSource):
    """Source that looks up precomputed differential rates in a template
    (probably a histogram from a simulation).

    Arguments:
        - template: numpy array, multhist.Histdd, or (hist/boost_histogram).
            containing the differential rate.
        - bin_edges: None, or a list of numpy arrays with bin edges.
            If None, get this info from template.
        - axis_names: None, or a sequence of axis names.
            If None, get this info from template.
        - events_per_bin: set to True if template specifies expected events per
            bin, rather than differential rate.

    For other arguments, see flamedisx.source.Source
    """

    def __init__(
            self,
            template,
            interp_2d=False,
            bin_edges=None,
            axis_names=None,
            events_per_bin=False,
            *args,
            **kwargs):
        # Get template, bin_edges, and axis_names
        if bin_edges is None:
            # Hopefully we got some kind of histogram container
            if isinstance(template, tuple) and len(template) == 2:
                # (hist, bin_edges) tuple, e.g. from np.histdd
                template, bin_edges = template
            elif hasattr(template, "to_numpy"):
                # boost_histogram / hist
                if not axis_names:
                    axis_names = [ax.name for ax in template.axes]
                template, bin_edges = template.to_numpy()
            elif hasattr(template, "bin_edges"):
                # multihist
                if not axis_names:
                    axis_names = template.axis_names
                template, bin_edges = template.histogram, template.bin_edges
            else:
                raise ValueError("Need histogram, bin_edges, and axis_names")

        if not axis_names or len(axis_names) != len(template.shape):
            raise ValueError("Axis names missing or mismatched")
        self.final_dimensions = axis_names

        if interp_2d:
            assert len(self.final_dimensions) == 2, "Interpolation only supported for 2D histogram!"
            centers_dim_1 = 0.5 * (bin_edges[0][1:] + bin_edges[0][:-1])
            centers_dim_2 = 0.5 * (bin_edges[1][1:] + bin_edges[1][:-1])
            self.interp_2d = interp2d(centers_dim_1, centers_dim_2, np.transpose(template))
        else:
            self.interp_2d = None

        # Build a diff rate and events/bin multihist from the template
        _mh = Histdd.from_histogram(template, bin_edges=bin_edges)
        if events_per_bin:
            self._mh_events_per_bin = _mh
            self._mh_diff_rate = _mh / _mh.bin_volumes()
        else:
            self._mh_events_per_bin = _mh * _mh.bin_volumes()
            self._mh_diff_rate = _mh

        self.mu = fd.np_to_tf(self._mh_events_per_bin.n)

        # Generate a random column name to use to store the diff rates
        # of observed events
        self.column = (
            'template_diff_rate_'
            + ''.join(random.choices(string.ascii_lowercase, k=8)))

        super().__init__(*args, **kwargs)

    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        if self.interp_2d is not None:
            self.data[self.column] = np.array([self.interp_2d(r[self.data.columns.get_loc(self.final_dimensions[0])],
                                                              r[self.data.columns.get_loc(self.final_dimensions[1])])[0]
                                               for r in self.data.itertuples(index=False)])
        else:
            self.data[self.column] = self._mh_diff_rate.lookup(
                *[self.data[x] for x in self.final_dimensions])

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        """Simulate n events.
        """
        if fix_truth:
            raise NotImplementedError("TemplateSource does not yet support fix_truth")
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        # TODO: all other arguments are ignored, they make no sense
        # for this source. Should we warn about this? Remove them from def?

        return pd.DataFrame(dict(zip(
            self.final_dimensions,
            self._mh_events_per_bin.get_random(n_events).T)))


@export
class TemplateProductSource(fd.ColumnSource):
    """
    """

    def __init__(
            self,
            templates=None,
            axis_names=None,
            t_start=None,
            t_stop=None,
            decay_constant_ns=None,
            *args,
            **kwargs):
        assert(len(templates) == len(axis_names))
        self.interp_2d_list = []
        self.final_dimensions_list = []
        self.mh_list = []

        self.t_start = t_start
        self.t_stop = t_stop
        self.decay_constant_ns = decay_constant_ns

        # Get templates, bin_edges, and axis_names
        for template, axis in zip(templates, axis_names):
            this_template, these_bin_edges = template.histogram, template.bin_edges
            assert len(np.shape(this_template)) == len(axis)
            if len(np.shape(this_template)) == 2:
                centers_dim_1 = 0.5 * (these_bin_edges[0][1:] + these_bin_edges[0][:-1])
                centers_dim_2 = 0.5 * (these_bin_edges[1][1:] + these_bin_edges[1][:-1])
                self.interp_2d_list.append(interp2d(centers_dim_1, centers_dim_2, np.transpose(this_template)))
            else:
                self.interp_2d_list.append(None)
            self.final_dimensions_list.append(axis)
            mh = Histdd.from_histogram(this_template, bin_edges=these_bin_edges)
            self.mh_list.append(mh)

        self.final_dimensions = sum(self.final_dimensions_list, ())

        self.mu = fd.np_to_tf(1.)

        # Generate a random column name to use to store the diff rates
        # of observed events
        self.column = (
            'template_diff_rate_'
            + ''.join(random.choices(string.ascii_lowercase, k=8)))

        super().__init__(*args, **kwargs)

    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        self.data[self.column] = np.ones_like(len(self.data))
        for final_dims, interp_2d, mh in zip(self.final_dimensions_list, self.interp_2d_list, self.mh_list):
            if interp_2d is not None:
                self.data[self.column] *= np.array([interp_2d(r[self.data.columns.get_loc(final_dims[0])],
                                                            r[self.data.columns.get_loc(final_dims[1])])[0]
                                                    for r in self.data.itertuples(index=False)])
            else:
                self.data[self.column] *= mh.lookup(*[self.data[x] for x in final_dims])

        if (self.decay_constant_ns is not None) and ('event_time' in self.final_dimensions):
            pdf = np.exp(-(self.data['event_time'].values - self.t_start.value) / self.decay_constant_ns)
            normalisation = 1. / (self.decay_constant_ns * (1. - np.exp(-(self.t_stop.value - self.t_start.value) / self.decay_constant_ns)))
            uniform_pdf = 1. / (self.t_stop.value - self.t_start.value)
            self.data[self.column] *= normalisation * pdf / uniform_pdf

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        """Simulate n events.
        """
        if fix_truth:
            raise NotImplementedError("TemplateSource does not yet support fix_truth")
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        df = pd.DataFrame()

        if 'event_time' in self.final_dimensions:
            if self.decay_constant_ns is None:
                df['event_time'] = np.random.uniform(
                    self.t_start.value,
                    self.t_stop.value,
                    size=n_events)
            else:
                b = (self.t_stop.value - self.t_start.value) / self.decay_constant_ns
                df['event_time'] = stats.truncexpon.rvs(b,
                    loc=self.t_start.value, scale=self.decay_constant_ns,
                    size=n_events)

        sim_colums = []
        for final_dims, mh in zip(self.final_dimensions_list, self.mh_list):
            if 'event_time' in final_dims:
                events = []
                for t in df['event_time'].values:
                    test = mh.slicesum(t, axis=final_dims.index('event_time'))
                    new_final_dims = tuple([x for x in final_dims if x != 'event_time'])
                    events.append(pd.DataFrame(dict(zip(
                        new_final_dims, test.get_random(1).T))))
                sim_colums.append(pd.concat(events, ignore_index=True,))
                sim_colums.append(df['event_time'])
                continue
            sim_colums.append(pd.DataFrame(dict(zip(
                final_dims, mh.get_random(n_events).T))))
        df = pd.concat(sim_colums, axis=1)

        return df

    
@export
class CorrelatedTemplateProductSource(fd.ColumnSource):
    """
    """

    def __init__(
            self,
            templates=None,
            axis_names=None,
            *args,
            **kwargs):
        assert(len(templates) == len(axis_names))
        ## Make sure all the templates are 2D
        for t in templates:
            assert len(np.shape(t.histogram)) == 2, 'Correlated products currently only supported for 2D templates.'
            
        ## Make sure all the templates add new information; e.g. if template[0] has axes (a,b) and 
        ## template[1] has axes (b,c), template[2] better not have axes (a,c). 
        self.final_dimensions_list = []
        self.final_dimensions = []
        for axs in axis_names:
            ax1, ax2 = axs
            if ax1 in self.final_dimensions and ax2 in self.final_dimensions:
                raise RuntimeError(f'No unique information added to model from template with axes ({ax1}, {ax2})')
            self.final_dimensions_list.append(axs)
            if ax1 not in self.final_dimensions and ax2 not in self.final_dimensions:
                self.final_dimensions.append(axs)
            elif ax1 not in self.final_dimensions:
                self.final_dimensions.append((ax1,))
            else:
                self.final_dimensions.append((ax2,))
        self.final_dimensions = sum(self.final_dimensions,())
        self.interp_2d_list = []
        self.interp_1d_list = []
        self.sample_1d_list = []
        
        ## Create a list of lists containing the names of shared axes between the set of input templates
        self.shared_dimensions_matrix = []
        for ans in axis_names:
            shared_axes = []
            for ans2 in axis_names:
                if ans == ans2:
                    ## If comparing the axes of a template with itself, label it as 1
                    shared_axes.append(1)
                else:
                    ## Else get all the axis names in common between two templates
                    ax = [a for a in ans if a in ans2]
                    try:
                        ## By earlier assertion, we assume there will be at most 1 shared axis between two templates
                        shared_axes.append(ax[0])
                    except:
                        ## If there are no shared axes, we'll get an index error. In that case label as 0
                        shared_axes.append(0)
                    
            self.shared_dimensions_matrix.append(shared_axes)
        
        self.mh_list = []

        # Get templates, bin_edges, and axis_names
        for n, template, axis in enumerate(zip(templates, axis_names)):
            this_template, these_bin_edges = template.histogram, template.bin_edges
            assert len(np.shape(this_template)) == len(axis)
            mh = Histdd.from_histogram(this_template, bin_edges=these_bin_edges, axis_names=axis)
            self.mh_list.append(mh)
            
            ## The first template will always be kept as a 2D template. For subsequent templates
            ## in the list, we'll check if they share an axis with any templates preceeding it.
            shared_axes = self.shared_dimensions_matrix[n]
            no_shared_axes = True
            for ax in shared_axes:
                if ax == 1:
                    ## Breaking here ensures we only look at the previous templates in the list relative to this one
                    break
                if ax == 0:
                    ## Continue if there's no shared axis between this template and a given previous one
                    continue
                ## If we get here, there must be a shared axis between this template and a previous one. We will
                ## generate a list of 1D interpolators, which we can reference later as a PDF for a conditional 
                ## probability. E.g. we get (S1, S2) from P(S1, S2) and then we get R from P(R | S2).
                no_shared_axes = False
                shared_axis_name = ax
                break
            
            if no_shared_axes:
                centers_dim_1 = 0.5 * (these_bin_edges[0][1:] + these_bin_edges[0][:-1])
                centers_dim_2 = 0.5 * (these_bin_edges[1][1:] + these_bin_edges[1][:-1])
                self.interp_2d_list.append(interp2d(centers_dim_1, centers_dim_2, np.transpose(this_template)))
                self.interp_1d_list.append(None)
                self.sample_1d_list.append(None)
            else:
                interpolaters = dict()
                samplers = dict()
                if shared_axis_name == axis[0]
                    other_axis_name = axis[1]
                else:
                    other_axis_name = axis[0]
                axis_index = mh.get_axis_number(shared_axis_name)
                other_axis_index = mh.get_axis_number(other_axis_name)
                
                centres = []
                centres.append(0.5 * (these_bin_edges[0][1:] + these_bin_edges[0][:-1]))
                centres.append(0.5 * (these_bin_edges[1][1:] + these_bin_edges[1][:-1]))
                
                for i, c in centres[axis_index]:
                    temp_slice = mh.slice(start=c,stop=c,axis=axis_index)
                    temp_slice_proj = temp_slice.projection(other_axis_name)
                    temp_slice_proj = temp_slice_proj/temp_slice_proj.n
                    temp_slice_proj = temp_slice_proj/temp_slice_proj.bin_volumes()
                    interpolaters[c] = interp1d(centres[other_axis_index], temp_slice_proj.histogram.T[0])
                    samplers[c] = temp_slice_proj
                self.interp_2d_list.append(None)
                self.interp_1d_list.append(interpolaters)
                self.sample_1d_list.append(samplers)
            #self.final_dimensions_list.append(axis)
        #self.final_dimensions = sum(self.final_dimensions_list, ())

        self.mu = fd.np_to_tf(1.)

        # Generate a random column name to use to store the diff rates
        # of observed events
        self.column = (
            'template_diff_rate_'
            + ''.join(random.choices(string.ascii_lowercase, k=8)))

        super().__init__(*args, **kwargs)
    
    def sample_conditional_pdf(self, sample_dict, condition):
        arr = np.array(list(interp1d_dict.keys()))
        if hasattr(condition,'__iter__'):
            indexes = [np.argwhere(c >= arr)[0][0] for c in condition]
            keys = [arr[i] for i in indexes]
            return [sample_dict[k].get_random(1)[0] k in keys]
        else:
            index = np.argwhere(condition[0] >= arr)[0][0]
            key = arr[index]
            return sample_dict[key].get_random(1)[0]
    
    def conditional_probability(self, interp1d_dict, vals):
        arr = np.array(list(interp1d_dict.keys()))
        if hasattr(vals[0],'__iter__'):
            indexes = [np.argwhere(v >= arr)[0][0] for v in vals[0]]
            keys = [arr[i] for i in indexes]
            return [interp1d_dict[k](v) k, v in zip(keys, vals[1])]
        else:
            index = np.argwhere(vals[0] >= arr)[0][0]
            key = arr[index]
            return interp1d_dict[key](vals[1])
            
        
    def _annotate(self):
        """Add columns needed in inference to self.data
        """
        self.data[self.column] = np.ones_like(len(self.data))
        for final_dims, interp_2d, interp_1d, mh in zip(self.final_dimensions_list, self.interp_2d_list, self.interp_1d_list, self.mh_list):
            if interp_2d is not None:
                self.data[self.column] *= np.array([interp_2d(r[self.data.columns.get_loc(final_dims[0])],
                                                            r[self.data.columns.get_loc(final_dims[1])])[0]
                                                    for r in self.data.itertuples(index=False)])
            elif interp_1d is not None:
                col0 = self.data[final_dims[0]].to_numpy()
                col1 = self.data[final_dims[1]].to_numpy()
                vals = np.array((col0,col1))
                self.data[self.column] *= np.array(self.conditional_probability(interp_1d, vals))
            else:
                self.data[self.column] *= mh.lookup(*[self.data[x] for x in final_dims])

        if (self.decay_constant_ns is not None) and ('event_time' in self.final_dimensions):
            pdf = np.exp(-(self.data['event_time'].values - self.t_start.value) / self.decay_constant_ns)
            normalisation = 1. / (self.decay_constant_ns * (1. - np.exp(-(self.t_stop.value - self.t_start.value) / self.decay_constant_ns)))
            uniform_pdf = 1. / (self.t_stop.value - self.t_start.value)
            self.data[self.column] *= normalisation * pdf / uniform_pdf

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        """Simulate n events.
        """
        if fix_truth:
            raise NotImplementedError("TemplateSource does not yet support fix_truth")
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        sim_events_dict = dict()
        for i, final_dims, mh, samp1d in enumerate(zip(self.final_dimensions_list, self.mh_list, self.sample_1d_list)):
            if samp1d is None:
                df_data = mh.get_random(n_events).T
                sim_events_dict[final_dims[0]] = df_data[0]
                sim_events_dict[final_dims[1]] = df_data[1]
            else:
                shared_dims = self.shared_dimensions_matrix[i]
                for d in shared_dims:
                    if d == 1:
                        break
                    if final_dims[0] == d:
                        conditional_dim = final_dims[0]
                        simulated_dim = final_dims[1]
                    if final_dims[1] == d:
                        conditional_dim = final_dims[1]
                        simulated_dim = final_dims[0]
                try:
                    conditions = np.array(sim_events_dict[conditional_dim].values())
                except:
                    raise RuntimeError(f'Accessing {conditional_dim} data when it has not been simulated yet.')
                sim_events_dict[simulated_dim] = np.array(self.sample_conditional_pdf(samp1d, conditions))

        df = pd.DataFrame(sim_events_dict)

        return df
