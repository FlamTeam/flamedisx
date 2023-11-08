import random
import string

from multihist import Histdd
import pandas as pd
from scipy.interpolate import interp2d

import numpy as np

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
            *args,
            **kwargs):
        assert(len(templates) == len(axis_names))
        self.interp_2d_list = []
        self.final_dimensions_list = []
        self.mh_list = []
        self.t_start = t_start
        self.t_stop = t_stop

        # Get templates, bin_edges, and axis_names
        for template, axis in zip(templates, axis_names):
            this_template, these_bin_edges = template.histogram, template.bin_edges
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

    def simulate(self, n_events, fix_truth=None, full_annotate=False,
                 keep_padding=False, **params):
        """Simulate n events.
        """
        if fix_truth:
            raise NotImplementedError("TemplateSource does not yet support fix_truth")
        assert isinstance(n_events, (int, float)), \
            f"n_events must be an int or float, not {type(n_events)}"

        sim_colums = []
        for final_dims, mh in zip(self.final_dimensions_list, self.mh_list):
            sim_colums.append(pd.DataFrame(dict(zip(
                final_dims, mh.get_random(n_events).T))))
        return pd.concat(sim_colums, axis=1)
