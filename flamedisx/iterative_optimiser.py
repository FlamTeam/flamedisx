import typing as ty

import numpy as np
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


class IterativeColumnSource(fd.ColumnSource):
    def __init__(self, *args, column_name=None, mu=None, **kwargs):
        self.column = column_name
        self.mu = mu
        super().__init__(*args, **kwargs)


@export
class IterativeOptimiser():
    """
    """
    def __init__(self,
                 regular_sources: ty.Dict[str, fd.Source.__class__] = None,
                 column_sources: ty.Dict[str, fd.Source.__class__] = None,
                 data: pd.DataFrame = None,
                 bounds: ty.Dict[str, ty.Tuple[float]] = None,
                 guess_dict: ty.Dict[str, float] = None,
                 log_constraint = None,
                 batch_size=5000,  
                 n_iterations=5):

        self.batch_size = batch_size

        self.guess_dict = guess_dict
        shape_params = dict()
        for key, value in self.guess_dict.items():
            if key[-16:] == '_rate_multiplier':
                continue
            shape_params[key] = value

        self.column_sources = column_sources
        for sname, source in self.column_sources.items():
            s = source(batch_size=self.batch_size)
            s.set_data(data)
            data[f'{sname}_diff_rate'] = s.batched_differential_rate(**shape_params, progress=False)

        self.data = data

        column_names = dict()
        self.mus = dict()
        for sname, source in self.column_sources.items():
            column_names[sname] = f'{sname}_diff_rate'
            self.mus[sname] = source().estimate_mu(**shape_params)

        est = dict()
        for sname, source in regular_sources.items():
            est[sname] = fd.SimulateEachCallMu(source=source())
        for sname, source in self.column_sources.items():
            est[sname] = fd.ConstantMu(source=IterativeColumnSource(mu=self.mus[sname]))

        sources = dict()
        arguments = dict()
        for sname, source in regular_sources.items():
            sources[sname] = source
            arguments[sname] = dict()
        for sname in self.column_sources.keys():
            sources[sname] = IterativeColumnSource
            arguments[sname] = {'column_name': column_names[sname]}

        self.likelihood = fd.LogLikelihood(sources=sources,
                                           arguments=arguments,
                                           free_rates={sname for sname in sources.keys()},
                                           batch_size=self.batch_size,
                                           log_constraint=log_constraint,
                                           mu_estimators = est,
                                           **bounds
                                          )

        self.n_iterations = n_iterations

    def bestfit(self,
                fix_dict: ty.Dict[str, float] = None):
        self.likelihood.set_data(self.data)

        if fix_dict is None:
            fix_dict = dict()

        guess_dict_cond = self.guess_dict.copy()
        for key in fix_dict.keys():
            if key in guess_dict_cond:
                guess_dict_cond.pop(key)

        bf = self.likelihood.bestfit(guess=guess_dict_cond,
                                     fix=fix_dict)
        print(bf)

        for i in range(self.n_iterations - 1):
            shape_params = dict()
            for key, value in bf.items():
                if key[-16:] == '_rate_multiplier':
                    continue
                shape_params[key] = value

            for sname, source in self.column_sources.items():
                s = source(batch_size=self.batch_size)
                s.set_data(self.data)
                self.data[f'{sname}_diff_rate'] = s.batched_differential_rate(**shape_params, progress=False)
                self.mus[sname] = s.estimate_mu(**shape_params)

            for sname, source in self.column_sources.items():
                self.likelihood.mu_estimators[sname] = fd.ConstantMu(source=IterativeColumnSource(mu=self.mus[sname]))

            self.likelihood.set_data(self.data)

            guess_dict_cond = bf.copy()
            for key in fix_dict.keys():
                if key in guess_dict_cond:
                    guess_dict_cond.pop(key)

            bf = self.likelihood.bestfit(guess=guess_dict_cond,
                                         fix=fix_dict)
            print(bf)

        return bf
