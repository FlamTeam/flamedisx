import typing as ty

import numpy as np
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


class Particles():
    """
    """
    def __init__(self,
                 sources: ty.Dict[str, fd.Source] = None,
                 fit_params: ty.Tuple[str] = None,
                 bounds: ty.Dict[str, ty.Tuple[float]] = None,
                 guess_dict: ty.Dict[str, float] = None,
                 n_particles=50,
                 velocity_scaling=0.01):

        self.sources = sources

        self.fit_params = fit_params
        self.bounds = bounds
        self.guess_dict = guess_dict

        self.n_particles = n_particles

        self.X = np.zeros((len(fit_params), n_particles))
        self.V = np.zeros((len(fit_params), n_particles))

        self.initialise_particles(velocity_scaling=velocity_scaling)

    def initialise_particles(self, velocity_scaling):

        for i, param in enumerate(self.fit_params):
            self.X[i, :] = np.random.rand(np.shape(self.X)[1]) * \
                (self.bounds[param][1] - self.bounds[param][0]) + self.bounds[param][0]
            self.V[i, :] = np.random.randn(np.shape(self.X)[1]) * velocity_scaling * self.guess_dict[param]

        self.pbest = self.X
        self.pbest_obj = np.zeros(len(self.X[0, :]))

        for i in range(len(self.pbest_obj)):
            eval_dict = dict()
            for j, param in enumerate(self.fit_params):
                eval_dict[param] = self.X[j, i]

            self.pbest_obj[i] = -2. * self.likelihood(**eval_dict)

        self.gbest = self.pbest[:, self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

    def update_particles(self, c1, c2, w):

        r1, r2 = np.random.rand(2)
        self.V = w * self.V + c1 * r1 * (self.pbest -self. X) + \
            c2 * r2 * (self.gbest.reshape(-1, 1) - self.X)
        self.X = self.X + self.V

        obj = np.zeros_like(self.pbest_obj)
        for i in range(len(obj)):
            eval_dict = dict()
            for j, param in enumerate(self.fit_params):
                eval_dict[param] = self.X[j, i]

            obj[i] = -2. * self.likelihood(**eval_dict)

        self.pbest[:, (self.pbest_obj >= obj)] = self.X[:, (self.pbest_obj >= obj)]
        self.pbest_obj = np.array([self.pbest_obj, obj]).min(axis=0)

        self.gbest = self.pbest[:, self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

    def likelihood(self, **params):
        params_filter = dict()
        for key, value in params.items():
            if key[-16:] == '_rate_multiplier':
                continue
            params_filter[key] = value

        dr_sum = 0.
        mu_sum = 0.
        for sname, source in self.sources.items():
            dr_sum += source.batched_differential_rate(**params_filter, progress=False) * params[f'{sname}_rate_multiplier']
            mu_sum += source.estimate_mu(**params_filter, fast=True) * params[f'{sname}_rate_multiplier']

        return (-mu_sum + np.sum(np.log(dr_sum)))


@export
class PSOOptimiser():
    """
    """
    def __init__(self,
                 sources: ty.Dict[str, fd.Source] = None,
                 fit_params: ty.Tuple[str] = None,
                 bounds: ty.Dict[str, ty.Tuple[float]] = None,
                 guess_dict: ty.Dict[str, float] = None,
                 n_particles=50,
                 n_iterations=100,
                 c1=0.1, c2=0.1, w=0.8):

        self.n_iterations = n_iterations

        self.particles = Particles(sources=sources, fit_params=fit_params,
                                   bounds=bounds, guess_dict=guess_dict,
                                   n_particles=n_particles)

        self.c1 = c1
        self.c2 = c2
        self.w = w

    def run_routine(self):
        for i in range(self.n_iterations):
            self.particles.update_particles(self.c1, self.c2, self.w)

    def bestfit(self):
        bf = dict()
        for i, param in enumerate(self.particles.fit_params):
            bf[param] = self.particles.gbest[i]

        return bf
