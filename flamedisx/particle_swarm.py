import typing as ty

import numpy as np
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


class Particles():
    """
    """
    def __init__(self,
                 likelihood: fd.LogLikelihood = None,
                 fit_params: ty.Tuple[str] = None,
                 bounds: ty.Dict[str, ty.Tuple[float]] = None,
                 guess_dict: ty.Dict[str, float] = None,
                 n_particles=50,
                 velocity_scaling=0.01):

        self.fit_params = fit_params
        self.bounds = bounds
        self.guess_dict = guess_dict

        self.n_particles = n_particles

        self.X = np.zeros((len(fit_params), n_particles))
        self.V = np.zeros((len(fit_params), n_particles))

        self.initialise_particles(likelihood=likelihood, velocity_scaling=velocity_scaling)

    def initialise_particles(self, likelihood, velocity_scaling):

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
                self.pbest_obj[i] = -2. * likelihood(**eval_dict)

        self.gbest = self.pbest[:, self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

@export
class PSOOptimiser():
    """
    """
    def __init__(self,
                 likelihood: fd.LogLikelihood = None,
                 fit_params: ty.Tuple[str] = None,
                 bounds: ty.Dict[str, ty.Tuple[float]] = None,
                 guess_dict: ty.Dict[str, float] = None,
                 n_particles=50,
                 n_iterations=50):

        self.likelihood = likelihood

        self.n_iterations = n_iterations

        self.particles = Particles(likelihood=self.likelihood, fit_params=fit_params,
                                   bounds=bounds, guess_dict=guess_dict,
                                   n_particles=n_particles)
