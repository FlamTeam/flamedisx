import typing as ty

import numpy as np
import pandas as pd

import flamedisx as fd
export, __all__ = fd.exporter()


class Particles():
    """
    """
    def __init__(self,
                 fit_params: ty.Tuple[str] = None,
                 bounds: ty.Dict[str, ty.Tuple[float]] = None,
                 guess_dict: ty.Dict[str, float] = None,
                 n_particles=50,
                 velocity_scaling=0.01):
        self.X = np.zeros((len(fit_params), n_particles))
        self.V = np.zeros((len(fit_params), n_particles))

        for i, param in enumerate(fit_params):
            self.X[i, :] = np.random.rand(n_particles) * (bounds[param][1] - bounds[param][0]) + bounds[param][0]
            self.V[i, :] = np.random.randn(n_particles) * 0.01 * guess_dict[param]f

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
        self.fit_params = fit_params
        self.bounds = bounds
        self.guess_dict = guess_dict

        self.n_particles = n_particles
        self.n_iterations = n_iterations

        self.particles = Particles(fit_params=fit_params, bounds=bounds, guess_dict=guess_dict,
                                   n_particles=n_particles)
