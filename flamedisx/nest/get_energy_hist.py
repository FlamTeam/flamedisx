from concurrent.futures import ProcessPoolExecutor

import numericalunits as nu
import numpy as np
import pandas as pd

import flamedisx as fd
import multihist as mh
import wimprates as wr


export, __all__ = fd.exporter()


@export
def get_energy_hist(
    wimp_mass=40.0,
    sigma=1e-45,
    min_E=1e-2,
    max_E=80.0,
    n_energy_bins=800,
    min_time="2019-09-01T08:28:00",
    max_time="2020-09-01T08:28:00",
    n_time_bins=25,
    modulation=True,
    migdal_model=None,
):

    wimprates_kwargs = dict(
        mw=wimp_mass,
        sigma_nucleon=sigma,
    )

    if migdal_model is not None:
        if migdal_model not in ["Ibe", "Cox", "Cox_dipole"]:
            raise ValueError(
                "Invalid Migdal model. Choose from 'Ibe', 'Cox', 'Cox_dipole'"
            )
        dipole = False
        if migdal_model == "Cox_dipole":
            migdal_model = "Cox"
            dipole = True

        wimprates_kwargs.update(
            dict(
                detection_mechanism="migdal",
                migdal_model=migdal_model,
                dipole=dipole,
            )
        )

    energy_bin_edges = np.linspace(min_E, max_E, n_energy_bins + 1)
    energy_bin_width = (energy_bin_edges[1] - energy_bin_edges[0]) * nu.keV
    energy_values = (energy_bin_edges[1:] + energy_bin_edges[:-1]) / 2

    time_bin_edges = (
        pd.date_range(min_time, max_time, periods=n_time_bins + 1).to_julian_date()
        - 2451545.0  # Convert to J2000
    )
    times = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2

    scale = (
        energy_bin_width / nu.keV
    )  # Convert from [per keV] to [per energy_bin_width]

    rates_list = []
    if modulation:
        with ProcessPoolExecutor() as executor:
            for time in times:
                rates_list.append(
                    executor.submit(
                        wr.rate_wimp_std, energy_values, t=time, **wimprates_kwargs
                    )
                )

            rates_list = [future.result() for future in rates_list]
    else:
        rates = wr.rate_wimp_std(energy_values, **wimprates_kwargs)
        for _ in times:
            rates_list.append(rates)

    RATES = np.array(rates_list) * scale

    hist = mh.Histdd.from_histogram(
        RATES, [time_bin_edges, energy_bin_edges], axis_names=("time", "energy")
    )

    return hist
