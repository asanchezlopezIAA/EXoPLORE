"""Physical atmosphere diagnostics for Sunbather helium simulations."""
from pathlib import Path
import json

import numpy as np


def write_atmosphere_diagnostics(directory: Path, settings, planet, profile: dict) -> None:
    """Save original Parker arrays and Cloudy profiles with labelled figures.

    Files are created exclusively. T0 is a Parker input; Tmax is the maximum
    over the entire saved Cloudy domain, not a temperature consistency test.
    """
    import matplotlib.pyplot as plt

    directory = Path(directory)
    # Metal-bearing models retain intermediate mean-mass profiles underneath
    # mu_iterations. Only the final, top-level profile enters the atmosphere.
    paths = list((directory / 'parker_profiles' / planet.name / 'exoplore').glob('pprof_*.txt'))
    if len(paths) != 1:
        raise ValueError('Expected one original Parker profile for this atmosphere')
    parker = np.loadtxt(paths[0])
    radius_cm = planet.planet_radius_rjup * 7.1492e9
    radial = np.asarray(profile['radius_rp'])
    temperature = np.asarray(profile['temperature_K'])
    peak = int(np.argmax(temperature))
    products = ['parker_profiles.npz', 'sunbather_profiles.npz',
                'atmosphere_summary.json', 'helium_atmosphere.png', 'helium_atmosphere.pdf']
    if any((directory / name).exists() for name in products):
        raise FileExistsError('Atmosphere diagnostics already exist; preserve the saved products')
    with (directory / products[0]).open('xb') as stream:
        np.savez_compressed(stream, radius_rp=parker[:, 0]/radius_cm,
                            density_g_cm3=parker[:, 1], velocity_kms=parker[:, 2]/1e5,
                            mean_molecular_weight=parker[:, 3])
    with (directory / products[1]).open('xb') as stream:
        np.savez_compressed(stream, **{key: profile[key] for key in
            ('radius_rp', 'temperature_K', 'density_g_cm3', 'velocity_m_s', 'helium_triplet_m3')})
    summary = dict(T0_K=settings.temperature_K, Tmax_K=float(temperature[peak]),
        Tmax_radius_rp=float(radial[peak]), mass_loss_rate_g_s=settings.mass_loss_rate_g_s,
        hydrogen_number_fraction=settings.hydrogen_number_fraction,
        stellar_tidal_gravity=settings.sunbather['stellar_tidal_gravity'],
        parker_source=str(paths[0]), cloudy_source=str(profile['cloudy_simulation']),
        interpretation='Prescribed Parker density and velocity; energy-balanced Cloudy temperature and helium populations. No new hydrodynamic solution or observational fit.')
    with (directory / products[2]).open('x') as stream:
        json.dump(summary, stream, indent=2)
    with plt.rc_context({'font.size': 14, 'axes.labelsize': 15, 'legend.fontsize': 12}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
        axes[0, 0].semilogy(parker[:, 0]/radius_cm, parker[:, 1], color='black', label='Parker wind')
        axes[0, 0].semilogy(radial, profile['density_g_cm3'], '--', label='Sunbather profile')
        axes[0, 0].set_ylabel('Mass density (g cm⁻³)')
        axes[0, 0].legend()
        axes[0, 1].plot(parker[:, 0]/radius_cm, parker[:, 2]/1e5, color='black')
        axes[0, 1].set_ylabel('Parker radial velocity (km s⁻¹)')
        axes[1, 0].plot(radial, temperature, label='Cloudy T(r)')
        axes[1, 0].axhline(settings.temperature_K, color='black', ls='--', label=f'Parker T₀ = {settings.temperature_K:g} K')
        axes[1, 0].plot(radial[peak], temperature[peak], 'o', label=f'Tmax = {temperature[peak]:.0f} K')
        axes[1, 0].set_ylabel('Temperature (K)')
        axes[1, 0].legend()
        axes[1, 1].semilogy(radial, np.asarray(profile['helium_triplet_m3'])/1e6)
        axes[1, 1].set_ylabel('Metastable He I density (cm⁻³)')
        for axis in axes[-1]:
            axis.set_xlabel('Radius / planet radius')
        for axis in axes.flat:
            axis.set_xlim(1, settings.radial_max_rp)
        for extension in ('png', 'pdf'):
            with (directory / f'helium_atmosphere.{extension}').open('xb') as stream:
                fig.savefig(stream, format=extension, dpi=180)
        plt.close(fig)
