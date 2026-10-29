"""Plot stellar-correction comparisons from saved standard CCF sodium runs.

Usage: python scripts/plot_rm_clv_sodium.py CONFIG RUN_DIRECTORY OUTPUT_DIRECTORY
The run must retain its RM+CLV model, flux and uncertainties.
Saved CCF products are needed only with --include-ccf.


The standard CCF runner saves rm_clv_order, mat_noise and std_noise products
read here. The integrated direct-sodium runner writes its three-panel figures
from core/atomic_simulation.py through plotting/helium.py. This script
compares matched and factorized stellar corrections on a saved observation.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from exoplore.config import SimulationConfig
from exoplore.observation import get_event
from exoplore.pipelines.seidel22 import prepare_transmission
from exoplore.pipelines.helium_math import window_mask
from exoplore.planets import load_planet
from exoplore.stellar.observation import TransitStellarModel


def _product(root, pattern):
    """Require one matching product so comparison runs cannot be mixed silently."""
    # Find the saved product under the supplied run directory.
    paths = list(root.rglob(pattern))
    # Require a single match so the plot uses one unambiguous run product.
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} under {root}, found {len(paths)}")
    return paths[0]


def _transmission_diagnostic(results, phase, planet, relative_velocity, output,
                             half_width_kms):
    """Show the direct-transmission products with the Tutorial 12 convention.

    Negative excess means absorption. The light curves use complete fixed
    bands after planetary alignment, as in the helium transmission plots.
    The comparison curves show the prepared fluxes from each stellar-correction choice.
    """
    # Use the selected corrected case for the stellar-frame residual map.
    corrected = results['RM+CLV corrected']
    # Convert phase to hours from mid-transit for the time axes.
    hours = phase*planet.orbital_period_days*24
    # Mark the first and last optical contacts from the supplied transit duration.
    contacts = [-planet.transit_duration_hours/2, planet.transit_duration_hours/2]
    # Give each sodium line its own map, planet-frame spectrum and light curve.
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    # Collect the plotted band measurements for numerical inspection.
    products = dict(hours=hours, phase=phase)
    # Use the vacuum D2 and D1 rest wavelengths in nm for the two plot columns.
    for column, (centre, line) in enumerate(zip([589.158326, 589.755814], ['D2', 'D1'])):
        # Convert the common stellar-frame wavelength coordinate to velocity around this line.
        velocity = 299792.458*(corrected.stellar_wave_nm/centre-1)
        # Display the interval within 80 km/s of the stellar rest wavelength.
        selected = abs(velocity) <= 80
        # Plot percent transmission excess against time, with absorption shown in red.
        image = axes[0, column].pcolormesh(velocity[selected], hours,
            100*(corrected.stellar_transmission[:, selected]-1),
            shading='auto', cmap='RdBu', vmin=-.2, vmax=.2)
        # Overlay the configured planet motion relative to the stellar frame.
        axes[0, column].plot(relative_velocity, hours, 'k--', lw=1,
                             label='Planetary velocity')
        for contact in contacts:
            axes[0, column].axhline(contact, color='k', ls=':', lw=1)
        axes[0, column].set(xlim=(-80, 80), xlabel='Stellar-frame velocity (km/s)',
                            ylabel='Time from mid-transit (h)', title='Na '+line)
        axes[0, column].legend(fontsize=7, loc='upper left')
        fig.colorbar(image, ax=axes[0, column], label='Transmission excess (%)')
        # Convert the requested planet-frame band half-width from km/s to nm.
        bounds = [[centre*(1-half_width_kms/299792.458),
                   centre*(1+half_width_kms/299792.458)]]
        # Show every correction choice using the same planet-frame coordinates and band.
        for name, result in results.items():
            # Express each planet-frame wavelength grid as velocity around its rest line.
            velocity = 299792.458*(result.planet_wave_nm/centre-1)
            # Convert the coadded ratio to percent excess for comparison.
            axes[1, column].plot(velocity, 100*(result.planet_coadd-1),
                                 label=name, alpha=.85)
            # np.mean preserves missing-band coverage as NaN, keeping the
            # requested band definition consistent across exposures.
            # Select the requested planet-following wavelength band.
            band = window_mask(result.planet_wave_nm, bounds)
            if band.sum() < 2:
                raise ValueError('The light-curve band contains fewer than two pixels')
            # Calculate one band-mean percent excess for every exposure.
            curve = 100*(np.mean(result.planet_transmission[:, band], axis=1)-1)
            axes[2, column].plot(hours, curve, '.-', ms=3, label=name, alpha=.85)
            # Create a stable name for the saved line/correction light-curve arrays.
            key = line+'_'+name.lower().replace(' ', '_').replace('+', '_')
            products[key+'_excess_percent'] = curve
        # Retain the exact wavelength limits and pixel count of the plotted band.
        products[line+'_band_nm'] = np.asarray(bounds[0])
        products[line+'_band_pixels'] = np.asarray(int(band.sum()))
        axes[1, column].set(xlim=(-80, 80), xlabel='Planet-frame velocity (km/s)',
                            ylabel='Transmission excess (%)')
        axes[2, column].set(xlabel='Time from mid-transit (h)',
            ylabel='Band-averaged excess (%)', title=f'Planet-frame band: ±{half_width_kms:g} km/s')
        for contact in contacts:
            axes[2, column].axvline(contact, color='k', ls=':', lw=1)
        for row in (1, 2):
            axes[row, column].axhline(0, color='grey', lw=.7)
            axes[row, column].grid(alpha=.2)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Sodium transmission: corrected residual maps, spectra and light curves')
    fig.tight_layout()
    fig.savefig(output/'transmission_diagnostic.png', dpi=150)
    plt.close(fig)
    # Save the plotted light curves alongside the figure.
    np.savez_compressed(output/'lightcurves.npz', **products)


def plot_saved_run(config_path, run_directory, output_directory, *,
                   half_width_kms=20., include_ccf=False, compare_factorized=False):
    """Use the saved observation for both stellar-correction choices."""
    # Read the config that belongs to the supplied saved run.
    config = SimulationConfig.from_json(config_path)
    # Load its stellar velocities, orbital parameters and transit timing.
    planet = load_planet(config.planet.parameter_file)
    # Keep the saved observation source separate from the new plot directory.
    root, output = Path(run_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    # Read the saved stellar model, frame coordinates and correction-stage metadata.
    with np.load(_product(root, 'rm_clv_order*.npz')) as saved:
        stellar = {name: saved[name] for name in saved.files}
    wave, phase = stellar['wavelength_um'], stellar['phase']
    distortion = stellar['distortion']
    # Read the reference-night observing times used by this standard simulation.
    jd = fits.getdata(Path(config.paths.inputs_dir)/'reference_night/julian_date_0.fits')
    # Reconstruct the same in-transit and reference-exposure selection.
    _, inside, outside, _ = get_event(config.observation, planet, jd)

    # Read one saved standard-run matrix from its array archive.
    def matrix(prefix):
        with np.load(_product(root, prefix+'*.npz')) as saved:
            return saved['a'][0]

    # Restore the pre-correction observation, then process the same flux twice.
    flux, error = matrix('mat_noise_order'), matrix('std_noise_order')
    # Undo a native-flux stellar correction when it was already applied during simulation,
    # so every comparison below starts from the same pre-correction observation.
    if bool(stellar['correct']) and str(stellar.get('correction_stage', 'native_flux')) == 'native_flux':
        flux, error = flux*distortion, error*distortion
    # Reconstruct the opaque-body stellar spectrum from its saved factors.
    model_flux = stellar['reference']*distortion
    # For a spatial model, also restore the opaque-body continuum transit factor.
    if 'opaque_transmission' in stellar:
        model_flux *= stellar['opaque_transmission']
    # Prepare a separate factorized correction when that comparison is requested.
    simple_flux = None
    # Use the configured alternative intensity bundle or the same injected stellar synthesis.
    if compare_factorized or config.rm_clv.correction_method == 'factorized':
        choices = replace(config.rm_clv, coupling='factorized',
            model_file=config.rm_clv.correction_model_file or config.rm_clv.model_file)
        # Calculate the factorized stellar model at the CARMENES VIS resolution used here.
        simple = TransitStellarModel(choices, planet, 94600.).evaluate(
            wave, phase, stellar['stellar_velocity_kms'],
            config.observation.exposure_time_seconds)
        simple_flux = simple.reference*simple.distortion
    # Choose the matched or factorized stellar spectrum for the main corrected result.
    selected_model = (simple_flux if config.rm_clv.correction_method == 'factorized'
                      else model_flux)
    # Process the same observed flux with stellar correction omitted and applied.
    cases = [('RM+CLV uncorrected', flux, error, None),
             ('RM+CLV corrected', flux, error, selected_model)]
    if compare_factorized and config.rm_clv.correction_method == 'matched':
        cases.append(('Factorized correction', flux, error, simple_flux))
    if 'opaque_transmission' in stellar:
        # The opaque-body diagnostic retains transit geometry and stellar effects,
        # while omitting atmospheric opacity. Use it to assess residual structure.
        bare = stellar['reference']*stellar['opaque_transmission']*distortion
        cases.append(('Opaque body, corrected', bare, error, selected_model))
    # Use the saved planet velocity when present, or reconstruct its circular-orbit convention.
    velocity = stellar.get('planet_velocity_kms',
        planet.kp_kms*np.sin(2*np.pi*phase)+planet.systemic_velocity_kms
        - config.observation.berv_kms)
    label = 'Spatial coupling' if str(stellar['coupling']) == 'spatial' else 'Factorized coupling'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    results = {}
    # Apply Seidel22 separately to each correction case while keeping the same data selection.
    for name, spectra, uncertainties, correction in cases:
        # Supply the observed stellar shift and the planet's relative motion to preparation.
        result = prepare_transmission(wave*1000, spectra, uncertainties,
            stellar_rv_kms=stellar['stellar_velocity_kms'],
            planet_rv_stellar_kms=velocity-stellar['stellar_velocity_kms'],
            reference_indices=outside, transit_indices=inside,
            settings=config.pipeline.seidel22, telluric_corrected=True,
            sky_subtracted=True, stellar_model_flux=correction)
        results[name] = result
        for ax, centre, line in zip(axes, [589.158326, 589.755814], ['D2', 'D1']):
            ax.plot(299792.458*(result.planet_wave_nm/centre-1),
                    100*(1-result.planet_coadd), label=name, alpha=.85)
            ax.set(xlim=(-80, 80), xlabel='Planet-frame velocity (km/s)',
                   ylabel='Excess absorption (%)', title='Na '+line)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(label+': sodium transmission spectra')
    fig.tight_layout()
    fig.savefig(output/'profiles.png', dpi=150)
    plt.close(fig)

    # Write the shared map, spectrum and band-curve comparison for both lines.
    _transmission_diagnostic(results, phase, planet,
        velocity-stellar['stellar_velocity_kms'], output, half_width_kms)
    # Finish after the direct-transmission comparisons unless CCF plots were requested.
    if not include_ccf:
        return

    # These are the actual saved CCF and map, with their own velocity axes.
    # Load the actual per-exposure CCF and its velocity coordinate from this run.
    with np.load(_product(root, 'ccf_store_order*.npz')) as saved:
        ccf = saved['a'][0].T
    with np.load(_product(root, 'v_ccf_*.npz')) as saved:
        lag = saved['a']
    # Read the saved normalized Kp-Vrest map and its two axes.
    with np.load(_product(root, 'ccf_tot_sn_map*.npz')) as saved:
        values, vrest, kp = saved['ccf_tot_sn'], saved['v_rest'], saved['kp_range']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    image = axes[0].pcolormesh(lag, phase, ccf, shading='auto')
    fig.colorbar(image, ax=axes[0], label='CCF statistic')
    axes[0].plot(velocity, phase, 'w--', lw=1, label='Configured orbital guide')
    axes[0].legend(fontsize=7)
    axes[0].set(xlabel='Earth-frame trial velocity (km/s)', ylabel='Orbital phase',
                title=label+': time–velocity CCF')
    image = axes[1].pcolormesh(vrest, kp, values.T, shading='auto')
    fig.colorbar(image, ax=axes[1], label='Off-peak-normalized map statistic')
    axes[1].axhline(planet.kp_kms, color='k', ls=':')
    axes[1].axvline(0, color='k', ls=':')
    axes[1].set(xlabel='Vrest (km/s)', ylabel='Kp (km/s)', ylim=(0, 250),
                title='Noiseless sodium experiment' if config.observation.noiseless
                else 'Sodium detection map')
    fig.tight_layout()
    fig.savefig(output/'ccf_map.png', dpi=150)
    plt.close(fig)


# Parse the run paths and optional comparison choices when invoked from the shell.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--half-width-kms', type=float, default=20.,
                        help='Half-width of each planetary-frame light-curve band')
    parser.add_argument('--include-ccf', action='store_true',
                        help='Also plot the saved CCF and detection map')
    parser.add_argument('--compare-factorized', action='store_true',
                        help='Also process the same observation with a factorized stellar correction')
    arguments = parser.parse_args()
    if not np.isfinite(arguments.half_width_kms) or arguments.half_width_kms <= 0:
        parser.error('--half-width-kms must be finite and positive')
    plot_saved_run(arguments.config, arguments.run_directory, arguments.output_directory,
                  half_width_kms=arguments.half_width_kms, include_ccf=arguments.include_ccf,
                  compare_factorized=arguments.compare_factorized)
