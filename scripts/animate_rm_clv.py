"""Animate the stellar and spatial sodium calculations used in Tutorial 13.

Run from the repository root:
    python scripts/animate_rm_clv.py RUN_DIRECTORY OUTPUT_DIRECTORY

RUN_DIRECTORY contains resolved.json, observations.npz and the stellar bundle
named by that config. The spatial spectra come from the completed observation.
The factorized stellar illustration evaluates TransitStellarModel at the same
selected phases, using the saved geometry, synthesis and instrumental resolution.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys

# Read EXoPLORE from this checkout when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d

from exoplore.config import SimulationConfig
from exoplore.io.stellar_surface import load_local_intensities
from exoplore.instruments import load_instrument
from exoplore.planets import load_planet
from exoplore.stellar.observation import TransitStellarModel

C_KMS = 299792.458


def make_animations(run_directory, output_directory, frames=32):
    """Save two GIFs using the full run's phases and sodium flux components."""
    root, output = Path(run_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    names = ['tutorial13_stellar_rm_clv.gif', 'tutorial13_spatial_coupling.gif']
    for name in names:
        if (output / name).exists():
            raise FileExistsError(output / name)
    config = SimulationConfig.from_json(root / 'resolved.json')
    if (config.instrument.name != 'ANDES_UBV'
            or not config.atomic_transmission.enabled
            or config.rm_clv.coupling != 'spatial'
            or config.rm_clv.effects != 'rm_clv'):
        raise ValueError('Supply the completed Tutorial 13 spatial ANDES sodium run')
    if type(frames) is not int or frames < 2:
        raise ValueError('Choose at least two animation phases')
    resolution = load_instrument(config).res
    planet = load_planet(config.planet.parameter_file)
    local, _, _ = load_local_intensities(config.rm_clv.model_file,
                                         wavelength_medium='vacuum')
    # Read the noiseless observation and its opaque-body stellar counterpart.
    with np.load(root / 'observations.npz', allow_pickle=False) as data:
        phase = data['phase']
        wave = data['wave_nm'][0] / 1000
        opaque = data['bare_model']
        coupled = data['no_telluric'] / data['throughput'][:, None]
    # Show ingress, full transit and egress, with a short baseline on either side.
    hours = phase * planet.orbital_period_days * 24
    visible = np.flatnonzero(abs(hours) <= planet.transit_duration_hours * .65)
    selected = visible[np.linspace(0, len(visible)-1, frames).round().astype(int)]
    phases, times = phase[selected], hours[selected]
    centre = config.atomic_transmission.line_wavelengths_nm[0] / 1000
    # All profile panels use stellar-rest velocity around Na D2.
    velocity = np.linspace(-65., 65., 321)
    plot_wave = centre * (1 + velocity / C_KMS)
    star_rv = (planet.systemic_velocity_kms
               - planet.stellar_rv_semiamplitude_kms*np.sin(2*np.pi*phases)
               - config.observation.berv_kms)
    planet_rv = (planet.kp_kms + planet.stellar_rv_semiamplitude_kms)*np.sin(2*np.pi*phases)
    # The standalone stellar factor uses the same configuration and its square
    # surface grid. Zero observed stellar velocity places its output in this frame.
    choices = replace(config.rm_clv, coupling='factorized')
    stellar = TransitStellarModel(choices, planet, resolution).evaluate(
        plot_wave, phases, np.zeros(len(phases)),
        config.observation.exposure_time_seconds)
    # Restore stellar-rest wavelengths for the saved spatial observation.
    bare, joint = [], []
    for index, rv in zip(selected, star_rv):
        rest_wave = wave / (1 + rv/C_KMS)
        bare.append(np.interp(plot_wave, rest_wave, opaque[index]))
        joint.append(np.interp(plot_wave, rest_wave, coupled[index]))
    bare, joint = np.asarray(bare), np.asarray(joint)
    # This ratio measures the absorbed fraction relative to the opaque-body
    # observation. It retains the local stellar illumination of atmospheric rays.
    absorption = 100*(1-joint/bare)
    # Sky coordinates follow the same circular-orbit convention as the simulator.
    scaled_a = planet.a/(planet.R_star/1e5)
    px = scaled_a*np.sin(2*np.pi*phases)
    py = -scaled_a*np.cos(2*np.pi*phases)*np.cos(planet.inc)
    radius = planet.R_pl/planet.R_star
    # The stellar disk colours show projected rotation. Brightness follows the
    # supplied continuum at Na D2, preserving its centre-to-limb intensity change.
    axis = np.linspace(-1, 1, 241)
    x, y = np.meshgrid(axis, axis)
    disk = x*x+y*y <= 1
    mu = np.sqrt(np.maximum(1-x*x-y*y, 0))
    column = np.argmin(abs(local.wavelength_um-centre))
    continuum = np.interp(mu, local.mu, local.continuum[:, column])
    angle = np.deg2rad(config.rm_clv.projected_obliquity_deg or 0.)
    rotation = planet.v_rotsini_kms*(x*np.cos(angle)-y*np.sin(angle))
    rgba = plt.get_cmap('coolwarm')((rotation/planet.v_rotsini_kms+1)/2)
    rgba[..., :3] *= (.35+.65*continuum/local.continuum[-1, column])[..., None]
    rgba[..., 3] = disk
    # For the local-profile comparison, interpolate the synthesis at the planet
    # centre and give it the same illustrative instrumental Gaussian width.
    local_profiles = []
    for xx, yy in zip(px, py):
        if xx*xx+yy*yy >= 1:
            local_profiles.append(np.full(len(velocity), np.nan))
            continue
        m = np.sqrt(1-xx*xx-yy*yy)
        upper = np.clip(np.searchsorted(local.mu, m), 1, len(local.mu)-1)
        fraction = (m-local.mu[upper-1])/(local.mu[upper]-local.mu[upper-1])
        spectrum = (1-fraction)*local.intensity[upper-1]+fraction*local.intensity[upper]
        cont = (1-fraction)*local.continuum[upper-1]+fraction*local.continuum[upper]
        rv = planet.v_rotsini_kms*(xx*np.cos(angle)-yy*np.sin(angle))
        rest = plot_wave/(1+rv/C_KMS)
        sigma = C_KMS/resolution/2.354820045/(velocity[1]-velocity[0])
        # Convolve line and continuum intensities separately before their ratio,
        # following the ordering used for the disk-integrated stellar profiles.
        broadened = gaussian_filter1d(np.interp(rest, local.wavelength_um, spectrum), sigma)
        continuum_profile = gaussian_filter1d(np.interp(rest, local.wavelength_um, cont), sigma)
        local_profiles.append(broadened/continuum_profile)
    local_profiles = np.asarray(local_profiles)
    # Short endpoint pauses make the direction of the transit easier to follow.
    sequence = [0, 0, *range(len(selected)), len(selected)-1, len(selected)-1]
    for spatial, filename in zip([False, True], names):
        fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1., 1.45])
        star_ax = fig.add_subplot(gs[:, 0])
        top = fig.add_subplot(gs[0, 1])
        bottom = fig.add_subplot(gs[1, 1], sharex=top)
        star_ax.imshow(rgba, origin='lower', extent=(-1, 1, -1, 1))
        star_ax.add_patch(Circle((0, 0), 1, fill=False, ec='.3', lw=.8))
        star_ax.plot(px, py, color='.4', ls=':', lw=1)
        body = Circle((px[0], py[0]), radius, fc='black', ec='white', lw=1)
        star_ax.add_patch(body)
        if spatial:
            # The ring indicates where atmospheric rays pass. Its displayed
            # width is enlarged for visibility. All curves use the saved fluxes.
            annulus = Circle((px[0], py[0]), radius*1.35, fill=False, ec='#e68a00', lw=3)
            star_ax.add_patch(annulus)
        star_ax.set(xlim=(-1.35, 1.35), ylim=(-1.3, 1.3), aspect='equal',
                    xlabel='Sky position / stellar radius', ylabel='Sky position / stellar radius')
        star_ax.text(-.97, 1.12, 'Approaching', color='#345eaa', fontsize=10)
        star_ax.text(.15, 1.12, 'Receding', color='#b64a3a', fontsize=10)
        clock = star_ax.text(.5, .01, '', transform=star_ax.transAxes, ha='center', fontsize=10)
        top.set(ylabel='Continuum-relative flux', xlim=(-65, 65))
        bottom.set(xlabel='Stellar-frame velocity around Na D2 (km/s)', xlim=(-65, 65))
        for ax in [top, bottom]:
            ax.grid(alpha=.15)
            ax.axvline(0, color='.6', lw=.7, ls=':')
        if spatial:
            top.set_title('Spatially coupled star and planet', fontsize=12)
            first, = top.plot(velocity, bare[0], color='.4', label='Opaque-body transit')
            second, = top.plot(velocity, joint[0], color='#d25c21', label='Star + planetary atmosphere')
            top.set_ylim(max(0, np.min(joint)-.03), 1.01)
            third, = bottom.plot(velocity, absorption[0], color='#d25c21')
            track = bottom.axvline(planet_rv[0], color='black', ls='--', lw=1, label='Planet orbital velocity')
            bottom.set(ylabel='Atmospheric absorbed fraction (%)',
                       ylim=(-.015, float(np.max(absorption)*1.08)))
            bottom.legend(fontsize=8, loc='upper left')
            star_ax.set_title('Local stellar light through atmospheric rays', fontsize=11)
        else:
            top.set_title('Opaque planet: the stellar RM+CLV calculation', fontsize=12)
            top.plot(velocity, stellar.reference[0], color='.3', lw=2, label='Unocculted star')
            first, = top.plot(velocity, stellar.reference[0]*stellar.distortion[0],
                               color='#286aa6', label='Occulted star')
            second, = top.plot(velocity, local_profiles[0], color='#d58b13', alpha=.8,
                                label='Local profile at planet centre')
            top.set_ylim(0, 1.04)
            third, = bottom.plot(velocity, 100*(stellar.distortion[0]-1), color='#286aa6')
            span = float(np.max(abs(100*(stellar.distortion-1))))*1.08
            bottom.set(ylabel='Stellar line distortion (%)', ylim=(-span, span))
            bottom.axhline(0, color='.4', lw=.6)
            star_ax.set_title('Rotation and centre-to-limb brightness', fontsize=11)
        top.legend(fontsize=8, loc='upper right')
        def update(frame):
            body.center = px[frame], py[frame]
            clock.set_text(f'Time from mid-transit: {times[frame]:+.2f} h')
            if spatial:
                annulus.center = body.center
                first.set_ydata(bare[frame])
                second.set_ydata(joint[frame])
                third.set_ydata(absorption[frame])
                track.set_xdata([planet_rv[frame], planet_rv[frame]])
            else:
                first.set_ydata(stellar.reference[frame]*stellar.distortion[frame])
                second.set_ydata(local_profiles[frame])
                third.set_ydata(100*(stellar.distortion[frame]-1))
            return first, second, third, body, clock
        animation = FuncAnimation(fig, update, frames=sequence, interval=220, blit=False)
        animation.save(output/filename, writer=PillowWriter(fps=5), dpi=95)
        plt.close(fig)
    return dict(selected_exposures=selected.tolist(), physical_frames=len(selected),
                rendered_frames=len(sequence), files=names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(make_animations(args.run_directory, args.output_directory), indent=2))


if __name__ == '__main__':
    main()
