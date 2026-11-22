"""Animate the completed Tutorial 14 observations without recalculating them.

Run from the repository root:
    python scripts/animate_stellar_activity_tutorial.py \
        outputs/hd189733b_andes_stellar_activity_rotating docs/figures
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from exoplore.config import SimulationConfig
from exoplore.planets import load_planet
from exoplore.stellar.coupled import SpatialTransitModel
from plot_stellar_activity_tutorial import C_KMS, LABELS, rotation_surface_image
from run_stellar_activity_tutorial import CASES, case_config, file_hash, load_arrays


def exposure_groups(count, frames):
    """Partition the original sequence into contiguous, non-overlapping bins."""
    if not 1 <= frames <= count:
        raise ValueError('The frame count must be between one and the number of exposures')
    return np.array_split(np.arange(count), frames)


def mean_profiles(values, groups):
    """Average finite samples for display, keeping fully masked channels masked."""
    profiles = []
    for group in groups:
        block = values[group]
        count = np.isfinite(block).sum(axis=0)
        profiles.append(np.divide(np.nansum(block, axis=0), count,
            out=np.full(block.shape[1], np.nan), where=count > 0))
    return np.asarray(profiles)


def save_gif(fig,update,count,destination,fps):
    """Keep smooth rotation colours with one shared, dithered GIF palette."""
    frames=[]
    for index in range(count):
        update(index)
        fig.canvas.draw()
        # Copy the canvas before the next frame reuses its memory.
        rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
        frames.append(Image.fromarray(rgb))
    # Sample several times so the palette includes every line and surface colour.
    samples=np.unique(np.linspace(0,count-1,min(5,count),dtype=int))
    sheet=Image.new('RGB',(frames[0].width,frames[0].height*len(samples)))
    for row,index in enumerate(samples):
        sheet.paste(frames[index],(0,row*frames[0].height))
    palette=sheet.quantize(colors=256)
    encoded=[frame.quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG)
             for frame in frames]
    encoded[0].save(destination,save_all=True,append_images=encoded[1:],
        duration=round(1000/fps/10)*10,loop=0,optimize=False,disposal=2)


def prepare_display_rows(config, planet, data, spectral, flux, error, wave, model=None):
    """Apply the tutorial preparation to a saved spectrum for display.

    The pre-correction observation uses the accepted Molecfit products. The
    opaque-body comparison uses the archived atmosphere-free stellar model.
    Both retain the observing sequence and the configured continuum windows.
    """
    from exoplore.observation.timing import get_event
    from exoplore.pipelines.seidel22 import prepare_transmission

    _, inside, outside, _ = get_event(config.observation, planet, data['bjd'])
    centre = config.atomic_transmission.line_wavelengths_nm[0]
    return prepare_transmission(
        wave, flux, error,
        stellar_rv_kms=spectral['stellar_velocity_kms'],
        planet_rv_stellar_kms=(spectral['planet_velocity_kms']
                              - spectral['stellar_velocity_kms']),
        reference_indices=outside, transit_indices=inside,
        settings=config.pipeline.seidel22,
        telluric_corrected=True, sky_subtracted=True,
        stellar_model_flux=model,
        lightcurve_window_nm=(centre*(1-20/C_KMS), centre*(1+20/C_KMS)))


def residual_limit(arrays):
    """Use one fixed symmetric colour limit from finite residual samples."""
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    return max(float(np.quantile(abs(finite), .98)), 1e-6)


def residual_matrix(axis, values, velocity, hours, limit, title, planet_velocity):
    """Show every exposure, with masked pixels white and zero residual neutral."""
    from matplotlib.colors import TwoSlopeNorm

    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('white')
    artist = axis.pcolormesh(
        velocity, hours, values, shading='nearest', cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axis.plot(planet_velocity, hours, 'k--', lw=.7, alpha=.7)
    axis.axvline(0, color='.3', ls=':', lw=.6)
    axis.set(title=title, xlabel='Stellar-frame velocity (km/s)',
             ylabel='Hours from mid-transit', xlim=(-30, 30))
    return artist


def animate_case(root, config, name, destination, frames=32, fps=3):
    """Connect the rotating disk, noisy residual matrices and recovered Na line."""
    current = case_config(config, name)
    planet = load_planet(config.planet.parameter_file)
    centre = config.atomic_transmission.line_wavelengths_nm[0]
    matched = load_arrays(root/name/'matched/Na_D2_transmission.npz')
    assumed = load_arrays(root/name/'quiet/Na_D2_transmission.npz')
    data = load_arrays(root/name/'observations.npz')
    spectral = load_arrays(root/name/'intrinsic_spectra.npz')
    corrections = load_arrays(root/name/'corrected_observations.npz')
    for key in ('phase', 'stellar_wave_nm', 'planet_wave_nm'):
        np.testing.assert_array_equal(assumed[key], matched[key])
    np.testing.assert_array_equal(data['phase'], matched['phase'])

    # Retain the same accepted telluric samples used in the archived analysis.
    good = ((corrections['transmittance'] > config.tellurics.mask_threshold)
            & np.isfinite(corrections['flux']) & np.isfinite(corrections['error'])
            & (corrections['error'] > 0))
    observed = prepare_display_rows(
        config, planet, data, spectral,
        np.where(good, corrections['flux'], np.nan),
        np.where(good, corrections['error'], np.nan), corrections['wave_nm'])
    opaque = prepare_display_rows(
        config, planet, data, spectral,
        data.get('matched_bare_model', data['bare_model']),
        data['raw_error'], data['wave_nm'])
    wave = matched['stellar_wave_nm']
    # Molecfit's wavelength refinement can shift the display grid slightly.
    # Put only this plotting copy on the archived corrected-product grid.
    before = np.array([np.interp(
        wave, observed.stellar_wave_nm, row, left=np.nan, right=np.nan)
        for row in observed.stellar_transmission])
    velocity = C_KMS*(wave/centre-1)
    selected = abs(velocity) <= 30
    velocity = velocity[selected]
    before = 100*(before[:, selected]-1)
    after = 100*(assumed['stellar_transmission'][:, selected]-1)
    corrected = 100*(matched['stellar_transmission'][:, selected]-1)
    model_display = np.array([np.interp(
        wave, opaque.stellar_wave_nm, row, left=np.nan, right=np.nan)
        for row in opaque.stellar_transmission])
    model_rows = 100*(model_display[:, selected]-1)
    limit = residual_limit([before, after, corrected])

    # The matrices retain every exposure. Moving profiles average each frame's group.
    hours = data['phase']*planet.orbital_period_days*24
    groups = exposure_groups(len(hours), frames)
    group_hours = np.array([hours[group].mean() for group in groups])
    times = np.array([data['bjd'][group].mean() for group in groups])
    phase = np.array([data['phase'][group].mean() for group in groups])
    planet_velocity = spectral['planet_velocity_kms']-spectral['stellar_velocity_kms']
    group_velocity = np.array([planet_velocity[group].mean() for group in groups])
    profiles = [mean_profiles(values, groups)
                for values in (before, corrected, after)]
    model_profiles = mean_profiles(model_rows, groups)
    surface = SpatialTransitModel(current.rm_clv, planet, 100000,
                                  surface_config=current.stellar_surface)
    coordinate = np.linspace(-1, 1, 181)
    x, y = np.meshgrid(coordinate, coordinate)
    scaled_a = planet.a/(planet.R_star/1e5)
    px = scaled_a*np.sin(2*np.pi*phase)
    py = -scaled_a*np.cos(2*np.pi*phase)*np.cos(planet.inc)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    disk, raw_map, corrected_map, original_line, corrected_line, coadd = axes.flat
    image = disk.imshow(rotation_surface_image(surface, x, y, times[0], centre/1000),
                        origin='lower', extent=(-1, 1, -1, 1))
    disk.add_patch(Circle((0, 0), 1, fill=False, color='black', lw=.8))
    disk.plot(px, py, ':', color='.3')
    radius = planet.R_pl/planet.R_star
    annulus = None
    if current.rm_clv.coupling == 'spatial':
        # The enlarged, translucent ring locates transmitted atmospheric rays.
        # Its inner edge meets the drawn catalog-radius disk. Spectra retain
        # the actual pRT radii and transmissions of the completed observation.
        annulus = Wedge((px[0], py[0]), 1.35*radius, 0, 360,
                        width=.35*radius, facecolor='#e68a00',
                        edgecolor='#ba6b00', alpha=.38, lw=1, zorder=4)
        disk.add_patch(annulus)
        disk.text(.5, .015, 'Atmospheric annulus: width enlarged',
                  transform=disk.transAxes, ha='center', fontsize=8)
    body = Circle((px[0], py[0]), radius, color='black', zorder=5)
    disk.add_patch(body)
    disk.set(aspect='equal', xlim=(-1.3, 1.3), ylim=(-1.15, 1.15),
             title=LABELS[name], xlabel='x / stellar radius', ylabel='y / stellar radius')
    for axis, values, title in (
            (raw_map, before, 'Noisy data: before stellar correction'),
            (corrected_map, after, 'Noisy data: quiet-star correction')):
        artist = residual_matrix(axis, values, velocity, hours, limit, title, planet_velocity)
        fig.colorbar(artist, ax=axis, label='Residual (%)')
    cursors = [axis.axhline(group_hours[0], color='black', lw=1.2)
               for axis in (raw_map, corrected_map)]

    line_limit = 1.3*residual_limit([profiles[0], model_profiles])
    original_line.set(title='Current data and opaque-planet model',
                      xlabel='Stellar-frame velocity (km/s)', ylabel='Residual (%)',
                      xlim=(-30, 30), ylim=(-line_limit, line_limit))
    model_line, = original_line.plot(velocity, model_profiles[0], color='black',
                                    ls='--', label='Opaque-planet model', alpha=.8)
    observed_line, = original_line.plot(velocity, profiles[0][0], color='#80502c',
                                       label='Noisy star + planet', alpha=.85)
    original_line.axhline(0, color='.6', lw=.6)
    original_line.legend(fontsize=8)
    line_limit = 1.3*residual_limit(profiles[1:])
    corrected_line.set(title='Same data: two stellar corrections',
                       xlabel='Stellar-frame velocity (km/s)', ylabel='Residual (%)',
                       xlim=(-30, 30), ylim=(-line_limit, line_limit))
    matched_line, = corrected_line.plot(velocity, profiles[1][0], color='#1768a6',
                                       label='Matched', alpha=.8)
    quiet_line, = corrected_line.plot(velocity, profiles[2][0], color='#ca5727',
                                     label='Quiet-star', alpha=.8)
    corrected_line.axhline(0, color='.5', lw=.7)
    marker = corrected_line.axvline(group_velocity[0], color='black', ls='--', lw=.8)
    corrected_line.legend(fontsize=8)
    # Full-transit coadds stay fixed while the cursor moves through individual exposures.
    planet_v = C_KMS*(matched['planet_wave_nm']/centre-1)
    band = abs(planet_v) <= 25
    for saved, color, label in ((matched, '#1768a6', 'Matched correction'),
                                (assumed, '#ca5727', 'Quiet-star correction')):
        coadd.plot(planet_v[band], 100*(saved['planet_coadd'][band]-1),
                   color=color, alpha=.85, label=label)
    coadd.plot(planet_v[band], 100*(matched['truth_planet_coadd'][band]-1),
               color='black', ls='--', lw=1, label='Matched noiseless prediction')
    coadd.axhline(0, color='.6', lw=.7)
    coadd.set(title='Full-transit Na D2 recovery (fixed)',
              xlabel='Planet-frame velocity (km/s)', ylabel='Transmission residual (%)',
              xlim=(-25, 25))
    coadd.legend(fontsize=7)
    clock = fig.suptitle('', fontsize=13)

    def update(frame):
        image.set_data(rotation_surface_image(surface, x, y, times[frame], centre/1000))
        body.center = (px[frame], py[frame])
        if annulus is not None:
            annulus.set_center(body.center)
        for line in cursors:
            line.set_ydata([group_hours[frame], group_hours[frame]])
        model_line.set_ydata(model_profiles[frame])
        observed_line.set_ydata(profiles[0][frame])
        matched_line.set_ydata(profiles[1][frame])
        quiet_line.set_ydata(profiles[2][frame])
        marker.set_xdata([group_velocity[frame], group_velocity[frame]])
        clock.set_text(f'{LABELS[name]} | {group_hours[frame]:+.2f} h'
                       ' | red = excess flux, blue = absorption')

    save_gif(fig, update, len(groups), destination, fps)
    plt.close(fig)
    return dict(file=destination.name, frames=len(groups), fps=fps,
                exposures_per_frame=[len(group) for group in groups],
                frame_phase=phase.tolist(), bytes=destination.stat().st_size,
                sha256=file_hash(destination),
                matrix_exposures=len(hours), matrix_colour_limit_percent=limit,
                colour_limit_percentile=98,
                observation='saved noisy observation shared by both corrections',
                illustrative_annulus_outer_radius_factor=(1.35 if annulus is not None else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('figure_directory', type=Path)
    args = parser.parse_args()
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig.from_json(args.run_directory/'campaign.json')
    records = [animate_case(args.run_directory, config, name,
        args.figure_directory/f'tutorial14_{name}.gif') for name in CASES[1:]]
    (args.run_directory/'animation_inventory.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
