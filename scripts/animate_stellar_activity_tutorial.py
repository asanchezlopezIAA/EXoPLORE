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
from matplotlib.patches import Circle
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


def intrinsic_profiles(path, centre_nm, velocity, groups):
    """Display each stored star-plus-planet model on a common stellar-frame axis.

    The saved intrinsic flux already contains exposure integration and spatial
    atmospheric transport. We only shift its plotting coordinates and average
    the selected exposures. Its original continuum is retained.
    """
    with np.load(path, allow_pickle=False) as saved:
        wave = saved['wavelength_um']*1000
        flux, continuum = saved['flux'], saved['continuum']
        stellar_velocity = saved['stellar_velocity_kms']
        values = np.array([np.interp(velocity,
            C_KMS*(wave/centre_nm-1)-stellar_velocity[n], flux[n]/continuum[n])
            for n in range(len(flux))])
    return mean_profiles(values, groups)


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


def animate_case(root, config, name, destination, frames=32, fps=3):
    """Synchronize the actual geometry, intrinsic line and paired preparations."""
    current = case_config(config, name)
    planet = load_planet(config.planet.parameter_file)
    centre = config.atomic_transmission.line_wavelengths_nm[0]
    matched = load_arrays(root/name/'matched/Na_D2_transmission.npz')
    assumed = load_arrays(root/name/'quiet/Na_D2_transmission.npz')
    reference = load_arrays(root/'quiet/matched/Na_D2_transmission.npz')
    # Both preparations must describe the same exposure sequence and wavelength axis.
    for other in (assumed, reference):
        np.testing.assert_array_equal(other['phase'], matched['phase'])
        np.testing.assert_array_equal(other['stellar_wave_nm'], matched['stellar_wave_nm'])
    groups = exposure_groups(len(matched['phase']), frames)
    phase = np.array([matched['phase'][group].mean() for group in groups])
    with np.load(root/name/'observations.npz') as saved:
        times=np.array([saved['bjd'][group].mean() for group in groups])
    velocity = C_KMS*(matched['stellar_wave_nm']/centre-1)
    selected = abs(velocity) <= 45
    profiles = [100*(mean_profiles(data['stellar_transmission'][:, selected], groups)-1)
                for data in (reference, matched, assumed)]
    line_velocity = np.linspace(-30, 30, 401)
    intrinsic = [intrinsic_profiles(root/case/'intrinsic_spectra.npz', centre,
                                   line_velocity, groups) for case in ('quiet', name)]

    # Draw the same projected component map used by the surface integrator.
    model = SpatialTransitModel(current.rm_clv, planet, 100000,
                                surface_config=current.stellar_surface)
    axis = np.linspace(-1, 1, 241)
    x, y = np.meshgrid(axis, axis)
    a = planet.a/(planet.R_star/1e5)
    px = a*np.sin(2*np.pi*phase)
    py = -a*np.cos(2*np.pi*phase)*np.cos(planet.inc)

    fig = plt.figure(figsize=(10.8, 6.4),dpi=90)
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.5], hspace=.55, wspace=.35)
    disk = fig.add_subplot(grid[:, 0])
    star = fig.add_subplot(grid[0, 1])
    change = fig.add_subplot(grid[1, 1])
    trans = fig.add_subplot(grid[2, 1])
    surface_image=disk.imshow(rotation_surface_image(model,x,y,times[0],centre/1000),
                             origin='lower',extent=(-1,1,-1,1))
    disk.add_patch(Circle((0, 0), 1, fill=False, color='black', lw=1))
    disk.plot(px, py, ':', color='.35', lw=1)
    body = Circle((px[0], py[0]), planet.R_pl/planet.R_star, color='black', alpha=.65)
    disk.add_patch(body)
    disk.set(xlim=(-1.35, 1.35), ylim=(-1.15, 1.15), aspect='equal',
             xlabel=r'$x/R_\star$', ylabel=r'$y/R_\star$', title=LABELS[name])
    # Fixed axes across frames keep a changing scale from mimicking line variability.
    star.set(xlim=(-30, 30), ylim=(0, min(1.05, np.max(intrinsic)+.04)),
             ylabel=r'$F/C$', title='Star + planet, before instrumental convolution')
    star.tick_params(labelsize=8)
    active_color = '#cc8e11' if name == 'crossed_facula' else '#75503b'
    stellar_lines = [star.plot([], [], color=c, lw=1.5, alpha=.85, label=label)[0]
        for c, label in [('.45', 'Quiet reference'), (active_color, 'Active surface')]]
    star.legend(fontsize=7, loc='upper right')
    # The relative curve makes small line-shape changes visible when the
    # quiet and active profiles overlap at the scale of the full stellar line.
    difference = 100*(intrinsic[1]/intrinsic[0]-1)
    low, high = difference.min(), difference.max()
    padding = .08*(high-low)
    change.set(xlim=(-30, 30), ylim=(low-padding, high+padding),
        ylabel='Change from quiet (%)', title='Intrinsic line-shape change')
    change.axhline(0, color='.5', ls=':', lw=.8)
    change_line, = change.plot([], [], color=active_color, lw=1.5)
    change.tick_params(labelsize=8)
    finite = np.concatenate([p[np.isfinite(p)] for p in profiles])
    lower, upper = finite.min(), finite.max()
    margin = .08*(upper-lower)
    trans.set(xlim=(-45, 45), ylim=(lower-margin, upper+margin),
        xlabel='Stellar-frame velocity (km/s)', ylabel='Transmission excess (%)',
        title='Prepared observation: short exposure averages')
    trans.tick_params(labelsize=8)
    trans.axhline(0, color='.6', lw=.6)
    prepared_lines = [trans.plot([], [], color=c, lw=lw, ls=ls, alpha=.85, label=label)[0]
        for c, lw, ls, label in [('.6', 1, ':', 'Quiet reference'),
            ('#1768a6', 1.6, '-', 'Matched correction'),
            ('#ca5727', 1.6, '-', 'Quiet-star correction')]]
    track = trans.axvline(0, color='black', ls='--', lw=.8, label='Planet velocity')
    trans.legend(fontsize=6.5, loc='upper right', ncol=2)
    clock = fig.suptitle('', fontsize=11)
    fig.subplots_adjust(left=.09, right=.98, bottom=.09, top=.88)

    def update(frame):
        body.center = (px[frame], py[frame])
        surface_image.set_data(rotation_surface_image(model,x,y,times[frame],centre/1000))
        for line, values in zip(stellar_lines, intrinsic):
            line.set_data(line_velocity, values[frame])
        change_line.set_data(line_velocity, difference[frame])
        for line, values in zip(prepared_lines, profiles):
            line.set_data(velocity[selected], values[frame])
        rv = (planet.kp_kms+planet.stellar_rv_semiamplitude_kms)*np.sin(2*np.pi*phase[frame])
        track.set_xdata([rv, rv])
        hours = phase[frame]*planet.orbital_period_days*24
        clock.set_text(f'{LABELS[name]}  |  {hours:+.2f} h from mid-transit'
                       f'  |  {len(groups[frame])} exposures')

    save_gif(fig,update,len(groups),destination,fps)
    plt.close(fig)
    return dict(file=destination.name, frames=len(groups), fps=fps,
                exposures_per_frame=[len(group) for group in groups],
                frame_phase=phase.tolist(), bytes=destination.stat().st_size,
                sha256=file_hash(destination))


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
