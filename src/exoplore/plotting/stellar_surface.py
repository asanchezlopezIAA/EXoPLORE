"""Show the component geometry and its effect on a transit spectrum."""
import numpy as np


def plot_stellar_surface(model, *, time_bjd=None, planet_position=None,
                         transit_chord=None, wavelength_um=None,
                         quiet_spectrum=None, heterogeneous_spectrum=None):
    """Return a figure without writing files or importing plotting into the physics.

    Positions use stellar radii in the same sky frame as the transit model.
    The optional spectra share their caller's flux scale and wavelength grid.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch

    if model.field is None:
        raise ValueError('Supply a model with an active component-region configuration')
    field = model.field
    # Render a dense map independently of the numerical integration resolution.
    axis = np.linspace(-1., 1., 401)
    x, y = np.meshgrid(axis, axis)
    names = list(field.components)
    labels = field.labels(x, y, time_bjd)
    image = np.array([[names.index(name) for name in row] for row in labels], float)
    image[x*x+y*y > 1] = np.nan
    colors = plt.get_cmap('tab10')(np.arange(len(names)) % 10)
    figure, axes = plt.subplots(1, 2 if wavelength_um is not None else 1,
                               figsize=(10, 4) if wavelength_um is not None else (5, 4), squeeze=False)
    disk = axes[0, 0]
    disk.imshow(image, origin='lower', extent=(-1, 1, -1, 1),
                cmap=ListedColormap(colors), vmin=-.5, vmax=len(names)-.5)
    disk.add_patch(Circle((0, 0), 1, fill=False, color='black', lw=1))
    if transit_chord is not None:
        disk.plot(*np.asarray(transit_chord).T, color='black', ls=':', lw=1)
    if planet_position is not None:
        disk.add_patch(Circle(planet_position, model.planet.R_pl/model.planet.R_star,
                              color='black', alpha=.65))
    disk.set(xlabel='x / stellar radius', ylabel='y / stellar radius', aspect='equal')
    disk.legend(handles=[Patch(color=colors[i], label=name) for i, name in enumerate(names)],
                loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    if wavelength_um is not None:
        spectrum = axes[0, 1]
        spectrum.plot(wavelength_um, quiet_spectrum, label='Quiet surface', alpha=.8)
        spectrum.plot(wavelength_um, heterogeneous_spectrum, label='Configured surface', alpha=.8)
        spectrum.set(xlabel='Wavelength (micrometres)', ylabel='Continuum-relative flux')
        spectrum.legend(fontsize=8)
    figure.tight_layout()
    return figure
