#!/usr/bin/env python
"""Build a cheap analytic surface example and its disk/spectrum diagnostic.

Run from the repository root with its scientific environment active:
    PYTHONPATH=src python scripts/example_stellar_components.py --output /tmp/stellar_example
The generated grids are controlled examples for learning the input format.
Replace them with physical local intensities for a scientific simulation.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from exoplore.config import SimulationConfig, StellarComponentConfig
from exoplore.planets import load_planet
from exoplore.stellar.coupled import SpatialTransitModel
from exoplore.plotting.stellar_surface import plot_stellar_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep each component on a common absolute scale before any disk integration.
    wave = np.linspace(.5885, .5915, 601)
    mu = np.array([0., .25, .5, .75, 1.])
    components = {}
    for name, brightness, depth, limb in [('quiet', 1., .5, .5), ('cool_region', .4, .8, .5),
                                          ('facular_region', 1.2, .2, .1)]:
        continuum = brightness*(limb+(1-limb)*mu[:, None])*np.ones_like(wave)
        intensity = continuum*(1-depth*np.exp(-.5*((wave-.59)/.00004)**2))
        path = args.output/f'{name}.npz'
        np.savez(path, wavelength_um=wave, mu=mu, intensity=intensity, continuum=continuum,
                 intensity_kind='specific_intensity', intensity_unit='shared analytic units',
                 wavelength_medium='vacuum', resolving_power=0., source='analytic example',
                 metadata_json=json.dumps({'purpose': 'controlled numerical illustration'}))
        components[name] = StellarComponentConfig(str(path))
    # The configuration uses the same root objects as an ordinary EXoPLORE run.
    config = SimulationConfig.from_json('configs/stellar_surface_example.json')
    surface = replace(config.stellar_surface, components=components)
    choices = replace(config.rm_clv, model_file=components['quiet'].intensity_model_file)
    config.stellar_surface, config.rm_clv = surface, choices
    config.to_json(args.output/'surface_example.json')
    planet = load_planet(config.planet.parameter_file)
    model = SpatialTransitModel(choices, planet, 100000., surface_config=surface)
    quiet = SpatialTransitModel(choices, planet, 100000.)
    detector_wave = np.linspace(.5897, .5903, 81)
    phases = np.linspace(-.025, .025, 21)
    # An opaque planet isolates stellar contamination in this lightweight demonstration.
    result = model.evaluate(detector_wave, phases, 0., 30., chords=None, planet_velocity_kms=0.)
    reference = quiet.evaluate(detector_wave, phases, 0., 30., chords=None, planet_velocity_kms=0.)
    a = planet.a/(planet.R_star/1e5)
    chord = np.column_stack((a*np.sin(2*np.pi*phases), -a*np.cos(2*np.pi*phases)*np.cos(planet.inc)))
    figure = plot_stellar_surface(model, planet_position=chord[len(phases)//2], transit_chord=chord,
        wavelength_um=detector_wave, quiet_spectrum=reference.reference[0], heterogeneous_spectrum=result.reference[0])
    figure.savefig(args.output/'surface_components.png', dpi=150)
    np.savez_compressed(args.output/'surface_spectra.npz', wavelength_um=detector_wave,
        phase=phases, flux=result.flux, quiet_flux=reference.flux)
    (args.output/'stellar_surface_provenance.json').write_text(json.dumps(model.field.provenance(), indent=2))
    print('Example products:', args.output)


if __name__ == '__main__':
    main()
