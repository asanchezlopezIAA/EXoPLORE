#!/usr/bin/env python3
"""Prepare one VULCAN column using the same simulation JSON as the observing run."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'src'))

from exoplore.atmosphere.models import RadiativeTransferInputs
from exoplore.atmosphere.pressure import create_log_pressure_grid
from exoplore.atmosphere.temperature import calculate_temperature_structure
from exoplore.atmosphere.vulcan import run_vulcan
from exoplore.config import SimulationConfig
from exoplore.planets import load_planet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='EXoPLORE simulation JSON')
    parser.add_argument('--region', default='planet_model', choices=(
        'planet_model','ccf_template','morning_day','morning_night','evening_day','evening_night'))
    args = parser.parse_args()
    config = SimulationConfig.from_json(args.config)
    region = getattr(config.atmosphere, args.region)
    if region is None or region.chemistry_backend != 'vulcan':
        parser.error('The selected atmospheric region must configure chemistry.backend=vulcan')
    # Use the system quantities and temperature routine used by the molecular runner.
    planet_file = config.planet.parameter_file or str(Path(config.paths.planet_parameter_dir)/(config.planet.name+'.json'))
    planet = load_planet(planet_file)
    gravity = 6.67430e-8*planet.M_pl/planet.R_pl**2
    pressures = create_log_pressure_grid(region.pressure_min_bar, region.pressure_max_bar, region.pressure_grid_size)
    temperatures = calculate_temperature_structure(
        planet.t_int_K, pressures, gravity, region.isothermal, region.isothermal_temperature_K,
        region.equilibrium_temperature_K, region.two_point, region.two_point_pressures_bar,
        region.two_point_temperatures_K, region.kappa_ir, region.gamma_guillot, None)
    physical = RadiativeTransferInputs(config.observation.event_type, planet.M_pl, planet.R_pl,
        planet.R_star, planet.stellar_teff_K, 1., planet.t_int_K, None, False)
    profile = run_vulcan(region.chemistry.vulcan, pressures, temperatures, region.species, physical, config.paths,
                         reference_pressure_bar=region.reference_pressure_bar)
    print('Converged VULCAN gas profile:', profile.provenance_file.parent/'gas_profile.npz')
    print('Calculation record:', profile.provenance_file)
    print('Pressure-radius conversion:', profile.radius_alignment_file)
    print('Gas mean molecular weight range [amu]:', profile.mean_molecular_weight.min(), profile.mean_molecular_weight.max())
    print('Maximum condensate/gas mass ratio:', profile.condensate_to_gas_mass_ratio.max())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
