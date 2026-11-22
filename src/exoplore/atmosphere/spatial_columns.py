"""Build leading and trailing pRT ray tables from the existing limb regions."""
import numpy as np

from .prt import call_pRT
from .transmission import TwoColumnTransmission, calculate_chord_transmission


def build_spatial_columns(model_inputs, atmosphere, pressures, wave_ins, planet, paths):
    """Evaluate each chemistry column once per order and export intrinsic rays.

    VULCAN keeps its existing physical-column cache across orders. Each pRT
    object owns its opacity state until the matching ray table is exported.
    The half-annulus spectra returned alongside the tables are diagnostics.
    The stellar integrator uses the unbroadened transmission tables.
    """
    from petitRADTRANS.radtrans import Radtrans
    from petitRADTRANS import physical_constants as cst
    regions = (atmosphere.morning_day,atmosphere.evening_day)
    if regions[0].reference_pressure_bar != regions[1].reference_pressure_bar:
        raise ValueError('Both spatial columns require the same radius/reference-pressure anchor')
    columns = []
    for region, pressure in zip(regions,pressures):
        if region.species[:2] != ['H2','He']:
            raise ValueError('Spatial molecular columns require H2 and He as their first two species')
        rt = Radtrans(pressures=pressure,line_species=region.species[2:],
            rayleigh_species=['H2','He'],gas_continuum_contributors=['H2--H2','H2--He'],
            wavelength_boundaries=[wave_ins.min()-.01,wave_ins.max()+.01],line_opacity_mode='lbl')
        _, _, mass, mmw, _, temperature = call_pRT(
            model_inputs,pressure,rt,region.species,region.vmr,
            region.mean_molecular_weight,region.reference_pressure_bar,
            region.isothermal,region.isothermal_temperature_K,region.two_point,
            region.two_point_pressures_bar,region.two_point_temperatures_K,
            region.kappa_ir,region.gamma_guillot,region.equilibrium_temperature_K,
            region.metallicity_wrt_solar,region.carbon_to_oxygen_ratio,
            chemistry=region.chemistry,chemistry_paths=paths,use_easyCHEM=region.use_easychem,
            P_cloud=region.cloud_pressure_bar,easychem_CtoO_ret=region.use_easychem)
        columns.append(calculate_chord_transmission(rt,temperatures=temperature,
            mass_fractions=mass,mean_molar_masses=mmw,
            reference_gravity=cst.G*planet.M_pl/planet.R_pl**2,
            reference_pressure=region.reference_pressure_bar,planet_radius=planet.R_pl,
            opaque_cloud_top_pressure=region.cloud_pressure_bar))
    omega = (2*np.pi/(planet.orbital_period_days*86400)
             if atmosphere.spatial_planet_rotation == 'synchronous_aligned' else 0.)
    tables = TwoColumnTransmission(*columns,angular_speed_s=omega,
        spin_inclination_rad=planet.inc,morning_wind_kms=regions[0].wind_velocity_kms,
        evening_wind_kms=regions[1].wind_velocity_kms)
    wave = columns[0].wavelength_um
    depths = [np.interp(wave,c.wavelength_um,(c.effective_radius_cm()/planet.R_star)**2)
              for c in columns]
    return tables,wave,.5*(depths[0]+depths[1])
