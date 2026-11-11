# Atmospheric chemistry

The atmospheric composition can come from EasyChem equilibrium chemistry,
manual mass fractions or an external VULCAN calculation. Each atmospheric
region has its own selection, so we can inject a disequilibrium atmosphere
and search its spectrum with an equilibrium template.

The pressure grid and temperature profile remain in the surrounding
`atmosphere.planet_model`, `ccf_template` or limb-region block. Chemistry
returns gas mass fractions and gas mean molecular weight on that pressure grid.
petitRADTRANS then calculates the atmospheric spectrum with the selected
opacity tables.

[Tutorial 15](tutorial.md#tutorial-15-equilibrium-and-disequilibrium-chemistry-with-vulcan)
compares gas abundance profiles with equilibrium, vertical mixing and photochemistry.

## Choose the chemistry

```json
"chemistry": {
  "backend": "easychem"
}
```

The choices are `easychem`, `manual` and `vulcan`. EasyChem uses the region's
`metallicity_wrt_solar` and `carbon_to_oxygen_ratio`. Manual chemistry uses
its `mass_fractions` and `mean_molecular_weight`.
For manual limb spectra, the current calculation fills the molecular-weight
profiles from `atmosphere.planet_model.mean_molecular_weight`. Keep the manual
regional values consistent with that value. The explicit EasyChem and manual
selectors use the same limb calculations as `use_easychem`.

Existing configurations using `use_easychem` keep their current behaviour.
When adding `chemistry`, remove the old boolean or give it the matching
value (`true` for EasyChem, `false` for manual or VULCAN). Contradictory
explicit settings raise an error.

## Install VULCAN separately

Use the original [VULCAN repository](https://github.com/shami-EEG/VULCAN).
The adapter uses its `vulcan.py` and `vulcan_cfg.py` script interface. The
FormingWorlds and neoVULCAN forks have different interfaces.
The implementation was inspected against upstream commit
`c870b1a52702defb032abd09202c4f1ea6f809be`.

Keep VULCAN in its own environment. From a directory alongside EXoPLORE:

```bash
git clone https://github.com/shami-EEG/VULCAN.git VULCAN
git -C VULCAN checkout c870b1a52702defb032abd09202c4f1ea6f809be
conda create -n vulcan -c conda-forge python=3.11 numpy=1.26.4 scipy=1.11.1 sympy=1.14 matplotlib=3.8.3 astropy=7.2
conda activate vulcan
cd VULCAN/fastchem_vulcan
make
```

FastChem needs a C++ compiler. Use the serial `make` command supplied above.
The upstream makefile can reach its linking step before compilation finishes
when invoked with parallel jobs.
The package versions above are those used for the local validation.

In the simulation JSON, set `paths.vulcan_directory` to the checkout and
`paths.vulcan_python` to that environment's Python executable. Both paths
must exist. The observing run can still use the usual EXoPLORE environment.
EXoPLORE starts the configured VULCAN Python in a subprocess.

## A thermal column

Copy an existing simulation JSON and edit the atmospheric region. This
example uses a 1500 K H/He/C/N/O column with constant vertical mixing.
The elemental ratios below specify this example's composition. They are
number abundances relative to hydrogen nuclei.

```json
"atmosphere": {
  "limb_asymmetries": false,
  "planet_model": {
    "species": ["H2", "He", "H2O", "CO", "CH4", "NH3", "HCN"],
    "pressure_min_bar": 1e-6,
    "pressure_max_bar": 10.0,
    "pressure_grid_size": 32,
    "isothermal": true,
    "isothermal_temperature_K": 1500.0,
    "chemistry": {
      "backend": "vulcan",
      "vulcan": {
        "network_file": "thermo/NCHO_thermo_network.txt",
        "elemental_abundances": {
          "H": 1.0,
          "He": 0.0838,
          "C": 0.000295,
          "O": 0.000537,
          "N": 0.0000708
        },
        "photochemistry": false,
        "eddy_diffusion": {
          "mode": "constant",
          "value_cm2_s": 1e9
        },
        "molecular_diffusion": true,
        "molecular_diffusion_scheme": "central",
        "top_boundary": {"type": "zero_flux"},
        "bottom_boundary": {"type": "zero_flux"}
      }
    }
  }
}
```

Complete the existing `paths` block with:

```json
"vulcan_directory": "../VULCAN",
"vulcan_python": "environments/vulcan/bin/python",
"vulcan_cache_dir": "outputs/chemistry/vulcan"
```

Replace the example executable path with the Python in the environment you
created. Relative paths follow the working directory from which EXoPLORE is
launched, as for other simulation inputs. A relative `network_file` is
resolved inside the VULCAN checkout.

VULCAN uses the explicit `elemental_abundances` in this block. The region's
EasyChem metallicity and C/O fields apply to the EasyChem backend. To compare
equilibrium and kinetics, first match the elemental abundance prescriptions.
The two packages' solar reference compositions can differ.

We can prepare the chemistry before running an observation:

```bash
python scripts/prepare_vulcan_chemistry.py configs/my_simulation.json
```

Use `--region ccf_template`, `--region morning_day` or another named region
to prepare that column. The same simulation JSON supplies its inputs.
The ordinary observing command also prepares the chemistry automatically:

```bash
python scripts/run_exoplore.py configs/my_simulation.json --run
```

The spectrum calculation loads the converged profile for each selected order.
Changes to physical inputs, chemistry settings, external files, executable
or library versions produce a new cache entry.

## Different chemistry in the two limbs

The master example is `configs/hd189733b_andes_transit_clean.json`. Its
`chemistry: null` entries retain the existing selections. For a VULCAN
experiment, copy that simulation file and set `limb_divisions: "gradual"`.
This mode uses `morning_day` and `evening_day`. Replace those two region
blocks with the following settings and set the VULCAN installation paths
as above. The example holds the elemental composition and mixing fixed
while changing the limb temperature.

```json
"limb_asymmetries": true,
"limb_divisions": "gradual",
"morning_day": {
  "species": ["H2", "He", "H2O", "CO", "CH4"],
  "pressure_min_bar": 1e-6,
  "pressure_max_bar": 100.0,
  "pressure_grid_size": 32,
  "reference_pressure_bar": 0.01,
  "isothermal": true,
  "isothermal_temperature_K": 1200.0,
  "chemistry": {
    "backend": "vulcan",
    "vulcan": {
      "network_file": "thermo/NCHO_thermo_network.txt",
      "elemental_abundances": {"H": 1.0, "He": 0.0838, "C": 0.000295, "O": 0.000537, "N": 0.0000708},
      "eddy_diffusion": {"mode": "constant", "value_cm2_s": 1e9},
      "molecular_diffusion": true
    }
  }
},
"evening_day": {
  "species": ["H2", "He", "H2O", "CO", "CH4"],
  "pressure_min_bar": 1e-6,
  "pressure_max_bar": 100.0,
  "pressure_grid_size": 32,
  "reference_pressure_bar": 0.01,
  "isothermal": true,
  "isothermal_temperature_K": 1500.0,
  "chemistry": {
    "backend": "vulcan",
    "vulcan": {
      "network_file": "thermo/NCHO_thermo_network.txt",
      "elemental_abundances": {"H": 1.0, "He": 0.0838, "C": 0.000295, "O": 0.000537, "N": 0.0000708},
      "eddy_diffusion": {"mode": "constant", "value_cm2_s": 1e9},
      "molecular_diffusion": true
    }
  }
}
```

The four-region `quarters` mode also accepts independent chemistry in
`morning_night` and `evening_night`. Each column has its own pressure,
temperature, chemistry and radius-conversion record. Choose the reaction
network for the elements you supply and the species you want to model.
The NCHO example above supplies H, He, C, N and O. The wider default
Tutorial 1 species list also contains elements absent from that network.

To prepare the two columns before calculating spectra:

```bash
python scripts/prepare_vulcan_chemistry.py configs/my_simulation.json --region morning_day
python scripts/prepare_vulcan_chemistry.py configs/my_simulation.json --region evening_day
```

## Photochemistry and transport

For photochemistry, choose a network with photolysis reactions and add:

```json
"network_file": "thermo/NCHO_photo_network.txt",
"photochemistry": true,
"stellar_flux_file": "../VULCAN/atm/stellar_flux/sflux-HD189_Moses11.txt",
"stellar_flux_reference": "stellar_surface",
"orbit_radius_au": 0.03142,
"zenith_angle_deg": 48.0,
"diurnal_factor": 1.0
```

These are HD 189733 example inputs supplied by VULCAN. The stellar file has
one commented header followed by wavelength in nm and stellar-surface energy
flux in erg cm⁻² s⁻¹ nm⁻¹. The adapter requires coverage from 2 to 700 nm,
which spans the original solver's photochemical bin limits. VULCAN applies
orbital dilution using the host-star radius and the explicitly supplied
`orbit_radius_au`. The photochemical zenith angle describes irradiation of
the atmospheric column, separately from the observing geometry.

A visible or infrared photospheric spectrum alone does not constrain the
UV flux needed for photolysis. Supply a suitable UV spectrum for the host
star and record its source.

| Field inside `chemistry.vulcan` | Meaning |
|---|---|
| `initialization` | `equilibrium` (default) starts with native FastChem. `constant_mixing_ratios` starts with the supplied `initial_mixing_ratios` gas mole fractions, whose sum must be one. Both initial states are checked before integration. |
| `initial_mixing_ratios` | Species-to-mole-fraction object for constant initialisation. Its elemental ratios must match `elemental_abundances`. |
| `initial_element_tolerance` | Maximum relative error in each initial element/H ratio. Default 0.001. Every network species enters this check. |
| `eddy_diffusion` | `mode: "constant"` with `value_cm2_s`, `"file"` with `profile_file`, or `"off"`. A file has a commented header and ascending pressure [bar], Kzz [cm²/s] columns spanning the atmosphere. |
| `molecular_diffusion` | Enable the native molecular and associated thermal-diffusion treatment. |
| `molecular_diffusion_scheme` | `central`, `upwind` or `hybrid`. These select the upstream numerical schemes. The generated configuration records the choice. |
| `vertical_velocity_cm_s` | Constant vertical advection velocity, positive upwards in VULCAN. Default zero. |
| `background_gas` | Background used for diffusion and settling: `H2`, `N2`, `O2` or `CO2`. Default `H2`. |
| `top_boundary`, `bottom_boundary` | `type: "zero_flux"` or `"flux_file"` with `file`. A flux file follows VULCAN's native species/flux/deposition convention. The bottom also accepts `"fixed_mixing_ratios"` with a `mixing_ratios` object. |
| `species_map` | Explicit pRT-opacity-name to VULCAN-gas-name mapping when the usual molecular aliases are insufficient. Missing requested species raise an error. |
| `timeout_seconds` | Wall-clock limit for one solver subprocess. Default 7200 s. |
| `max_atom_loss` | Maximum absolute fractional elemental-conservation diagnostic accepted from VULCAN. Default 0.01. Boundary sources and sinks require a physically appropriate interpretation of that diagnostic. |
| `solver_options` | Numerical integration settings. Accepted keys are listed below. |

`solver_options` accepts `dttry`, `trun_min`, `runtime`, `dt_min`, `dt_max`,
`dt_var_max`, `dt_var_min`, `count_min`, `count_max`, `atol`, `mtol`,
`mtol_conv`, `loss_eps`, `rtol`, `yconv_cri`, `slope_cri`, `yconv_min`,
`flux_cri`, `flux_atol`, `conv_step`, `use_adapt_rtol` and `update_frq`.
`use_adapt_rtol` is a boolean. `update_frq` is a positive integer giving the
number of steps between native hydrostatic updates. The native adaptive
controller can change `rtol` during integration. Set `use_adapt_rtol: false`
when testing a fixed tolerance. Their definitions follow the
[VULCAN configuration guide](https://github.com/shami-EEG/VULCAN/blob/master/vulcan_cfg_README.txt).
The remaining native numerical settings come from the recorded checkout
configuration. The adapter disables the native hot-bottom pressure cut so
the supplied atmospheric column is kept in full.

The initial-state check verifies finite, non-negative number densities and
the configured elemental ratios in every layer. A failed equilibrium solution
can contain finite numbers with incorrect abundances. Such a column stops
before time integration and records the failing layer in
`initialisation_checks.json`. Inspect the native FastChem diagnostics before
choosing another initialisation. A constant mixture is an initial condition
for the kinetics calculation and must also reach the accepted steady state.

## Condensation

The optional `condensation` block enables VULCAN's gas depletion and particle
settling. Supply `enabled`, the condensing `gas_species`, `condensed_species`,
`particle_radius_cm` and `particle_density_g_cm3` mappings keyed by condensate
name. `settling` selects the native settling treatment. The reaction network
must contain the corresponding condensation reactions.
The direct growth rates use VULCAN's molecular diffusion coefficients, so
this route requires `molecular_diffusion: true`.
The generated configuration uses native direct growth with continuous
condensation. Its initial cold trap, relaxation and frozen-species options
remain disabled.

VULCAN normalises its gas and condensate abundances by gas number density.
Its hydrostatic mean mass includes condensed material:

```{math}
\mu_\mathrm{hydro} = \mu_\mathrm{gas}
\left(1 + \rho_\mathrm{cond}/\rho_\mathrm{gas}\right).
```

This quantity describes the native VULCAN pressure-height structure. EXoPLORE
supplies pRT with gas-phase mean molecular weight and mass fractions relative
to the complete gas mixture. Every gas enters the denominator, including gases
without selected opacity tables. Condensation changes these quantities through
the remaining gas abundances.

The returned profile also retains `condensate_to_gas_mass_ratio` and
`native_hydrostatic_mean_mass`. The native calculation and pRT can therefore
have different pressure-radius structures when condensed mass is appreciable.
The radius-alignment record gives the maximum loading and the difference
between the gas-only and loaded radius conversions at the native reference
pressure. Both solvers retain the configured radius at the region's reference
pressure. The gas-only pRT approximation still needs assessment for each
condensation case.

Condensation remains experimental in this adapter. Controlled cold-column
tests have failed convergence or elemental-budget checks, including a case
which reached the native steady-state criterion. Check the numerical timestep
and the gas-plus-condensate budget for each proposed science case.

The inspected upstream Jupiter
example fails during FastChem initialisation below 200 K in the local test
installation. The initial-state check rejects its invalid abundances before
they propagate into hydrostatic and photochemical quantities.

Two ammonia corrections are applied to the private working copy when
`NH3_l_s` is requested. The inspected native table assigns it 16.023 amu,
while gaseous `NH3` has 17.031 amu. The adapter sets the formula-unit mass to
17.031 amu. The direct condensation dispatcher is also changed to match the
network's `NH3 -> NH3_l_s` reaction name. `source_corrections.json` and the
manifest record the changes and file hashes. A checkout containing those
corrections already passes through unchanged. Other inconsistent ammonia
masses raise an error. The original external checkout remains unchanged.

The adapter sets `use_sat_surfaceH2O=False` in the generated native configuration.
The master JSON's boundary block therefore controls the lower boundary. The
native ocean/surface-water shortcut would otherwise replace the configured
water abundance and introduce a fixed lower-boundary mixing ratio.

A successful condensation/settling science case still requires convergence and
an elemental budget including gas, condensed material and boundary fluxes.
VULCAN's `atom_loss` diagnostic alone does not establish the budget for an open
column with influx, deposition or rainout.

The native output retains condensate abundances. The pRT interface calculates
gas composition and condensed mass per gas particle from the native number
densities.
Condensate extinction needs optical constants and a particle-opacity model.
That connection is outside this gas-chemistry adapter. The existing opaque
cloud-pressure setting remains a separate atmospheric choice.

## Inspect the result

Each cache entry contains:

- `vulcan_cfg.py` and `exoplore_inputs/`, the generated native inputs.
- `run.log` and `output/column.vul`, the solver log and native output.
- `initialisation_checks.json`, the initial elemental-abundance checks.
- `source_corrections.json`, any applied ammonia corrections and file hashes.
- `gas_profile.npz`, the pressure [bar], temperature [K], gas species,
  mole fractions `[n_layers, n_gas_species]` and molecular masses [amu].
- `manifest.json`, the input and source hashes, source revision, environment,
  settings, runtime, termination status and elemental-conservation diagnostics.

A successful subprocess must also report VULCAN's steady-state termination
status. The adapter rejects invalid abundances, missing opacity species,
insufficient pressure coverage, temperature mismatches and conservation
failures. Failed workspaces retain their logs for inspection. A subsequent
attempt starts in a fresh workspace.

All gases enter the mean molecular weight, including gases with no selected
opacity tables. Composition is interpolated in log pressure and converted
from mole fractions to mass fractions after interpolation. The selected
opacity species keep their fractions of the complete gas mixture.

## Radius and pressure references

For giant columns with a bottom pressure of at least 1 bar, VULCAN anchors
its altitude grid at the pressure interface nearest 1 bar. For shallower
columns, it uses the bottom interface. pRT anchors the planet radius at
`reference_pressure_bar`, often 0.01 bar in the EXoPLORE examples.
EXoPLORE converts the configured radius to VULCAN's actual reference interface
using pRT's hydrostatic equation. The planetary mass stays fixed. Each accepted
VULCAN column supplies its native hydrostatic mean-mass profile for the next
conversion. The gas-only conversion is recorded alongside it for comparison.
The iteration stops when the radius adjustment is at most 1e-8 of the configured
planetary radius, with a maximum of eight chemistry columns.

The hydrostatic integral uses an auxiliary refined grid. Chemistry and opacity
sampling retain their configured values. Both reference pressures must lie
inside the supplied pressure column. A shallow column whose native bottom
interface falls outside that coverage raises an error. Extend its pressure
range before using this route.

The final cache directory contains a `radius_alignment_*.json` record with both
pressure-radius references, the fixed mass, reference gravity and the accepted
iteration history. Native finite-layer integration can still leave a small
geometrical discrepancy. Check pressure-grid convergence for the science case.

## Current observing routes

Explicit chemistry is connected to the molecular forward model, its CCF
atmosphere and the supported limb columns. The observing route retains its
H₂/He Rayleigh and CIA setup and requires these two gases first in the species
list, with `background_gas: "H2"` for VULCAN. Spatial RM+CLV uses the same pRT
composition for the atmospheric chords.

The direct sodium and helium runners retain their own atmosphere models and
reject this chemistry block. Existing one-dimensional equilibrium or
free-abundance retrievals can analyse observations generated with a VULCAN
column. Explicit chemistry in sampled retrieval atmospheres and regional
retrievals requires separate validation and currently raises an error.
Check the temperature validity of the selected reaction network before applying it to a new target.
VULCAN columns describe one-dimensional chemistry and vertical transport.
Independent limb columns share no horizontal chemical exchange.

## References

- Tsai et al. (2017), *VULCAN: an Open-source, Validated Chemical Kinetics
  Python Code for Exoplanetary Atmospheres*, [paper](https://arxiv.org/abs/1607.00409).
- Tsai et al. (2021), *A Comparative Study of Atmospheric Chemistry with
  VULCAN*, [doi:10.3847/1538-4357/ac29bc](https://doi.org/10.3847/1538-4357/ac29bc).
