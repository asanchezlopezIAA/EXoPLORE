# Stellar components and surface regions

A cool region changes the spectrum illuminating the planetary atmosphere and
the spectrum blocked by the opaque planet. Supply a local intensity grid for
each stellar component, then place those components on the rotating surface.
The same RM+CLV geometry follows the planet across them.

A **component** describes the emitted specific intensity, `I(wavelength, mu)`,
and its continuum. A **region** selects that component at a location on the
star. Names such as `quiet`, `spot`, `facula` or `plage` are labels chosen in
the JSON. Their line profiles and brightness come from the supplied spectra.

## Try a small example

From the repository root, run:

```bash
PYTHONPATH=src python scripts/example_stellar_components.py --output outputs/stellar_surface_example
```

The script reads `configs/stellar_surface_example.json`, creates three analytic
intensity files and calculates an opaque transit through the heterogeneous
star. It writes `surface_components.png`, `surface_spectra.npz`, the resolved
configuration and `stellar_surface_provenance.json`. This example exercises
the stellar integrator directly and takes a few seconds. The analytic spectra
make brightness and geometry easy to inspect. Use physical intensity grids
for an atmospheric observation.

For a complete observation, add the `stellar_surface` block to your simulation
JSON and enable `rm_clv.simulate`. Keep the instrument, atmosphere, noise and
preparation settings appropriate to that observation. Tutorial 13 describes
the {ref}`spatial sodium observation <tutorial-13-sodium>`.

The calculation follows this sequence:

```text
component intensity files + region geometry + exposure times
  -> component at each visible stellar sample
  -> local mu interpolation and rotation/component Doppler shift
  -> opaque occultation and atmospheric-ray absorption
  -> disk integration and exposure averaging
  -> instrument response and existing observation/preparation
```

## Configuration

```json
{
  "stellar_surface": {
    "default_component": "quiet",
    "components": {
      "quiet": {"intensity_model_file": "inputs/star_quiet.npz"},
      "cool_region": {
        "intensity_model_file": "inputs/star_cool.npz",
        "velocity_offset_kms": 0.0,
        "metadata": {"source_model": "record the supplied synthesis here"}
      },
      "facular_region": {"intensity_model_file": "inputs/star_facula.npz"}
    },
    "regions": [
      {"component": "cool_region", "geometry": "projected",
       "x": -0.2, "y": -0.65, "radius": 0.22, "priority": 0},
      {"component": "facular_region", "geometry": "projected",
       "x": 0.4, "y": 0.1, "radius": 0.18, "priority": 1}
    ],
    "correction_surface": "matched"
  }
}
```

Paths are relative to the execution directory, following the other simulation
inputs. `rm_clv.model_file` remains the fallback model used by the existing
quiet-star route. Set it to the quiet grid in an activity experiment.

Omit `stellar_surface`, set it to `null`, or use `regions: []` to run the
existing quiet-star calculation. Empty regions leave the component files
unused. A nonempty region list loads every declared component. The default
component fills the rest of the visible surface. Highest integer `priority`
wins an overlap, with the later JSON entry winning a tie.

`correction_surface: "matched"` uses the injected surface for the opaque-body
stellar correction. `"quiet"` generates with the heterogeneous surface and
uses the default component over the whole star for correction. Pair this with
`rm_clv.correction_method: "matched"`. The atmospheric model stays the same
in both cases, while the correction uses opaque-body stellar flux.

Set `observation.simulate_planet: false` for a stellar-only contamination
experiment. The opaque planet still transits the star. With the flag true,
the existing atmospheric calculation supplies its transmission. Active
surfaces require `scale_injection: 1` and `signal_uses_light_curve: true`.

## Local intensity files

Each component uses the [existing numeric NPZ format](stellar_intensities.md).
Load files with `numpy.load(..., allow_pickle=False)`.

| Field | Shape / meaning |
|---|---|
| `wavelength_um` | `[n_lambda]`, positive increasing wavelengths in micrometres |
| `mu` | `[n_mu]`, strictly increasing `cos(theta)` between 0 and 1, at least two samples |
| `intensity` | `[n_mu, n_lambda]`, finite nonnegative specific intensities |
| `continuum` | Same shape, finite positive local continua, required for components |
| `intensity_kind` | Scalar string `specific_intensity` |
| `intensity_unit` | Scalar string, the same physical unit and calibration for every component |
| `wavelength_medium` | Scalar `air` or `vacuum`, matching `rm_clv.wavelength_medium` |
| `source` | Scalar nonempty description of the source synthesis |
| `resolving_power` | Scalar zero for intrinsic spectra, or prior resolving power |
| `metadata_json`, `provenance_json` | Optional scalar JSON objects, preserved in provenance |

Keep the absolute relative brightness between components and viewing angles.
For example, spectra in `erg s^-1 cm^-2 Angstrom^-1 sr^-1` must share that
unit, wavelength-density convention and calibration. Independently dividing
each component by its own continuum would erase the brightness contrast.
EXoPLORE checks matching unit strings. The supplied calibration remains the
input provider's responsibility.

Components may have different native wavelength and mu grids. Linear
interpolation evaluates each grid at the required local viewing angles and
Doppler-shifted wavelengths. The fine integration grid retains the smallest
native log-wavelength interval among components. Coverage failures raise an
error. Angular and wavelength extrapolation are disabled. Keep enough
wavelength margin for rotation, component velocities, observer motion and
instrument convolution. Input ordering is validated and retained.

Spatial coupling requires intrinsic spectra (`resolving_power: 0`). All
components share the same prior resolving power and intensity-unit string.
A facular component may have its own wavelength-dependent and mu-dependent
contrast. Its spectrum can come from any provider implementing this contract.

Raw integration retains the supplied intensity scale. Observer products
follow the existing continuum-relative convention. They divide the star by
its exposure's unocculted continuum, so broadband rotational brightness
changes should be inspected in the intrinsic flux and continuum arrays.

## Projected circles and rotating spherical caps

Projected regions use `geometry: "projected"`, with `x`, `y` and `radius` in
stellar radii. Their positions stay fixed in the sky frame. The planet moves
along the existing transit chord, and stellar line Doppler shifts use the
existing projected rotation field.

Physical regions use a spherical cap:

```json
{
  "component": "cool_region",
  "geometry": "stellar",
  "latitude_deg": 20.0,
  "longitude_deg": -35.0,
  "angular_radius_deg": 12.0,
  "priority": 10
}
```

These regions also require explicit fields in `stellar_surface`:

| Field | Meaning |
|---|---|
| `stellar_inclination_deg` | Spin-axis inclination to the observer, 0 to 180 degrees |
| `rotation_period_days` | Positive solid-body rotation period |
| `reference_bjd` | Rotational reference epoch, using the exposure BJD_TDB time convention |
| `reference_phase_deg` | Common longitude offset at that epoch, default zero |
| `rotation_consistency_rtol` | Relative tolerance linking period, inclination, stellar radius and catalog `v_rotsini_kms`, default 0.01 |

Longitude zero lies on the visible central meridian. Latitude is measured
from the stellar equator towards the spin pole. Increasing longitude moves
an equatorial region towards the receding rotation side. The projected pole
uses `rm_clv.projected_obliquity_deg`. Exposure midpoint BJDs and each temporal
subsample rotate the regions by `360*(BJD-reference_bjd)/rotation_period_days`.
Far-side regions contribute through their visible cap intersections only.
A cap crossing the limb is clipped by the stellar disk.

The model checks `v sin(i) = 2*pi*R_star*sin(i)/P_rotation`. Inconsistent inputs
raise an error showing that the rotation parameters need attention. Calls to
`evaluate()` with physical regions require `times_bjd`, one midpoint per
exposure. The simulator supplies the actual observing times, including
separate nights.

## Local velocities

A component defaults to zero additional velocity. Set `velocity_offset_kms`
for a constant shift, positive towards the red. Alternatively set
`velocity_model_file` to a numeric NPZ with increasing `mu` and a matching
finite `velocity_kms` vector. The table must cover every sampled viewing
angle. Linear interpolation supplies the local component velocity.

The integrator adds this term to the existing projected stellar rotation.
This can represent a supplied convective-shift prescription. Line-dependent
asymmetries or magnetic line profiles can be present in the intensity grid
itself. Record their velocity convention so a shift is applied once.

## RM+CLV and atmospheric illumination

In factorized mode, the selected local spectra are Doppler shifted, weighted
by projected cell area and occulted on the existing Cartesian disk. Exposure
fluxes are averaged before instrumental convolution and normalization. Active
surfaces retain their opaque continuum transit. The runner replaces the
catalog-radius opaque term in its existing factorized planet matrix with
that surface continuum term. Atmospheric spectral structure keeps the
existing factorized time weighting.

Spatial mode selects the component behind every illuminated star-centred or
planet-centred ray. The opaque body blocks that component's spectrum. The
atmospheric annulus absorbs its local light using the existing pRT
transmissions and native impact parameters. Crossing a cool or facular
region therefore changes both the occulted flux and the spectrum entering
the atmosphere. Angular quadrature splits at region boundaries. The opaque
and full-disk radial rules also split near projected region edges.

Check convergence with `cells_across_diameter` in factorized mode and the
radial/angular quadrature settings in spatial mode. Tiny regions need enough
surface samples. Atmospheric radial sampling continues to follow pRT.

## Provenance and future inputs

`stellar_surface_provenance.json` records component names, filenames, SHA256
hashes, supplied metadata, local velocities, regions and rotation parameters.
The resolved configuration retains the same controls. Optional metadata may
include Teff, logg, metallicity, dimensionality, LTE/NLTE, magnetic field,
simulation identifier and citations.

A future 3D/NLTE provider can export its angle-dependent emergent intensities
and continuum into this NPZ contract, then reference the file as another
component. Gloria Canocchi's grids could be supplied this way when available.
No collaborator's models are bundled by this example. The present contract
uses azimuth-averaged specific intensities. A spatially resolved 3D synthesis
needs an adapter producing the component's `I(wavelength, mu)` arrays.

For chromospheric tracers such as Halpha, He I or Ca II, supply local spectra
whose atmospheric calculation includes the relevant line-formation physics.
The surface layer transports those inputs. The current implementation uses
solid-body rotation, fixed region shapes and time-independent component
spectra. Differential rotation, region evolution, arbitrary maps and fitting
surface parameters remain future additions behind the component-selection
interface.

## Computational cost

Intensity grids are loaded once when constructing the stellar model. Static
Cartesian maps reuse component labels and local spectra across exposures.
Spatial maps reuse their ray geometry across wavelength chunks and accumulate
component spectra with the existing compiled interpolation kernel. A rotating
map updates the visible component selection at every exposure subsample.
Its full-disk reference also changes with time.

Memory for input spectra scales with the sum of component grid sizes.
Wavelength chunks bound integration work arrays. Boundary splitting adds
quadrature intervals around sharp regions, so cost depends on region count
and geometry. A quiet correction for a heterogeneous injection calculates
an additional stellar model. Empty region lists retain the original route.
