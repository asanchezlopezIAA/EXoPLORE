# Stellar spectra for RM and CLV

The transit calculation needs the spectrum emerging from each part of the
stellar disk. A local intensity spectrum, `I(wavelength, mu)`, supplies that
information. Here `mu=cos(theta)`, with `mu=1` at disk centre and smaller values
towards the limb. Keep the relative brightness between angles when preparing
these inputs. It determines how much light the planet blocks.

The [RM/CLV configuration](config_reference.md#19-stellar-rm-and-clv)
selects the model file and the calculation. The existing PHOENIX flux input
continues to supply the stellar spectrum for simulations with RM/CLV disabled.

## Obtaining local intensities

A practical synthesis route is **MARCS + Turbospectrum**, used by
[Dethier & Bourrier (2023), section 2.1.1](https://arxiv.org/pdf/2304.13759).
MARCS describes the stellar atmospheric structure. Turbospectrum computes
the emerging radiation using that structure, chemical abundances and line
data. NLTE calculations also need a model atom and its matching departure
coefficients. These coefficients describe departures from LTE level populations.

Start with a narrow wavelength interval around the lines of interest. Supply
the host star's temperature, surface gravity, metallicity and abundance
assumptions. For a 1D synthesis, record the adopted microturbulence as well.
The stellar sodium abundance and the planetary sodium abundance belong to
separate atmospheres and are separate inputs.

Sources and instructions:

- [MARCS models and downloads](https://marcs.astro.uu.se/)
- [Turbospectrum source and documentation](https://github.com/bertrandplez/Turbospectrum_NLTE)
- [Public departure coefficients and model atoms](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/)
- [VALD atomic line database](https://vald.astro.uu.se/)

Turbospectrum is an optional external synthesis program. It needs a Fortran
compiler. EXoPLORE can read a prepared intensity bundle without installing
Turbospectrum or adding its tools to the scientific Python environment.

Use the stellar synthesis in its intrinsic spectral form for spatial coupling.
EXoPLORE applies projected stellar rotation and the instrument response during
the transit calculation. The wavelength interval needs margins for every
velocity shift and the convolution kernel.

## Exporting the continuum and angular spectra

For a plane-parallel Turbospectrum calculation, select `Intensity` output and
provide a file of output angles through `MU-POINTS`. The relevant input entries
are shown below. The complete opacity and synthesis inputs are described in
the [Turbospectrum manual](https://github.com/bertrandplez/Turbospectrum_NLTE/blob/master/DOC/Turbospectrum_v20_Documentation_v6.pdf).

```text
'INTENSITY/FLUX:' 'Intensity'
'MU-POINTS:' 'mu.dat'
'SPHERICAL:' 'F'
```

The first line of `mu.dat` gives the number of angles. Subsequent values give
their cosines. Include disk centre and enough small positive values to cover
the limb rays used by EXoPLORE. Sampling more densely near the limb is useful.
The spatial integrator checks every sampled angle and reports any missing
coverage. It never extrapolates local intensities beyond the input grid.
Ingress, egress and finer quadratures can require smaller `mu` values than
the full-disk flux calculation alone. Check coverage again after changing
the spatial or exposure sampling.

The documented v20 ASCII export starts with `# mu-points`. Each wavelength
row contains three flux columns followed by an absolute intensity and a
continuum-normalised intensity for every angle. EXoPLORE reads the angles
from this header.

For accurate continuum values, make a second intensity export using the
same atmosphere, chemical composition, wavelength grid and angles, with
`NFILES: 0`. This removes the spectral line lists while retaining the continuous
opacities. The resulting absolute intensity supplies the local continuum.
Alternatively the reader can recover it as `I/(I/I_cont)`, which inherits the
rounding of the normalised output columns. A printed zero requires a separate
continuum export.

## Preparing the EXoPLORE bundle

Run the converter from the project environment:

```bash
python scripts/prepare_stellar_intensities.py stellar_lines.spec stellar_local.npz \
    --continuum stellar_continuum.spec \
    --input-medium air \
    --output-medium vacuum \
    --source "Turbospectrum version, atmosphere parameters, abundances and line data"
```

Set `--input-medium` from the synthesis and line-data convention. The converter
does not infer it from the file name. Air-to-vacuum conversion changes both
the wavelength coordinate and the per-Angstrom intensity using the coordinate
Jacobian. Intensities and continuum keep the same physical normalization.

Turbospectrum limits the number of angles in one output. To combine several
exports from the same synthesis, repeat the additional-input options:

```text
--additional-intensities stellar_more_angles.spec
--additional-continuum stellar_more_angles_continuum.spec
```

The converter checks the shared wavelength grid and agreement at overlapping
angles. It writes the arrays, units, source description and input-file hashes.
Existing output files are protected. Choose a new filename for a revised model.

## Selecting the model in the simulation JSON

```json
"rm_clv": {
  "simulate": true,
  "correct": false,
  "effects": "rm_clv",
  "coupling": "spatial",
  "model_file": "stellar_local.npz",
  "wavelength_medium": "vacuum",
  "projected_obliquity_deg": 0.0,
  "stellar_radial_points": 48,
  "planet_radial_points": 24,
  "azimuth_points": 128,
  "exposure_subsamples": 3
}
```

This example uses an aligned projected orbit. Set the obliquity for your
system and the stellar `v_rotsini_kms` in its planet-parameter JSON. The orbital
inclination in that file determines the transit chord across the disk.
For the circular orbit, both active RM+CLV modes use the observer-frame
stellar velocity
{math}`v_\star=V_{\rm sys}-V_{\rm BERV}-K_\star\sin(2\pi\phi)`,
with positive radial velocity indicating recession. The planetary orbital
term has the opposite sign, {math}`+K_p\sin(2\pi\phi)`.
Increase the quadrature and temporal sampling to check convergence for your
star, planet size and exposure time. The current spatial route supports a
circular transit with a spherical 1D planetary atmosphere and pRT 3.2.0.

The spatial calculation evaluates local stellar spectra along every illuminated
atmospheric ray. Runtime increases with wavelength coverage, pressure layers,
angular quadrature and exposure samples. A compiled Numba loop interpolates
the intensities and accumulates each ray's angular weight. It retains the
specified quadrature and all native atmospheric radii. The first invocation
can include compilation time. Benchmark a short wavelength interval before
planning a full-order calculation.

## Choosing a stellar model for scientific interpretation

An active RM+CLV calculation supplies its disk-integrated stellar reference
from the local-intensity bundle. With both `simulate` and `correct` disabled,
the standard runner uses its PHOENIX stellar inputs. Controlled comparisons
should track this source choice and include a matched stellar reference.

Sodium line profiles can change appreciably between LTE and NLTE and between
1D and 3D stellar atmospheres. Compare the disk-integrated model with available
out-of-transit spectra before interpreting a correction as an accurate model of
a particular star. Activity and convective line shifts can require additional
stellar modelling.

With spatial coupling, the atmospheric absorption is weighted by the local
stellar spectrum behind the annulus. With `rm_clv.correct: true`, Seidel22 processes the opaque-body stellar model
with the same alignment, continuum windows, reference exposures and weights
as the observation. It divides the individual stellar-frame residuals by
that prepared model before planetary alignment. Other pipelines divide the
native flux and uncertainties by the photospheric distortion. The atmospheric term retains
that local illumination, which can change its apparent line shape. Even a
grey absorbing annulus can leave stellar-line structure when the local line
profiles vary across the disk. Compare a fully coupled forward model with
the data when these effects are important for an atomic-line measurement.

For a narrow atomic-line window, inspect the columns retained by preparation.
Deep stellar cores can be included in masks. Validate the recovery of a known
injected velocity track through the selected preparation and CCF analysis
before interpreting a map maximum as an orbital measurement.

[Canocchi et al. (2024)](https://arxiv.org/abs/2410.15810) provide a
[3D LTE/NLTE intensity grid](https://zenodo.org/records/14055808) with continuum
spectra. The public FITS contains 12 angles spanning `mu=0.1` to `1`.
Further limb coverage is needed for our full-disk integration. Its intensity
units are per Hz, so conversion to per wavelength is also required.
Check the synthesis resolution treatment before using it for spatial coupling.
The converter above reads Turbospectrum's plane-parallel ASCII layout.

The archive records should distinguish spectral sampling from an already
applied instrumental convolution. `resolving_power: 0` in an EXoPLORE bundle
declares an unconvolved synthesis. A fine wavelength step by itself does not
establish the instrumental resolution.
