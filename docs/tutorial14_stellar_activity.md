(tutorial-14-stellar-activity)=
# Tutorial 14: Stellar spots and faculae in Na D2 transmission

During transit, the planet can cross a region whose spectrum differs from the
rest of the star. Regions outside the transit chord also change the stellar
reference against which we measure atmospheric absorption. [Cauley et al.
(2018)](https://doi.org/10.3847/1538-3881/aaddf9) examine this contrast effect in
high-resolution transmission spectra, including sodium. We now add these
surface components to the HD 189733 b observation of {ref}`Tutorial 13
<tutorial-13-sodium>` and examine the recovered Na D2 line.

The planetary atmosphere is identical throughout. For each heterogeneous
observation, we first apply the matching stellar correction. We then prepare
the same observation assuming a quiet star. The difference between these two
profiles comes from the stellar model used in the analysis.

## Step 1: Keep the Tutorial 13 observation

Open `configs/hd189733b_andes_stellar_activity.json`. It keeps the ANDES UBV
order, the full observing sequence and the atmospheric model of Tutorial 13.

| Component | Configuration |
|---|---|
| Instrument | ANDES UBV order 56, resolving power 100,000 |
| Observation | 388 exposures, 30 s integrations and 6 s readout |
| Integration | Three temporal samples per exposure, 48 stellar and 24 opaque-body radial nodes, 64 angular nodes per interval |
| Transit selection | 194 in-transit and 194 reference exposures |
| Wavelength coverage | 588.5–592.9 nm in vacuum, including both Na D lines |
| Planetary atmosphere | Isothermal 2,750 K, H₂/He, Na mass fraction 0.001 |
| Pressure grid | 445 layers from 10⁻¹⁵ to 100 bar, reference pressure 0.01 bar |
| Quiet stellar intensities | The Tutorial 13 MARCS/Turbospectrum Na-NLTE grid |
| Geometry | Projected rotation 3.5 km/s, projected obliquity 0° |
| Tellurics and noise | The same SkyCalc reference, ANDES ETC and seeds 12345 |
| Preparation | Individual Molecfit fits followed by `Seidel22`, with the same continuum windows and no stellar-core mask |

We retain the full Tutorial 13 interval because it supplies the continuum windows and
Molecfit anchors used by Tutorial 13. The figures focus on D2. Its configured
centre is **589.158326 nm in vacuum**, corresponding to the line commonly
quoted near 5889.95 Å in air. Keep this wavelength convention when comparing
velocities or selecting a line window.

Prepare the stellar grid and observing-sequence NPZ as described in Tutorial
13. Set `paths.prt_input_data` to your pRT data directory. The same pRT 3.2.0
opacities and ESO Molecfit installation are required.

## Step 2: Supply the local active-region spectra

A stellar component supplies its local specific intensity and continuum.
A surface region places that component on the disk. The spatial integrator
then blocks or transmits the light from the component behind each planetary
ray. We use the [stellar-component interface](heterogeneous_stellar_surface.md)
for all the calculations below.

For this experiment, we construct two controlled components from the quiet
intensity grid. At local stellar velocity (v), define

```{math}
g(v)=\exp\left[-\frac{v^2}{2(6~{\rm km\,s^{-1}})^2}\right],
\qquad q(\lambda,\mu)=I_{\rm quiet}/C_{\rm quiet}.
```

The cool component has

```{math}
C_{\rm spot}=0.60\,C_{\rm quiet},
\qquad
I_{\rm spot}=C_{\rm spot}\,q\,[1-0.40\,g(v)].
```

Its continuum is 40% dimmer, and the normalized D2 intensity is reduced by a
further factor of 0.60 at line centre. The facular component uses

```{math}
C_{\rm facula}=[1.10+0.20(1-\mu)]\,C_{\rm quiet},
\qquad
I_{\rm facula}=C_{\rm facula}\,[q+0.40(1-q)g(v)].
```

Its continuum contrast grows towards the limb. The added D2 core filling
produces a central reversal within the absorption profile. We choose the
0.40 profile coefficient and region radius below to make the stellar effect
visible in this single synthetic transit. These prescriptions retain the
quiet grid's wavelength and mu sampling. All components share the configured
rotation field and have zero additional velocity offsets. Their spectra
demonstrate the geometry and the effect of an incomplete stellar model.
They have not been validated as active-region models of HD 189733.

Generate the two component files with:

```bash
python scripts/run_stellar_activity_tutorial.py --prepare
```

This reads the quiet bundle at
`inputs/ANDES/HD189733b/sodium/stellar_Na_intensities.npz` and writes
`spot.npz` and `facula.npz` under
`inputs/ANDES/HD189733b/stellar_activity/`. Each file retains the intensity
units, continuum scale and provenance of its input. A component from an
external PHOENIX, 3D, MHD or NLTE calculation can use the same numerical input
contract when suitable local intensities are available.

## Step 3: Place the regions

The quiet component fills the disk. A circular region replaces its spectrum
at selected projected positions, in stellar-radius units:

```json
"stellar_surface": {
  "default_component": "quiet",
  "components": {
    "quiet": {
      "intensity_model_file": "inputs/ANDES/HD189733b/sodium/stellar_Na_intensities.npz"
    },
    "spot": {
      "intensity_model_file": "inputs/ANDES/HD189733b/stellar_activity/spot.npz"
    },
    "facula": {
      "intensity_model_file": "inputs/ANDES/HD189733b/stellar_activity/facula.npz"
    }
  },
  "regions": [
    {"component": "spot", "geometry": "projected",
     "x": -0.35, "y": -0.65, "radius": 0.25}
  ],
  "correction_surface": "quiet"
}
```

The comparison driver calculates these surfaces:

| Surface | Component | Centre `(x, y)` | Projected radius |
|---|---|---|---|
| Quiet reference | quiet throughout | — | — |
| Unocculted spot | spot | `(−0.35, +0.55)` | 0.25 |
| Spot crossing | spot | `(−0.35, −0.65)` | 0.25 |
| Facular crossing | facula | `(−0.35, −0.65)` | 0.25 |

The unocculted spot lies above the transit chord. The crossing cases place
the region on the approaching stellar hemisphere, so the planet encounters
it before mid-transit. The spot spectrum and radius are identical in both
spot cases. Changing position also changes the viewing angles sampled by
that region. The facular case uses the crossing geometry with its own local
spectrum.

Projected regions stay fixed during this observation. Local stellar rotation
still Doppler-shifts their spectra. This makes the chosen crossing geometry
easy to follow. Physical latitude/longitude and rotational epochs are described
in the surface guide.

```{figure} figures/tutorial14_surface_components.png
:width: 100%
:align: center

The projected regions and the common transit chord. The black disk marks the
planet near the crossing position using its catalog reference radius.
Bottom right: intrinsic local intensities at mu = 0.75, divided by the quiet
continuum on the same wavelength grid. The relative brightness of all
components is retained in the forward calculation.
```

## Step 4: Generate and prepare each observation

Run the comparison from the repository root:

```bash
python scripts/run_stellar_activity_tutorial.py --run
```

If you have already completed Tutorial 13, add
`--reuse-quiet outputs/hd189733b_andes_sodium` with its actual output directory.
The driver checks the scientific inputs, replays four phases and verifies
the archived telluric corrections before reusing that quiet observation.

The driver calculates one pRT atmospheric ray table and uses that same
immutable table for all surfaces. Each surface gets a separate forward
observation, including its local stellar illumination, telluric transmission,
instrument response, throughput and Gaussian noise. The observing conditions
and noise prescription stay fixed.

Molecfit is fitted separately to each observation. Its accepted corrections
are saved before the stellar comparison. For a given heterogeneous observation,
we then change only the assumed stellar correction:

```python
# The matched correction uses the injected heterogeneous surface.
config.stellar_surface.correction_surface = "matched"

# The quiet correction uses the default component over the whole star.
config.stellar_surface.correction_surface = "quiet"
```

RM and CLV remain enabled in both corrections, including the quiet-star
assumption. Both analyses use the same noisy flux, uncertainties, telluric fits, continuum
windows and reference exposures. The driver records their observation hashes
and checks that they agree. This avoids drawing another noise realization
when comparing the correction models.

As in Tutorial 13, the correction is prepared from the opaque-body stellar
spectrum. The planetary atmosphere enters the observation through the local
ray calculation. `Seidel22` divides the individual stellar-frame transmission
spectra by the prepared stellar correction before alignment with the planet.
It propagates 600 native-noise draws through the continuum and reference
calculation. The telluric and stellar models remain fixed during those draws.

The output directory contains one folder per surface, with `matched/` and
`quiet/` analyses below each heterogeneous observation. A quiet-star reference
needs only the matched analysis. Existing completed cases can be resumed after
the saved configurations and observation hashes have been checked.

```{figure} figures/tutorial14_stellar_Na_D2.png
:width: 100%
:align: center

Unobscured stellar spectra at the first exposure, before telluric absorption
and instrumental convolution. All spectra use the quiet disk's continuum
scale. The lower panel shows their fractional change from the quiet spectrum.
The two spot spectra nearly overlap here, although their transit signatures
differ once the planet encounters the lower region.
```

The cool region removes stellar light and deepens the local sodium core.
The facular component fills the core, producing a much larger fractional
change there than in the continuum. This is why a continuum brightness
contrast alone would miss part of the effect in a resolved spectral line.

## Step 5: Compare the recovered D2 profiles

Generate the comparison figures after the campaign finishes:

```bash
python scripts/plot_stellar_activity_tutorial.py \
    outputs/hd189733b_andes_stellar_activity docs/figures
```

The script also writes `line_measurements.json` beside the observations.
It reports the absorption averaged over ±20 km/s, following Tutorial 13,
and a central depth averaged over ±3 km/s. The velocity centroid is the
absorption-weighted first moment within ±20 km/s. The RMS width uses the
second moment in that same interval. These finite-window measures should be
compared using the same bounds in every case.

The planetary opacity is identical in all cases, while the stellar spectrum
illuminating the annulus can differ. The matched result therefore follows
the prepared noiseless model of its own observation. The paired quiet-star
analysis tests the residual introduced by an incomplete stellar correction.

```{figure} figures/tutorial14_Na_D2_phase_maps.png
:width: 100%
:align: center

Stellar-frame transmission around D2. The quiet reference uses its matching
correction. The other panels show the quiet-star assumption applied to each
heterogeneous observation. Red denotes absorption. The dashed curve follows
the planet's relative velocity, and the horizontal dotted lines mark the
four transit contacts. All panels share a scale enclosing 98% of the absolute
transmission deviations. Stronger residuals reach the colour limits.
```

The crossing region is encountered before mid-transit. Its residual remains
close to the stellar line while the planetary absorption moves across it.
The facular crossing produces a strong absorption residual when its brighter
line core is occulted and the correction assumes a quiet photosphere. The
unocculted spot produces a smaller change throughout the transit.

We can follow the crossing and the line profiles together in the animations
below. Generate them from the completed observations with:

```bash
python scripts/animate_stellar_activity_tutorial.py \
    outputs/hd189733b_andes_stellar_activity docs/figures
```

Each animation has 32 frames. A frame averages 12 or 13 consecutive exposures,
covering about 7.1–7.7 minutes. These temporal averages are used for the display.
The line measurements retain the full preparation described above.

```{figure} figures/tutorial14_unocculted_spot.gif
:width: 100%
:align: center

Unocculted spot. Left: the projected surface and moving planet, marked at its
catalog radius and the mean phase of the exposure bin. Top right: the intrinsic star-plus-planet line, divided by
its own stored continuum, for the quiet and active observations. Middle
right: the fractional change between those normalized line profiles.
Bottom right: the prepared transmission under matched and quiet-star
corrections, with the quiet observation in grey. The dashed vertical marker
follows the planet's relative velocity. All axes remain fixed during playback.
```

```{figure} figures/tutorial14_crossed_spot.gif
:width: 100%
:align: center

Spot crossing. The middle panel resolves the small change that is difficult
to see where the full stellar profiles overlap. The lower panel follows the
residual left by the quiet-star correction during the crossing.
```

```{figure} figures/tutorial14_crossed_facula.gif
:width: 100%
:align: center

Facular crossing. Occulting the brighter local D2 core changes the stellar
line. The quiet-star correction leaves a strong absorption residual during
the encounter. The matched correction follows the supplied heterogeneous
surface.
```

```{figure} figures/tutorial14_Na_D2_profiles.png
:width: 100%
:align: center

Planet-frame D2 profiles from the same observations under the two stellar
corrections. The dotted blue curve is each matched observation's prepared
noiseless model. Grey shows the quiet-star reference. Shading gives the
conditional noise uncertainty on each profile. The three panels use
independent vertical scales.
```

The measurements for this execution are:

| Surface | Assumed correction | Mean absorption, ±20 km/s (%) | Core, ±3 km/s (%) | Centroid (km/s) | RMS width (km/s) |
|---|---|---|---|---|---|
| Quiet reference | Matched | 0.1510 ± 0.0078 | 0.3577 | +1.25 | 9.03 |
| Unocculted spot | Matched | 0.1577 ± 0.0078 | 0.3719 | +1.24 | 9.03 |
| Unocculted spot | Quiet | 0.1644 ± 0.0078 | 0.3767 | +1.32 | 9.13 |
| Spot crossing | Matched | 0.1251 ± 0.0078 | 0.2989 | +0.97 | 9.18 |
| Spot crossing | Quiet | 0.0917 ± 0.0079 | 0.2653 | −2.26 | 8.80 |
| Facular crossing | Matched | 0.2583 ± 0.0075 | 0.6370 | +2.70 | 7.62 |
| Facular crossing | Quiet | 0.5889 ± 0.0074 | 1.2463 | +5.04 | 5.57 |

The matched band depths agree with their prepared noiseless values to within
about 0.011 percentage points. Those values are 0.1600%, 0.1686%, 0.1358%
and 0.2658% for the quiet, unocculted-spot, spot-crossing and facular-crossing
observations. Different local illumination and disk dilution change the
transmission spectrum even with the same planetary opacity.

To isolate the error from the stellar correction, subtract each matched result
from the quiet-star analysis of that same observation:

| Surface | Change in mean absorption (ppm) | Centroid change (km/s) |
|---|---|---|
| Unocculted spot | +67 | +0.09 |
| Spot crossing | −334 | −3.22 |
| Facular crossing | +3305 | +2.34 |

The unocculted-spot shift is small compared with the roughly 78 ppm conditional
band uncertainty of this night. The spot crossing weakens the recovered
absorption and leaves excess flux on its red side, shifting the centroid
bluewards. The facular crossing makes the recovered line much deeper and
asymmetric. The corresponding noiseless paired band shifts are +67, −334
and +3311 ppm, so these differences persist when Gaussian noise is removed.

The paired profiles share their noise and telluric fits, so their errors are
correlated. The quoted uncertainties condition on the fitted telluric and
stellar models. Centroids and widths describe the finite integration window,
including any stellar residual and excess-flux samples. A centroid shift here
cannot be assigned directly to an atmospheric wind, and the RMS width is
different from a fitted Gaussian FWHM. The planetary velocity and atmospheric
model have stayed fixed throughout.

## Runtime and checks

The full comparison took 6.06 hours on an Apple M2 Ultra with 24 CPU cores
and 64 GB RAM, running macOS 15.1. Molecfit used 16 workers.
Each forward calculation used one process, with the
next surface calculated alongside the previous surface's telluric fits.

| Surface | Forward calculation (min) | Molecfit, 388 exposures (min) | Both stellar analyses (min) |
|---|---|---|---|
| Unocculted spot | 69.6 | 76.2 | 13.9 |
| Spot crossing | 97.0 | 70.9 | 13.7 |
| Facular crossing | 97.7 | 73.7 | 13.5 |

The quiet reference reuses the completed Tutorial 13 observation and its 388
accepted telluric fits. Recalculating its full intrinsic spectrum at four
widely separated phases gave zero numerical difference in the stellar,
opaque-body, atmospheric and continuum arrays. Its line preparation was
repeated with the same 600-draw treatment used for the active cases. A fresh
execution without an archived quiet reference also calculates that observation.

The validation record checks the shared atmospheric-ray hash, observing
sequence, velocity arrays, telluric transmission and ETC response. It also
checks that each paired analysis uses the same observation and fits. The
unocculted spot has no overlap with either the opaque planet or its atmosphere.
Both crossing regions overlap the opaque planet during 80 exposure midpoints.
The supplied continuum ratios remain 0.60 for the spot and 1.10–1.30 for the
facular component.

For another experiment, change the component files and region positions in
`stellar_surface`. Keep the planetary model fixed while testing the stellar
assumption. The generated `comparison_validation.json` and
`line_measurements.json` record the numerical checks and measurements for
the run being inspected.
