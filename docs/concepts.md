# Concepts: how high-resolution spectroscopy works

This page is a self-contained primer on the physics and statistics behind
high-resolution spectroscopy (HRS) of exoplanet atmospheres, written for
readers who are comfortable with exoplanets and spectroscopy but new to the
cross-correlation technique and to HRS retrievals. Every idea introduced here
maps directly onto a configuration block or an output of EXoPLORE, so the page
doubles as a conceptual index to the rest of the documentation. The tutorials
then put each idea into practice.

The central idea is that the planet **moves** while the telluric and stellar
contaminants are comparatively fixed in velocity. Much of what follows is a
consequence of that separation.

---

## 1. The Doppler trick

A transiting hot Jupiter orbits its star at tens to hundreds of kilometres per
second. During the few hours of a transit, the component of that velocity
along our line of sight changes measurably, so the planet's spectral lines
**shift** in wavelength from one exposure to the next. The radial velocity of
the planet at time `t` is

```
V_p(t) = V_sys + V_bary(t) + K_P · sin(2π φ(t)),
```

where `V_sys` is the systemic velocity of the star, `V_bary(t)` is the
barycentric correction (the Earth's motion towards or away from the target),
`K_P` is the planet's orbital velocity semi-amplitude, and `φ(t)` is the
orbital phase. The planet's contribution, `K_P · sin(2π φ)`, sweeps through
velocity space during transit.

The two contaminants that dominate a ground-based near-infrared spectrum
behave very differently:

- **Telluric lines** (absorption by Earth's atmosphere) sit at rest in the
  observatory frame. They are essentially stationary in wavelength, varying
  only slowly with airmass and precipitable water vapour.
- **Stellar lines** sit at `V_sys + V_bary(t)`, which is also quasi-static
  over a night, provided the star is quiescent. This assumption is not always
  satisfied: flares, spots, and granulation introduce time-dependent stellar
  signals, and during a transit the Rossiter-McLaughlin effect and the
  centre-to-limb variation of the stellar lines distort the residual stellar
  spectrum in a way that the simple picture above neglects.

The planet signal is therefore the component that marches most steadily across
the detector while the contaminants remain comparatively fixed. This separation
in the time-velocity plane underlies the technique, and it is also why HRS does
not require a reference star or absolute flux calibration: the information lives
in the **motion** of narrow lines rather than in the broadband continuum.

```{figure} figures/doppler_trick.png
:width: 90%
:align: center

Schematic of a time-series spectral matrix during a transit (wavelength on the
horizontal axis, time or orbital phase on the vertical axis; darker means more
absorption). The telluric and stellar lines are stationary in wavelength and
appear as vertical features, while the planet's lines Doppler-shift during the
transit and trace an inclined trail. The cross-correlation technique exploits
this separation to isolate the moving planet signal.
```

The Doppler shift is the primary handle, but not the only one. The planetary
absorption signal is present **only in transit** (in transmission) or
**around secondary eclipse** (in dayside emission), so the timing of the signal
relative to the orbital ephemeris provides additional discrimination: a
correlation feature that appears outside the expected window is unlikely to be
planetary. This timing constraint is what allows slowly moving planets to be
studied at all, even when their velocity changes little across an event.

A planetary absorption line is individually far too weak to see (its depth is
a small fraction of the photon noise). The signal is built by combining the
hundreds or thousands of lines of a molecular band coherently, which is the
job of the cross-correlation function (Section 4).

> In EXoPLORE the velocities above are computed from the planet parameter file
> (`K_P`, `V_sys`, eccentricity) and the observing geometry. A planet with a
> large velocity swing during transit (for example HD 189733 b) separates
> cleanly from the stationary contaminants. A slowly moving planet (a temperate
> super-Earth on a wide orbit) barely shifts during an event and is
> correspondingly harder, since its lines overlap the contaminants for longer;
> it is not, however, beyond reach, and the in-transit timing constraint above
> helps to recover such signals.

---

## 2. The forward model: building the planet signal

Before we can analyse data we must be able to **generate** it. EXoPLORE builds
a synthetic transmission (or emission) spectrum with
[petitRADTRANS](https://petitradtrans.readthedocs.io) (Mollière et al. 2019,
A&A, 627, A67), using equilibrium chemistry from
[EasyChem](https://easychem.readthedocs.io) (Lei & Mollière 2024,
arXiv:2410.21364) or manual mass fractions, over a pressure-temperature profile
(isothermal or Guillot). This is configured in the `atmosphere` block and
explained in Tutorial 3.

```{figure} figures/model_spectrum.png
:width: 95%
:align: center

A model transmission spectrum of HD 189733 b over part of the near-infrared
(transit depth against wavelength), computed with petitRADTRANS. The forest of
narrow lines, here dominated by H₂O, is the planetary signal that EXoPLORE
injects and that the cross-correlation later combines. Individual lines are far
below the photon noise of a single exposure; their number is what makes the
detection possible.
```

The terminator of a real hot Jupiter is generally **not** uniform. The morning
(leading) limb probes gas arriving from the cool nightside; the evening
(trailing) limb probes gas from the hot dayside. These two limbs can differ in
temperature, chemistry, and the direction of their winds, and they contribute
to the observed spectrum with **phase-dependent weights** as the planet crosses
the stellar disk. EXoPLORE can build this pseudo-2D structure (two limbs, each
with its own temperature, chemistry, wind, and rotational broadening) and
combine them through a transit light curve, via the `limb_asymmetries` option
and the morning/evening sub-blocks (Tutorial 3, Example D).

```{figure} figures/limb_asymmetry_ccf.png
:width: 100%
:align: center

Time-resolved cross-correlation signal of pseudo-2D atmospheres, as EXoPLORE
simulates them for ANDES. **Top row:** cross-correlation maps in Earth's
rest-frame velocity against orbital phase (dashed lines mark the transit
contacts T₁, T₄). **Bottom row:** the corresponding 1D cross-correlation
functions averaged over ingress (blue) and egress (red), which isolate the
leading and trailing limbs respectively. **Left:** a H₂O signal from
HD 189733 b with a modest day-to-nightside wind; the two limbs share almost the
same velocity, so they merge into a single, slightly asymmetric trail (the
morning and evening limbs only enter and leave transit at slightly different
times), and the ingress and egress CCFs peak close together. **Middle:** neutral
iron in the ultra-hot Jupiter WASP-76 b, where a day-to-nightside wind plus a
time-varying limb contribution shift the peak progressively bluewards from
ingress to egress. **Right:** a hot Jupiter with a jet-like wind (≈ 7.7 km s⁻¹
towards the observer on the trailing limb and away from it on the leading limb),
of the kind resolved in WASP-127 b by Nortmann et al. (2025). The velocity
offset between the limbs now exceeds a resolution element, so the signal splits
into two distinct components (a morning component dominating from ingress to
mid-transit and an evening component from mid-transit to egress) that cross near
the centre of the transit and appear as two separate peaks in the ingress and
egress CCFs. The same atmosphere therefore writes a structured, time-dependent
imprint on the data that a single homogeneous model cannot reproduce.
```

This matters for a reason that becomes the central lesson of HRS retrievals in
the ELT era (Section 7): a spectrum produced by two distinct limbs, analysed
with a single homogeneous 1D model, yields parameters that are **precise but
biased**. You cannot diagnose that failure unless you can first simulate the
inhomogeneous truth, which is exactly what the forward model is for.

---

## 3. The preparation pipeline: removing the contaminants, and its cost

To expose the moving planet signal we must remove the stationary telluric and
stellar contributions. EXoPLORE implements several literature recipes
(`pipeline.name`: BL19, Blain24, ASL19, Gibson22), all of which share the same
logic:

1. **Normalise** each spectrum to remove the blaze and broadband throughput,
   typically by dividing out a low-order polynomial fit to the continuum.
2. **Remove the quasi-static structure**. The polynomial-based recipes exploit
   the near-linear dependence of log-telluric-transmission on airmass and fit a
   low-order polynomial in time (or in airmass) to each wavelength channel,
   dividing it out. The SYSREM-based recipes instead identify the dominant
   common-mode systematics by iteratively fitting and subtracting the leading
   components of the data matrix.
3. **Mask** the wavelength channels where telluric absorption is so deep that
   no usable signal survives.

```{figure} figures/pipeline_steps.png
:width: 90%
:align: center

The preparation pipeline acting on one ANDES order. (A) A single spectrum,
noiseless (black) and with noise added (red). (B) The noiseless time-series
matrix (wavelength against orbital phase); the transit appears as the central
band of reduced flux. (C) The same matrix with noise, throughput variations, and
telluric absorption (the vertical features). (D) The residuals after the
preparation pipeline, which removes the quasi-static telluric and stellar
structure; the strongly contaminated channels have been masked. EXoPLORE writes
this diagnostic for every run.
```

This removal comes at a cost. Because the planet's lines are only
quasi-stationary over a night (they move slowly relative to the dominant
systematics), any operation that removes time-correlated structure also erodes
and distorts a fraction of the planet signal.

A model template must therefore never be compared directly to prepared data.
The template must be put through the **same** preparation pipeline as the data,
so that it carries the same distortions. EXoPLORE does this when
`pipeline.prepare_template: true`. An inconsistent or omitted template
preparation is a recurrent source of biased results in the literature, and is
the subject of Tutorial 7.

> The SYSREM-based recipes raise a further question: how many components should
> be removed? Too few leaves systematics in the data; too many over-fits and
> erodes the planet signal. Choosing the number by maximising the detection
> significance is circular, since the analysis is then tuned until it shows the
> expected result. EXoPLORE provides three criteria via
> `pipeline.optimize_criterion`: maximising the recovered significance (the
> biased baseline, retained for comparison); optimising the recovery of a model
> signal injected at a velocity **away** from the planet, so the choice never
> sees the real signal; and a model-independent criterion based on the change in
> the residual standard deviation between successive SYSREM iterations
> (`delta_sigma`), following Parker et al. (2025, MNRAS, 538, 3263), which
> avoids assuming a model altogether. The two latter approaches both avoid tuning the
> analysis on the real signal.

---

## 4. The cross-correlation function and the Kp-Vsys map

With prepared data and a prepared template, we cross-correlate. For a velocity
shift `v` applied to the template at each time frame, EXoPLORE computes a
weighted correlation between the residual data and the shifted template. A
common definition of the weighted CCF (e.g. Parker et al. 2025, MNRAS, 538,
3263) is

```
CCF(v, t) = Σ_j  R_j(t) · M_j(v) / E_j(t)²,
```

where `R` is the matrix of residual spectra (the data after SYSREM or
polynomial correction), `M` is the synthetic model shifted by `v`, `E` is the
uncertainty on `R`, `t` indexes the exposures and `j` the wavelength channels.
The inverse-variance weighting `1/E²` ensures noisy channels contribute less.
EXoPLORE additionally offers a Pearson-style normalised correlation, bounded in
`[-1, 1]` and insensitive to any residual continuum offset, which allows
amplitudes to be compared order by order.

Computing this for every velocity shift and exposure produces a
**cross-correlation matrix** as a function of time and velocity. A planetary
signal traces a curved trail through this matrix, because its velocity changes
with orbital phase. To collapse the trail into a single detection statistic, we
test every candidate orbit: for each pair `(K_P, V_rest)`, each exposure is
shifted into the corresponding planet rest frame and co-added. The result is the
**Kp-Vsys map**, in which a signal appears as a compact peak at the planet's
`K_P` and rest velocity. EXoPLORE produces it for every run (Tutorials 1 and 2).

```{figure} figures/tutorial1_kpvsys_andes.png
:width: 80%
:align: center

Kp-Vsys detection map for a simulated ANDES transit of HD 189733 b. The H₂O
signal appears as a compact peak at the expected K_P and rest velocity (red
dashed lines). The colour scale is the cross-correlation significance. A peak
displaced from V_rest = 0 would indicate a net Doppler shift, such as a
day-to-night wind.
```

Two features of the map are physical rather than incidental:

- **The peak behaves as a matched filter.** The cross-correlation co-adds many
  lines that are quasi-periodic in velocity space, so the response is that of a
  matched filter with a zero-mean kernel. At high signal-to-noise this produces
  faint anti-correlation lobes flanking the main peak. These lobes arise from
  the autocorrelation function of the template itself and do not represent
  artefacts; they are a modest, expected cost of using a densely lined template,
  and they only rise above the noise in very high-quality data (such as
  simulated ANDES spectra).
- **The position encodes physics.** A peak displaced from `V_rest = 0` indicates
  a net Doppler shift, for example a day-to-night wind. A peak at an unexpected
  `K_P` more often signals an imperfect order selection or residual systematics
  than a discovery.

---

## 5. From cross-correlation to a likelihood

Cross-correlation indicates **whether** a species is present and roughly where
in velocity space. To measure **how much** of it is present, with formal
uncertainties, requires a likelihood and a sampler. EXoPLORE offers three
log-likelihood formulations (`retrieval.log_likelihood`). The choice affects the
resulting constraints, for reasons that are worth setting out explicitly.

All three assume Gaussian noise and differ in how they treat the **noise
scale**.

**Brogi & Line (2019, AJ, 157, 114), `BL19`.** Starting from a Gaussian
likelihood with a single unknown noise level `σ` common to all pixels of a
spectrum, `σ` is analytically replaced by its maximum-likelihood estimate (the
root mean square of the residuals) and substituted back. The result is

```
ln L_BL19 = -(N/2) · ln(s_f² - 2R + s_g²),
```

where `s_f²` and `s_g²` are the mean squared data and model and `R` is their
cross-covariance; the bracket equals the mean squared residual. No per-pixel
uncertainty enters: BL19 estimates one global noise level per spectrum from the
data itself. It is therefore self-calibrating and insensitive to absolute flux.
Brogi & Line (2019) developed and validated this formulation on simulated CRIRES
K-band data, in which the noise is dominated by stellar photon (Poisson)
statistics, and showed that it recovers statistically correct credibility
intervals (their test statistic follows a χ² distribution). Estimating the noise
from the data itself is a well-motivated choice when reliable per-pixel
uncertainties are not readily available.

**Blain, Sánchez-López & Mollière (2024, AJ, 167, 179), `Blain24`.** This uses
the **known** per-pixel uncertainty `σ(n)`, propagated through the preparation
pipeline, in a chi-square:

```
ln L_Blain24 = -(1/2) · Σ_n [ (d(n) - m(n)) / σ(n) ]².
```

Each pixel is weighted by `1/σ(n)²`. The formulation derives from the framework
of Gibson et al. (2020, MNRAS, 493, 2215).

**Gibson et al. (2022, MNRAS, 512, 4618), `Gibson22`.** The same chi-square with a
global scale factor `β` multiplying the uncertainties, plus a `-N ln β` penalty.
`β` absorbs a uniform under- or over-estimate of the noise. It is identifiable
on noisy data but diverges on noiseless data (the residuals vanish, so the
likelihood is maximised by `β → 0`); for noiseless bias tests `β` is pinned near
1 with an informative prior (Tutorial 7).

### Whether the difference matters depends on the noise structure

The per-pixel weighting of Blain24 is not always the better approach. The script
`scripts/illustrate_likelihood_weighting.py` examines this on a toy absorption
spectrum, comparing how tightly each likelihood constrains the line depth under
two noise structures of **identical total variance**:

```{figure} figures/likelihood_weighting.png
:width: 100%
:align: center

Log-likelihood as a function of the line-depth parameter (truth = 1) for a toy
spectrum. **Left:** with uniform (homoscedastic) noise the two formulations are
almost identical. **Right:** with heteroscedastic noise of the same total
variance, concentrated in 15 per cent of the pixels, the per-pixel-weighted
formulation of Blain et al. (2024) yields a tighter constraint than that of
Brogi & Line (2019) in this particular toy case. Generated with
`python scripts/illustrate_likelihood_weighting.py`.
```

- With **uniform** noise the two agree to within ten per cent. The per-pixel
  weighting has nothing to redistribute.
- With **heteroscedastic** noise (the same noise budget concentrated in a
  minority of pixels) the Blain24 constraint is roughly five times tighter in
  this configuration.

The mechanism follows from the formulas. The `s_f²` term of BL19 is a single
number for the whole spectrum, so a noisy minority of pixels inflates that
estimate and dilutes the signal carried by the clean majority. Blain24 uses
`σ(n)` directly and assigns the noisy pixels little weight, preserving the
signal in the clean ones. Real spectra are frequently heteroscedastic (telluric
cores, blaze edges, and channels near deep lines all carry elevated noise), so
for datasets with strong per-pixel noise variation, such as the single noisy
ANDES order considered later in this documentation, the Blain24 and Gibson22
formulations may yield tighter posteriors. This does not imply that BL19 is less
accurate: when the noise among the retained pixels is uniform the two converge,
and BL19 remains a sensible choice when reliable per-pixel uncertainties are not
available. The appropriate formulation depends on the dataset rather than on a
general ordering of the methods.

> The same reasoning explains why BL19 may yield broader constraints on a single
> noisy order while nearly matching the other formulations on noiseless data:
> with vanishing noise there is no heteroscedasticity to exploit, so the three
> converge. This can be seen directly in the noiseless and noisy corner plots of
> Tutorial 7.

---

## 6. Detection significance: three complementary measures

The significance of a detection has more than one defensible definition, and a
careful analysis reports more than one. EXoPLORE provides:

- **Cross-correlation S/N.** The co-added CCF value at the planet's position
  divided by the standard deviation of the CCF far from the trail (more than a
  chosen velocity away). It is fast and is the standard first-pass statistic,
  but it assumes the off-peak region represents the noise faithfully.
- **Welch's t-test.** A comparison of the distribution of in-trail correlation
  values with the out-of-trail distribution. It is less sensitive to a single
  hot pixel than the S/N, at the cost of assumptions about the two populations.
- **Bayesian evidence.** The marginal likelihood of a model containing the
  species compared with a null model without it; the ratio is a Bayes factor,
  which can be mapped to a frequentist sigma. It accounts for the prior volume
  and the full parameter space and is the most rigorous of the three, but also
  the most expensive.

The three measures rest on different assumptions and need not agree; comparing
them is itself informative, and is the subject of Tutorial 8.

---

## 7. Precision is not accuracy

As data quality improves (more orders, higher resolving power, the collecting
area of the ELT), the limiting factor shifts from the data to the **model**. A
1D homogeneous retrieval applied to a spectrum that is in reality the
phase-dependent sum of two chemically and dynamically distinct limbs still
converges, and returns tight posteriors. The recovered abundances are then
effective values that compress the inhomogeneous truth, and they can be biased:
weakly contributing species are sometimes over-constrained, acting as nuisance
absorbers that soak up structure the 1D model cannot otherwise reproduce. The
bias is also band-dependent, since different wavelength ranges weight the two
limbs differently, so the same atmosphere yields different effective abundances
in, for instance, the YJH and K bands.

For this reason EXoPLORE pairs the pseudo-2D forward model with the retrieval.
Simulating an inhomogeneous atmosphere allows the biases of the homogeneous
models still in common use to be exposed and quantified, and provides a testbed
for the multi-dimensional retrievals that ELT-quality data will require. A 1D
retrieval remains a useful diagnostic, but should not be read as a faithful
description of an inhomogeneous atmosphere at ELT quality: a tight posterior is
not on its own evidence of an accurate one.

---

## Where each concept lives in EXoPLORE

| Concept | Config / output | Tutorial |
|---|---|---|
| Planet velocities, observing geometry | planet parameter file, `observation` | 1 |
| Forward model, chemistry, T-P profile | `atmosphere` | 3 |
| Pseudo-2D limbs and winds | `atmosphere.limb_asymmetries`, morning/evening blocks | 3 |
| Preparation pipeline | `pipeline` | 4, 7 |
| Model reprocessing (template preparation) | `pipeline.prepare_template` | 7 |
| Cross-correlation and Kp-Vsys map | `cross_correlation`, Kp-Vsys output | 1, 2 |
| Multi-night co-addition | `observation.different_nights` | 5 |
| Likelihood choice | `retrieval.log_likelihood` | 6, 7 |
| Pipeline bias testing | noiseless retrieval | 7 |
| Detection significance | `cross_correlation` metrics | 8 |
