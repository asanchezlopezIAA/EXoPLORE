# Changelog

## v1.0 - First public release (2026)

First public release of EXoPLORE, a framework for simulating and retrieving
high-resolution transmission and emission spectra of exoplanet atmospheres.

**Simulation**
- Forward models built with petitRADTRANS, with equilibrium chemistry
  (EasyChem) or manual mass fractions, over isothermal or Guillot
  temperature profiles.
- Pseudo-2D atmospheres: independent morning and evening limbs with their own
  temperature, chemistry, winds, and rotation, combined through a transit
  light curve.
- Instrument models for CARMENES NIR and ANDES, with per-exposure telluric
  contamination (airmass-scaled or full per-exposure SkyCalc), realistic
  photon noise, and multi-night campaigns with night-specific conditions.

**Analysis**
- Preparation pipelines implementing the methods presented in
  [Brogi & Line (2019)](https://doi.org/10.3847/1538-3881/aaffd3),
  [Sánchez-López et al. (2019)](https://doi.org/10.1051/0004-6361/201936084),
  [Gibson et al. (2022)](https://doi.org/10.1093/mnras/stac091), and
  [Blain et al. (2024)](https://doi.org/10.3847/1538-3881/ad2c8b), including
  SYSREM detrending with fast model filtering. These are starting points to be
  studied and adapted for the data at hand, not applied blindly (see the
  [Concepts primer, Section 3](concepts.md#3-the-preparation-pipeline-removing-the-contaminants-and-its-cost)).
- Cross-correlation with Kp-Vsys mapping and three detection metrics
  (S/N, Welch t-test, and the cross-correlation-to-log-likelihood value).

**Retrieval**
- Bayesian atmospheric retrieval with nested sampling (PyMultiNest) using the
  log-likelihoods of Brogi & Line (2019), Blain et al. (2024), and
  Gibson et al. (2022), selected in the config via the `BL19`, `Blain24`, and
  `Gibson22` tags respectively.
- Statistical validation tools: significance studies over many noise
  realisations and p-p (coverage) calibration plots.

**Documentation**
- Installation and quick-start guides, a concepts primer, eight tutorials,
  and full configuration and output references.
