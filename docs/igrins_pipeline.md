# The Cheverall26 preparing pipeline (IGRINS HRCCS)

`pipeline.name: "Cheverall26"` selects the IGRINS high-resolution
cross-correlation recipe of Cheverall et al. (2026), which builds on the IGRINS
analysis lineage of Line et al. (2021), Brogi et al. (2023) and Smith et al.
(2024). It applies to any IGRINS transit time series ingested into a
`reference_night/` (see [Input files](input_files.md) for the format). A worked
end-to-end example is [Tutorial 10](tutorial.md#tutorial-10-analyse-real-igrins-data-l-98-59-d).

## What the pipeline does

Per echelle order, on the real-data time series:

1. **Rescaling and continuum normalisation.** Divide each spectrum by its
   median, clip outliers over time and divide by the estimated continuum.
   `pipeline.continuum_method` selects `maxima` (default), `cm2024` or `polyfit`.
2. **Bad-pixel and low-throughput masking.** Channels flagged as outliers, and
   the deepest telluric regions with insufficient continuum, are masked. Orders
   that are essentially fully masked in deep telluric bands are dropped from the
   co-add.
3. **Detrending.** Temporal components suppress correlated structure.
   `pipeline.detrend_method` selects SYSREM or PCA, and
   `pipeline.sysrem_iterations` sets the component count. The example JSON
   records its chosen settings. Order-by-order optimization can select the
   count using the configured criterion and an injected model.
4. **Matched-filter cross-correlation.** The detrended residuals are
   cross-correlated with a petitRADTRANS template convolved to the IGRINS
   resolution, over a velocity grid, with a data-driven per-channel weighting.
5. **Kp-Vsys maps and significance.** The per-order CCFs are shifted to each
   trial planet velocity and co-added. The S/N is read against the off-trail
   scatter (the velocity band within `cross_correlation.snr_exclude_kms` of the
   trail is excluded from the noise estimate).

## The noise convention

The detection statistic follows Cheverall et al. (2026): a **data-driven
per-channel noise**, the median absolute deviation of each wavelength channel
over time. This is the
standard of the IGRINS HRCCS literature (Line et al. 2021, Brogi et al. 2023,
Smith et al. 2024) and is set together with the χ²-form of the likelihood by the
pipeline name. The cross-correlation S/N is read against the scatter of the map
away from the planet trail.

## Injection-recovery

As for the CRIRES+ pipeline, a scaled template can be injected into the real
residuals (`observation.simulate_planet: true` with
`observation.scale_injection` and `pipeline.kp_vrest_injection`) and recovered
through the full detrending, to calibrate how much signal the pipeline preserves
and to turn a non-detection into an upper limit.

## Worked example

See [Tutorial 10](tutorial.md#tutorial-10-analyse-real-igrins-data-l-98-59-d)
for the L 98-59 d transit of Cheverall et al. (2026), including the config, the
run command, and the pipeline-stage and Kp-Vsys figures.

## References

- Cheverall, C., & Madhusudhan, N. 2026 (arXiv:2603.02209): the IGRINS HRCCS
  analysis of L 98-59 d reproduced by this pipeline.
- Cheverall, C., & Madhusudhan, N. 2024 (arXiv:2403.18894): HRCCS feasibility
  for low-velocity planets.
- Line, M. R., et al. 2021, Nature 598, 580 (WASP-77Ab): the IGRINS
  retrieval/likelihood framework.
- Brogi, M., et al. 2023, AJ 165, 91 (WASP-18b): IGRINS HRCCS + retrievals.
- Smith, P. C. B., et al. 2024, AJ 168, 293 (WASP-121b): IGRINS HRCCS.
- Brogi, M., & Line, M. R. 2019, AJ 157, 114: the HRCCS likelihood mapping.
- Tamuz, O., Mazeh, T., & Zucker, S. 2005, MNRAS 356, 1466: SYSREM.
