"""
exoplore.config.models
======================

Typed dataclasses that fully describe one EXoPLORE simulation.

A user defines a simulation by editing a JSON file (e.g.
``configs/hd189733b_andes_transit_clean.json``) and loading it with
:meth:`SimulationConfig.from_json`.  All scientific choices are explicit:
units are part of the field names, and defaults are conservative.

Design rules
------------
- Field names carry units where relevant (e.g. ``velocity_max_kms``).
- No ``inp_dat`` dictionary keys leak into this layer.
- Every field has a default so partial configs are accepted gracefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Planet
# ---------------------------------------------------------------------------

@dataclass
class PlanetConfig:
    """Identity and path to the planet parameter JSON file."""
    name: str = "HD189733b"
    parameter_file: str = ""


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

@dataclass
class InstrumentConfig:
    """Instrument and detector configuration.

    Parameters
    ----------
    name:
        Instrument identifier.  Supported values:
        ``"ANDES_YJHK"``, ``"ANDES_YJH"``, ``"ANDES_K"``,
        ``"ANDES_RIZ"``, ``"ANDES_UBV"``,
        ``"CARMENES_NIR"``, ``"CARMENES_VIS"``, ``"CRIRES+"``.
    observatory:
        Site name understood by Astropy / EsoSkycalc, e.g. ``"paranal"``.
        Usually set automatically by the instrument module; override only if
        you need to force a specific sky model.
    pixels_per_resolution_element:
        Number of detector pixels per resolution element (Nyquist = 2).
    order_indices:
        Spectral order indices to process.  Empty list = all orders for the
        selected instrument band.  For single-order instruments set ``[0]``.
    split_detectors:
        Treat each detector half as an independent order (e.g. CARMENES
        raw-data mode where each echelle order spans two detectors).
    convolve_to_resolution:
        Convolve synthetic spectra to the instrument's resolving power.
    """
    name: str = "ANDES_YJHK"
    observatory: str = "paranal"
    pixels_per_resolution_element: float = 2.5
    order_indices: List[int] = field(default_factory=list)
    split_detectors: bool = False
    convolve_to_resolution: bool = True


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class ObservationConfig:
    """Observing geometry and exposure parameters.

    Parameters
    ----------
    event_type:
        ``"transit"`` or ``"dayside"``.
    flag_event:
        Portion of the event to simulate: ``"full_event"``, ``"pre"``,
        or ``"post"``.  For transits use ``"full_event"``.
    pre_event_hours:
        Duration of the pre-event baseline window in hours.
    post_event_hours:
        Duration of the post-event baseline window in hours.
    n_nights:
        Number of independent nights to simulate.
    different_nights:
        If True, nights cover different orbital phases / orders.
    exposure_time_seconds:
        Individual exposure (DIT) in seconds.
    readout_time_seconds:
        Detector readout time in seconds.
    overhead_time_seconds:
        Per-exposure overhead in seconds.
    specific_event:
        Simulate a specific real event with provided JD / airmass.
    specific_T0_bjd:
        Reference mid-transit BJD for a specific event (None = use T_0
        from planet JSON).
    use_real_data:
        Analyse a real observed dataset instead of simulating one.
    noise_scaling_factor:
        Multiplicative scaling applied to all noise levels (1.0 = nominal).
    simulate_planet:
        Inject a planet signal.  If False only noise is simulated.
    external_planet_model:
        Use an externally provided planet model instead of generating one
        with pRT.
    external_planet_model_file:
        Path to the external planet model file (ASCII, two-column,
        wavelength and flux/contrast).  Required when
        ``external_planet_model=True``.
    signal_uses_light_curve:
        Scale in-transit signal with a BATMAN light curve.
    scale_injection:
        Multiplicative factor to scale the injected planet signal.
    significant_eccentricity:
        Use the eccentric Keplerian velocity formula (Wright & Howard 2009).
    berv_kms:
        Barycentric Earth Radial Velocity in km/s (0 for simulations).
    use_accurate_berv:
        Compute the per-exposure BERV from the target's sky coordinates
        and the observatory location (Astropy barycentric correction).
        Applies to fully synthetic ``different_nights`` simulations,
        where each night is placed at its own observable transit epoch
        and therefore carries its own BERV; stellar and planetary lines
        then shift relative to the telluric rest frame from night to
        night, as they do between real epochs.  Requires ``ra_deg`` and
        ``dec_deg`` in the planet parameter file (falls back to
        ``berv_kms`` with a warning if unavailable).  Single-night and
        real-data runs are unaffected and keep using ``berv_kms`` or the
        per-night BERV files.
    noiseless:
        Run a completely noiseless simulation (for testing).
    first_night_noiseless:
        Make only the first simulated night noiseless.
    add_throughput_variations:
        Add random throughput variations to each exposure.
    mask_v_rotsini:
        Mask the stellar v sin(i) velocity interval in the CCF
        (relevant for dayside observations).
    exposure_time_seconds_per_night:
        Per-night exposure times (DIT) in seconds for
        ``different_nights=True`` simulations.  Length must equal
        ``n_nights``.  When set, synthetic JD grids are generated
        independently for each night using the given DIT, enabling
        different cadences per night.  ``None`` = use
        ``exposure_time_seconds`` for all nights (uniform cadence).
    """
    event_type: str = "transit"
    flag_event: str = "full_event"
    pre_event_hours: float = 0.0
    post_event_hours: float = 0.0
    n_nights: int = 1
    different_nights: bool = False
    exposure_time_seconds: float = 30.0
    readout_time_seconds: float = 6.0
    overhead_time_seconds: float = 0.0
    specific_event: bool = False
    specific_T0_bjd: Optional[float] = None
    use_real_data: bool = False
    noise_scaling_factor: float = 1.0
    simulate_planet: bool = True
    external_planet_model: bool = False
    external_planet_model_file: str = ""
    signal_uses_light_curve: bool = True
    scale_injection: float = 1.0
    significant_eccentricity: bool = False
    berv_kms: float = 0.0
    use_accurate_berv: bool = True
    noiseless: bool = False
    first_night_noiseless: bool = False
    add_throughput_variations: bool = True
    mask_v_rotsini: bool = False
    exposure_time_seconds_per_night: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Noise model and SNR handling.

    Parameters
    ----------
    noise_choice:
        Source for the noise estimate: ``"SNR"`` (from SNR file / ETC)
        or ``"sig"`` (from uncertainty file / real data).
    fixed_snr:
        Fix a single SNR value for all pixels and frames.
        ``None`` = use the SNR from the file or ETC.
    use_mean_snr:
        Replace a time-varying SNR with its mean over time.
    snr_correction:
        Additive correction factor to shift the SNR at all pixels and
        frames (float; 0 = no correction).
    noise_seed:
        Integer seed for the Gaussian noise RNG.  Use the same value to
        reproduce identical noise realisations; set ``null`` for a random
        seed (different noise every run).  Default: 12345.
    """
    noise_choice: str = "SNR"
    fixed_snr: Optional[float] = None
    use_mean_snr: bool = False
    snr_correction: float = 0.0
    noise_seed: Optional[int] = 12345


# ---------------------------------------------------------------------------
# Atmosphere (one region, used for 1D, morning, evening, retrieval)
# ---------------------------------------------------------------------------

@dataclass
class AtmosphereRegionConfig:
    """Atmospheric model for one region (1D, morning, evening, retrieval).

    Parameters
    ----------
    species:
        pRT species identifiers, e.g. ``["H2", "He", "H2O", "CO"]``.
    use_easychem:
        Derive mass fractions from EasyChem given metallicity and C/O.
    metallicity_wrt_solar:
        log10 metallicity relative to solar (passed to EasyChem).
    carbon_to_oxygen_ratio:
        C/O ratio (passed to EasyChem).
    mass_fractions:
        Manual mass fractions (one per species).  Ignored when
        ``use_easychem=True``.
    mean_molecular_weight:
        Mean molecular weight in amu.
    reference_pressure_bar:
        Reference pressure level p0 in bar.
    cloud_pressure_bar:
        Cloud-top pressure in bar.  ``None`` = cloud-free.
    pressure_min_bar:
        Top of the pressure grid in bar.
    pressure_max_bar:
        Bottom of the pressure grid in bar.
    pressure_grid_size:
        Number of pressure levels.
    isothermal:
        Use a uniform temperature profile.
    isothermal_temperature_K:
        Temperature if ``isothermal=True``.  ``None`` = use T_equ.
    equilibrium_temperature_K:
        Equilibrium temperature for the Guillot profile.
    kappa_ir:
        Guillot profile IR opacity.
    gamma_guillot:
        Guillot profile gamma.
    two_point:
        Use a two-point (linear in log-P) temperature profile.
    two_point_pressures_bar:
        Two pressure anchor points [p_top, p_bottom] in bar.
    two_point_temperatures_K:
        Temperatures at the two pressure anchors in K.
    wind_velocity_kms:
        Bulk Doppler wind offset in km/s.
    """
    species: List[str] = field(default_factory=lambda: ["H2", "He", "H2O"])
    use_easychem: bool = True
    metallicity_wrt_solar: float = 0.0
    carbon_to_oxygen_ratio: float = 0.55
    mass_fractions: List[float] = field(default_factory=list)
    mean_molecular_weight: float = 2.33
    reference_pressure_bar: float = 1e-2
    cloud_pressure_bar: Optional[float] = None
    pressure_min_bar: float = 1e-6
    pressure_max_bar: float = 1e2
    pressure_grid_size: int = 100
    isothermal: bool = False
    isothermal_temperature_K: Optional[float] = None
    equilibrium_temperature_K: float = 1200.0
    kappa_ir: float = 0.01
    gamma_guillot: float = 0.4
    two_point: bool = False
    two_point_pressures_bar: List[float] = field(
        default_factory=lambda: [10**0.1, 10**-2.75]
    )
    two_point_temperatures_K: List[float] = field(
        default_factory=lambda: [1750.0, 520.0]
    )
    wind_velocity_kms: float = 0.0

    @property
    def vmr(self):
        """Mass fractions as a numpy array (one entry per species)."""
        import numpy as np
        arr = np.zeros(len(self.species), float)
        for i, mf in enumerate(self.mass_fractions):
            if i < len(arr):
                arr[i] = mf
        return arr


# ---------------------------------------------------------------------------
# Atmosphere (full simulation, 1D or with limb asymmetries)
# ---------------------------------------------------------------------------

@dataclass
class AtmosphereConfig:
    """Atmospheric configuration for the full simulation.

    Parameters
    ----------
    limb_asymmetries:
        Simulate morning and evening limb separately (Maguire+2024 style).
    limb_divisions:
        How to weight morning vs evening limb contributions over time.
        ``"gradual"``: smooth cubic transition, morning dominates ingress,
        equal mix at full transit, evening dominates egress (default, most
        realistic for typical hot Jupiters).
        ``"asymmetric"``: asymmetric, morning dominates the first quarter of
        full transit, cubic handover completes at mid-transit; calibrated for
        an ultra-hot Jupiter like WASP-76 b with extreme day-night contrast
        and poor heat redistribution.
        ``"simplified_step"``: hard step-function, pure morning during ingress,
        equal mix during full transit, pure evening during egress; use for
        reference tests and simplified scenarios where clean physical
        interpretation of each limb's contribution is needed.
    cc_with_true_model:
        Use the same model as the CCF template (ignores ``ccf_template``).
    planet_model:
        Injected planet atmosphere (1D case or combined limbs).
    ccf_template:
        Atmosphere for the CCF template.  Can differ from truth model.
    morning_day:
        Morning / day-side limb (used when ``limb_asymmetries=True``).
    morning_night:
        Morning / night-side limb.
    evening_day:
        Evening / day-side limb.
    evening_night:
        Evening / night-side limb.
    """
    limb_asymmetries: bool = False
    limb_divisions: str = "gradual"
    cc_with_true_model: bool = False
    planet_model: AtmosphereRegionConfig = field(
        default_factory=AtmosphereRegionConfig
    )
    ccf_template: AtmosphereRegionConfig = field(
        default_factory=AtmosphereRegionConfig
    )
    morning_day: Optional[AtmosphereRegionConfig] = None
    morning_night: Optional[AtmosphereRegionConfig] = None
    evening_day: Optional[AtmosphereRegionConfig] = None
    evening_night: Optional[AtmosphereRegionConfig] = None


# ---------------------------------------------------------------------------
# Tellurics
# ---------------------------------------------------------------------------

@dataclass
class TelluricConfig:
    """Telluric contamination model.

    Parameters
    ----------
    include_tellurics:
        Add telluric absorption to each simulated spectrum.
    use_full_skycalc:
        Use individually downloaded Skycalc spectra (one per airmass/PWV).
    use_accurate_airmass:
        Compute precise airmass from sky coordinates and observation time
        via Astropy.  If False, a simple parabolic model is used.  For
        fully synthetic ``different_nights`` runs this also places each
        night at its own observable transit epoch (target above 30 deg at
        night, searched from the reference epoch), so successive nights
        get genuinely different airmass curves and BERVs; with False the
        nights fall on consecutive orbits at ``T0 + n * P`` and use the
        parabolic model with ``airmass_limits`` /
        ``airmass_limits_per_night``.  Single-night synthetic runs
        currently always use the parabolic model.
    airmass_evolution:
        Simple airmass model: ``"up"``, ``"down"``, or ``"up_and_down"``.
    airmass_limits:
        [min, max] airmass for the simple geometric model.
    constant_pwv:
        Fix PWV over the night.
    pwv_mm:
        PWV in mm (used when ``constant_pwv=True``).  Used for all nights
        unless ``pwv_mm_per_night`` is set.
    pwv_mm_per_night:
        Per-night override for ``pwv_mm``.  List of floats, one per night.
        Each value must be on the SkyCalc grid.  Length must equal
        ``observation.n_nights``.  Ignored when ``different_nights=False``.
    reference_airmass:
        Airmass of the reference telluric spectrum used for SNR scaling.
    reference_telluric_file:
        Path to the reference telluric FITS file (used when
        ``use_full_skycalc=False``).
    mask_threshold:
        Telluric flux below this fraction is masked (e.g. 0.2 = 20 %).
    safety_window_pixels:
        Full pixel window to mask around each bad pixel (must be > 1).
    """
    include_tellurics: bool = True
    use_full_skycalc: bool = False
    use_accurate_airmass: bool = True
    airmass_evolution: str = "up_and_down"
    airmass_limits: List[float] = field(default_factory=lambda: [1.4, 1.7])
    airmass_limits_per_night: Optional[List[List[float]]] = None
    constant_pwv: bool = True
    pwv_mm: float = 10.0
    pwv_mm_per_night: Optional[List[float]] = None
    reference_airmass: float = 1.0
    reference_telluric_file: str = ""
    mask_threshold: float = 0.2
    safety_window_pixels: int = 7


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Data preparation and cleaning pipeline.

    Parameters
    ----------
    name:
        Named reduction recipe: ``"BL19"``, ``"ASL19"``,
        ``"Blain24"``, ``"Gibson22"``.
    sysrem_iterations:
        Number of SYSREM (PCA-like) passes.
    snr_mask_threshold:
        Mask pixels below this SNR value.
    prepare_template:
        Apply the same preparation steps to the CCF template as to the
        data to mimic signal distortions.
    optimize_sysrem_order_by_order:
        Find the optimal number of SYSREM iterations independently for
        each spectral order.  The criterion is set by
        ``optimize_criterion``.
    optimize_criterion:
        Criterion for per-order SYSREM optimisation.

        ``"DeltaSigma"`` (**recommended default**), model-independent:
        halts each order when the fractional decrease in residual
        standard deviation between consecutive passes falls below
        ``sysrem_delta_sigma_threshold``.  Described in Parker et al.
        (2025, MNRAS 538, 3263) and Peláez-Torres et al. (2026, A&A
        705, A256).

        ``"Maximum"`` and ``"Max_Diff"``, injection-recovery based:
        inject a planet signal and choose the iteration that maximises
        CCF S/N (``Maximum``) or its marginal gain per iteration
        (``Max_Diff``) respectively.  **When** ``kp_vrest_injection``
        is set to an offset position away from the real planet velocity
        (e.g. [Kp, +19] as in Cheverall et al. 2026), these criteria
        resemble the ΔCCF framework (Holmberg & Madhusudhan
        2022; Cheverall et al. 2023) and may present reduced biases,
        though they remain model-dependent.  Bias is more severe when
        the injection is placed at the real planet velocity, which
        allows noise at that position to inflate the recovered
        significance (Cabot et al. 2019; Cheverall et al. 2023).
    sysrem_delta_sigma_threshold:
        Fractional σ-drop threshold for the ``"DeltaSigma"`` halting
        criterion.  SYSREM stops when
        (σ_{i-1} - σ_i) / σ_{i-1} < threshold.
        Default 0.01 (1 %) following Parker et al. (2025) and
        Peláez-Torres et al. (2026).
    kp_vrest_injection:
        [Kp, Vrest] in km/s of the injected signal used for SYSREM
        optimisation (``"Maximum"`` / ``"Max_Diff"`` only).
    inject_scale_factor:
        Scale factor for the injected signal used in SYSREM optimisation.
    sysrem_robust_halt:
        Per-dataset SYSREM halt: stops all orders at the same
        iteration using a gradient-based plateau criterion.  Superseded
        by ``optimize_sysrem_order_by_order: true`` +
        ``optimize_criterion: "DeltaSigma"`` which applies the halt
        independently per order.
    """
    name: str = "BL19"
    sysrem_iterations: int = 4
    snr_mask_threshold: float = 10.0
    prepare_template: bool = True
    optimize_sysrem_order_by_order: bool = False
    optimize_criterion: str = "DeltaSigma"
    sysrem_delta_sigma_threshold: float = 0.01
    kp_vrest_injection: List[float] = field(default_factory=lambda: [0.0, 0.0])
    inject_scale_factor: float = 1.0
    sysrem_robust_halt: bool = False
    # Continuum-normalisation estimator for the Cheverall26 pipeline:
    #   "maxima", 2nd-order polynomial through the maxima of 80 bins
    #   "polyfit", per-exposure iterative 2nd-order polynomial fit to the
    #               continuum (rejecting absorption), as literally described in
    #               Cheverall et al. (2026) Sect. 2.3.
    continuum_method: str = "maxima"

    # detrend_method: detrending operator used to remove telluric/stellar
    #   systematics, "sysrem" (inverse-variance-weighted, Tamuz et al. 2005;
    #   default) or "pca" (unweighted principal-component subtraction; de Kok
    #   et al. 2013; Cheverall et al. 2023, who report minimal difference
    #   between the two).  Pipeline-agnostic: any preparing pipeline that
    #   detrends honours this switch.  The component count is sysrem_iterations.
    detrend_method: str = "sysrem"


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

@dataclass
class CrossCorrelationConfig:
    """Cross-correlation function (CCF) settings.

    Parameters
    ----------
    velocity_max_kms:
        CCF range: ``-velocity_max_kms`` to ``+velocity_max_kms`` (km/s).
    velocity_step_kms:
        Velocity grid step in km/s.  ``None`` = mean pixel velocity step.
    kp_max_kms:
        Maximum Kp to explore in the Kp-Vsys map (km/s).
    noise_velocity_max_kms:
        Velocity range used to estimate CCF noise std (km/s).
    snr_exclude_kms:
        Half-width of in-trail region excluded when computing CCF noise
        std (km/s).
    snr_noise_source:
        Where the noise std of the Kp-Vsys S/N map is measured.
        ``"peak_row"`` (default): per Kp row, from the velocities more
        than ``snr_exclude_kms`` away from that row's maximum (legacy
        behaviour).  For strong detections this region contains the
        signal's own correlation wings, which co-add coherently across
        nights and saturate the reported S/N.  ``"signal_free_rows"``:
        one global std from the rows more than
        ``snr_noise_kp_exclude_kms`` away from the detected peak's Kp
        (and beyond ``snr_exclude_kms`` of the peak velocity), a region
        with the same noise statistics but no signal-locked structure.
    snr_noise_kp_exclude_kms:
        Half-width in Kp (km/s) of the region excluded around the
        detected peak when ``snr_noise_source="signal_free_rows"``.
        The signal's wing structure decays to the few-per-cent level
        beyond ~100-150 km/s from the peak Kp.
    in_trail_left_right:
        Half-width (in pixels) of the in-trail window for significance
        metrics.  Total width = 2 * in_trail_left_right + 1.
    normalized:
        Normalize the CCF to a Pearson correlation coefficient.
    use_inverse_variance_weighting:
        Weight the CCF by per-pixel inverse variance.
    ccf_snr:
        Compute the CCF S/N significance metric.
    welch_ttest:
        Compute the Welch t-test significance metric.
    cc_metric:
        Compute the CC significance metric.
    all_significance_metrics:
        Compute S/N and Welch t-test significance maps simultaneously,
        saving both Kp-Vsys plots and 1D CCF plots.  If
        ``retrieval.enabled=false`` (default) only these two metrics are
        produced and Block 9 is skipped with an informational message.
        Set ``retrieval.enabled=true`` to additionally run Bayesian
        retrieval (substantially slower).
    ssim_metric:
        Compute the SSIM structural similarity metric (experimental).
    study_velocity_ranges:
        Study how significance changes with in/out-of-trail intervals.
    plot_ccf_xstep:
        Step of the CCF horizontal axis in plots (km/s).
    """
    velocity_max_kms: float = 325.0
    velocity_step_kms: Optional[float] = 1.0
    kp_max_kms: float = 320.0
    noise_velocity_max_kms: float = 250.0
    snr_exclude_kms: float = 25.0
    snr_noise_source: str = "peak_row"
    snr_noise_kp_exclude_kms: float = 120.0
    in_trail_left_right: int = 2
    normalized: bool = True
    use_inverse_variance_weighting: bool = True
    ccf_snr: bool = True
    welch_ttest: bool = False
    cc_metric: bool = True
    all_significance_metrics: bool = False
    ssim_metric: bool = False
    study_velocity_ranges: bool = False
    plot_ccf_xstep: float = 50.0
    # CCF kernel form (weighted, large-instrument path):
    #   "normalized", inverse-variance-weighted Pearson CCF (default)
    #   "matched_filter", un-normalised Σ R·M/E² (Nortmann et al. 2024 Eq. 1)
    ccf_kernel: str = "normalized"
    # Error E used in the CCF inverse-variance weighting (1/E^2):
    #   "propagated", per-pixel uncertainty propagated through the pipeline
    #   "mad_residual", per-wavelength-channel MAD of the time-series
    #                    residuals (Gibson+20; Nortmann+24; Cheverall+26)
    ccf_error_estimate: str = "propagated"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Bayesian atmospheric retrieval settings.

    Parameters
    ----------
    enabled:
        Run a retrieval after the CCF analysis.
    sampler:
        ``"nested_sampling"`` (MultiNest) or ``"mcmc"`` (emcee).
    log_likelihood:
        Log-likelihood formulation: ``"BL19"`` (matched filter, noise-free),
        ``"Blain24"`` (chi-squared with propagated uncertainties),
        ``"Gibson22"`` (chi-squared with noise hyperparameter β; requires
        ``dimensionality="1D_Gibson22"``).
    dimensionality:
        Retrieval parameter space.  Supported values and free parameters:

        ``"1D"``, log10(X_H2O), Kp, T_eq, v_wind  [use with BL19/Blain24]
        ``"1D_Gibson22"``, same + β noise scaling           [use with Gibson22 only]
        ``"1D_CtoO_met"``, C/O ratio, log10(Z/Z☉) via easyCHEM
        ``"1D_extended"``, log10(X_H2O,CH4,NH3,CO,CO2,HCN) + Kp + T_eq + v_wind + β
        ``"1D_extended_fast"``, same abundances + T_eq only (Kp/v_wind fixed)
        ``"2D"``, morning/evening limb asymmetry

    retrieval_choice:
        1 = single night (or the only night),
        2 = nights with max, min, and mean CCF S/N,
        3 = nights with max and min CCF S/N,
        4 = all nights combined,
        5 = all nights one by one,
        6 = time-resolved (ingress / first-half / second-half / egress).
    time_resolution:
        Divide the transit into equal time bins (requires
        ``retrieval_choice=6``).
    time_resolution_step:
        Number of exposures per time bin (``time_resolution=True``).
    live_points:
        MultiNest live points.  200 is sufficient for quick tests; use
        400 to 1000 for publication-quality results.
    constant_efficiency_mode:
        MultiNest constant-efficiency mode.  Keep False (recommended).
    n_walkers:
        emcee ensemble walkers.
    n_steps:
        emcee steps per walker.
    burnin:
        emcee burn-in steps to discard.
    pressure_n_levels:
        Number of pressure levels in the pRT forward model grid
        (log-spaced from ``pressure_log_min`` to ``pressure_log_max``).
    pressure_log_min:
        Minimum pressure exponent (bar), default -6 → 10⁻⁶ bar.
    pressure_log_max:
        Maximum pressure exponent (bar), default  2 → 100 bar.
    prior_bounds:
        Dict mapping dimensionality → list of [low, high] prior bounds,
        one pair per free parameter in the order listed above.
    atmosphere:
        Retrieval forward-model atmosphere.  If ``null``, falls back to the
        planet model settings.  Override here to set retrieval-specific
        values for ``mean_molecular_weight``, ``reference_pressure_bar``,
        ``isothermal``, ``kappa_ir``, ``gamma_guillot``, ``two_point``,
        ``two_point_pressures_bar``, ``two_point_temperatures_K``, etc.
        These correspond to the ``MMW_ret``, ``p0_ret``,
        ``isothermal_ret``, ``Kappa_IR_ret``, ``Gamma_ret``,
        ``two_point_T_ret``, ``p_points_ret``, ``t_points_ret`` parameters.
    """
    enabled: bool = False
    sampler: str = "nested_sampling"
    log_likelihood: str = "Blain24"
    dimensionality: str = "1D_CtoO_met"
    retrieval_choice: int = 1
    time_resolution: bool = False
    time_resolution_step: int = 20
    live_points: int = 200
    constant_efficiency_mode: bool = False
    # Likelihood error model (Cheverall26 pipeline): "propagated" (default, divide
    # by the pipeline-propagated σ) or "mad_residual" (σ_j = 1.4826·MAD of the
    # detrended residuals per wavelength channel over time, time-independent;
    # the Cheverall et al. 2026 / Gibson-style choice).  Only used by Cheverall26.
    error_model: str = "propagated"
    n_walkers: int = 32
    n_steps: int = 5000
    burnin: int = 1000
    pressure_n_levels: int = 100
    pressure_log_min: float = -6.0
    pressure_log_max: float = 2.0
    prior_bounds: Dict[str, List[Any]] = field(default_factory=lambda: {
        "1D":               [[-8.0, 0.0], [85.0, 200.0],
                             [400.0, 1500.0], [-25.0, 25.0]],
        "1D_Gibson22":           [[-8.0, 0.0], [85.0, 200.0],
                             [400.0, 1500.0], [-25.0, 25.0], [0.01, 100.0]],
        "1D_CtoO_met":      [[0.0, 2.0], [-2.5, 2.5]],
        "1D_extended":      [[-8.0, 0.0], [-8.0, 0.0], [-8.0, 0.0],
                             [-8.0, 0.0], [85.0, 200.0],
                             [400.0, 1500.0], [-25.0, 25.0], [0.01, 100.0]],
        "1D_extended_fast": [[-8.0, 0.0], [-8.0, 0.0], [-8.0, 0.0],
                             [-8.0, 0.0], [400.0, 1500.0]],
        "2D":               [[-8.0, 0.0], [-8.0, 0.0], [85.0, 200.0],
                             [400.0, 1500.0], [400.0, 1500.0], [-20.0, 20.0]],
    })
    atmosphere: Optional[AtmosphereRegionConfig] = None

    # ── Spectrum emulator (EXPERIMENTAL, see src/exoplore/retrieval/emulator.py) ──
    # Replaces call_pRT with a PCA+MLP surrogate trained on pre-computed spectra.
    # Benchmarked on CARMENES NIR order 23, noiseless, Blain24, 1D:
    #   speedup 1.9× (Block 9: 11.65 min vs 22.33 min), but posteriors 2-3×
    #   broader and T_eq biased. NOT recommended for scientific retrievals.
    use_emulator: bool = False
    emulator_path: str = ""

    # ── Likelihood emulator (EXPERIMENTAL, see src/exoplore/retrieval/likelihood_emulator.py) ──
    # Trains a MLP to map θ → logL(θ|data) directly from true pRT evaluations.
    # Benchmarked on the same setup: val MSE = 1.93 logL², log10X biased +1.45σ,
    # total runtime 37 min vs 22 min, slower AND less accurate.
    # Root cause: posterior occupies ~0.002% of prior; sparse uniform sampling
    # cannot map the likelihood surface accurately. NOT recommended.
    use_likelihood_emulator: bool = False
    likelihood_emulator_n_samples: int = 3000


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class StatisticsConfig:
    """Multi-night statistical analysis options.

    Parameters
    ----------
    enabled:
        Run a statistical analysis of final signal significances across
        multiple nights.
    noise_study:
        Study the distribution of noise-only CCF significances.
    noise_correlations:
        Compute noise correlations between nights.
    stack_group_size:
        Co-add nights in random groups of this size (e.g. 1000 groups of
        N nights).  ``None`` = no stacking.
    detectability_maps:
        Run the code recursively to produce detectability maps as a
        function of metallicity and C/O (requires cluster mode).
    """
    enabled: bool = False
    noise_study: bool = False
    noise_correlations: bool = False
    stack_group_size: Optional[int] = None
    detectability_maps: bool = False


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

@dataclass
class PlottingConfig:
    """Diagnostic plot settings.

    Parameters
    ----------
    pipeline_steps_order:
        Absolute spectral-order index to display in the pipeline-steps
        diagnostic plot.  If the requested order was not processed in this
        run (for example order 44 when the instrument has only 28 orders, or
        an order excluded by ``instrument.order_indices``), the first
        processed order is used instead.  Default ``null`` uses the first
        processed order.
    pipeline_steps_xlim_um:
        Wavelength window [lo, hi] in µm to zoom into.  For the SYSREM
        waterfall every panel uses this window; for the polynomial four-panel
        it applies to the 1-D spectrum panel only (the 2-D panels show the
        full order).  If the window does not fall within the displayed order,
        the full order is shown instead.  Set to ``null`` for the full order.
        Default [1.4862, 1.4890] is a water-band window present in CARMENES
        NIR (around order 22) and ANDES.
    pipeline_steps_sysrem_iterations:
        Which SYSREM iterations to show as intermediate panels in the
        pipeline-steps diagnostic plot.  Used ONLY for SYSREM-based
        pipelines (ASL19, Gibson22), where the cleaning proceeds iteration
        by iteration; the panels let you watch the common-mode systematics
        being removed.  Ignored by the polynomial pipelines (BL19, Blain24),
        which have no iterations and show a single residual panel instead.
        Each value is a 1-based iteration index (1 = after the first SYSREM
        pass).  Default [1, 5] shows an early and a late stage.
    """
    pipeline_steps_order: Optional[int] = None
    pipeline_steps_xlim_um: Optional[List[float]] = field(
        default_factory=lambda: [1.4862, 1.4890]
    )
    pipeline_steps_sysrem_iterations: List[int] = field(
        default_factory=lambda: [1, 5]
    )


# ---------------------------------------------------------------------------
# Output (which matrices to write to disk)
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    """Which per-order spectral matrices to write to disk.

    A full run can write several GB of per-order matrices, but most are only
    needed for specific kinds of re-analysis, so each can be switched off to
    save disk space.  Small metadata (masks, velocity grids, the Kp-Vsys map,
    phase, Julian dates) is ALWAYS written regardless of these flags.  The
    sizes quoted are for the 76-order Tutorial 1 run and scale with
    orders x exposures.

    The defaults keep everything needed to re-run cross-correlations with other
    templates, run retrievals, and use the Gibson22 log-likelihood, while
    dropping the pure-diagnostic matrices (about 1.2 GB instead of 2.2 GB for
    that run).  Set all to False for a minimal result (the Kp-Vsys map plus
    metadata, a few MB).

    Parameters
    ----------
    save_mat_res:
        Prepared residual data.  Needed to re-run CCFs with a different
        template and for retrievals.  (~334 MB)
    save_mat_back:
        Per-exposure CCF template.  Needed to re-run CCFs and for retrievals.
        (~298 MB)
    save_ccf_store:
        Per-order cross-correlation functions.  Needed to rebuild Kp-Vsys maps
        without re-running the CCF.  (~141 MB)
    save_propag_noise:
        Propagated per-pixel noise.  Needed for inverse-variance CCF weighting
        and for the retrieval log-likelihood.  (~416 MB)
    save_U_sysrem:
        SYSREM basis vectors.  Needed to filter the model for the Gibson22
        log-likelihood (negligible size).  (~4 KB)
    save_mat_cc:
        Noiseless injected model matrix.  Diagnostic only.  (~181 MB)
    save_mat_noise:
        The noise realisation.  Diagnostic only.  (~411 MB)
    save_std_noise:
        Per-pixel noise standard deviation.  Diagnostic only.  (~419 MB)
    """
    save_mat_res: bool = True
    save_mat_back: bool = True
    save_ccf_store: bool = True
    save_propag_noise: bool = True
    save_U_sysrem: bool = True
    save_mat_cc: bool = False
    save_mat_noise: bool = False
    save_std_noise: bool = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """Output and input path configuration.

    Parameters
    ----------
    output_root:
        Root directory for all simulation outputs.
    planet_parameter_dir:
        Directory containing planet JSON parameter files.
    prt_input_data:
        Path to the petitRADTRANS ``input_data/`` directory.
    phoenix_wave_file:
        Path to the PHOENIX wavelength FITS file (dayside sims).
    phoenix_flux_file:
        Path to the PHOENIX flux FITS file (dayside sims).
    skycalc_dir:
        Directory containing Skycalc telluric input/output files.
        Within this, subdirectories per ``flag_event`` are expected.
    inputs_dir:
        Explicit path to the pre-computed instrument inputs directory
        (ETC FITS files, Skycalc telluric files, etc.).  If set, this
        overrides the auto-constructed path
        ``output_root/<planet>/<instrument>/<event>/inputs/``.
        Point this at an existing ``inputs/`` folder to reuse
        instrument files without regenerating them.
    """
    output_root: str = "outputs"
    planet_parameter_dir: str = "planet_params"
    prt_input_data: str = ""
    phoenix_wave_file: str = ""
    phoenix_flux_file: str = ""
    skycalc_dir: str = ""
    inputs_dir: str = ""


# ---------------------------------------------------------------------------
# Top-level SimulationConfig
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Complete description of one EXoPLORE simulation.

    Compose all sub-configs here.  Serialize / deserialize with
    :meth:`to_json` and :meth:`from_json`.

    Examples
    --------
    Load from a JSON file::

        cfg = SimulationConfig.from_json("configs/hd189733b_andes_transit_clean.json")

    Access a scientific parameter::

        print(cfg.cross_correlation.velocity_max_kms)
        # 325.0
    """
    planet: PlanetConfig = field(default_factory=PlanetConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    tellurics: TelluricConfig = field(default_factory=TelluricConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cross_correlation: CrossCorrelationConfig = field(
        default_factory=CrossCorrelationConfig
    )
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    timing: bool = False

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the config to a JSON file."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        """Reconstruct from a plain dict (as loaded from JSON)."""

        def _region(sub: dict) -> AtmosphereRegionConfig:
            return AtmosphereRegionConfig(**{
                k: v for k, v in sub.items()
                if k in AtmosphereRegionConfig.__dataclass_fields__
            })

        def _atmos(sub: dict) -> AtmosphereConfig:
            kwargs = dict(sub)
            for key in ("planet_model", "ccf_template"):
                if key in kwargs and isinstance(kwargs[key], dict):
                    kwargs[key] = _region(kwargs[key])
            for key in ("morning_day", "morning_night",
                        "evening_day", "evening_night"):
                if key in kwargs:
                    if isinstance(kwargs[key], dict):
                        kwargs[key] = _region(kwargs[key])
                    # None values stay None
            return AtmosphereConfig(**{
                k: v for k, v in kwargs.items()
                if k in AtmosphereConfig.__dataclass_fields__
            })

        def _retrieval(sub: dict) -> RetrievalConfig:
            kwargs = dict(sub)
            atm_raw = kwargs.pop("atmosphere", None)
            if isinstance(atm_raw, dict):
                kwargs["atmosphere"] = _region(atm_raw)
            return RetrievalConfig(**{
                k: v for k, v in kwargs.items()
                if k in RetrievalConfig.__dataclass_fields__
            })

        def _safe(cls_, sub: dict):
            """Construct dataclass ignoring unknown keys (forward compat)."""
            return cls_(**{
                k: v for k, v in sub.items()
                if k in cls_.__dataclass_fields__
            })

        return cls(
            planet=_safe(PlanetConfig, d.get("planet", {})),
            instrument=_safe(InstrumentConfig, d.get("instrument", {})),
            observation=_safe(ObservationConfig, d.get("observation", {})),
            noise=_safe(NoiseConfig, d.get("noise", {})),
            atmosphere=_atmos(d.get("atmosphere", {})),
            tellurics=_safe(TelluricConfig, d.get("tellurics", {})),
            pipeline=_safe(PipelineConfig, d.get("pipeline", {})),
            cross_correlation=_safe(
                CrossCorrelationConfig, d.get("cross_correlation", {})),
            retrieval=_retrieval(d.get("retrieval", {})),
            statistics=_safe(StatisticsConfig, d.get("statistics", {})),
            plotting=_safe(PlottingConfig, d.get("plotting", {})),
            output=_safe(OutputConfig, d.get("output", {})),
            paths=_safe(PathConfig, d.get("paths", {})),
            timing=bool(d.get("timing", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        """Load a :class:`SimulationConfig` from a JSON file."""
        with open(path) as fh:
            cfg = cls.from_dict(json.load(fh))
        obs = cfg.observation
        if obs.exposure_time_seconds_per_night is not None:
            if not obs.different_nights:
                raise ValueError(
                    "exposure_time_seconds_per_night requires different_nights: true"
                )
            if len(obs.exposure_time_seconds_per_night) != obs.n_nights:
                raise ValueError(
                    f"exposure_time_seconds_per_night has "
                    f"{len(obs.exposure_time_seconds_per_night)} entries "
                    f"but n_nights={obs.n_nights}"
                )
        return cfg

    def __str__(self) -> str:
        cfg = self
        lines = [
            "SimulationConfig",
            "----------------",
            f"  Planet              : {cfg.planet.name}",
            f"  Instrument          : {cfg.instrument.name}",
            f"  Event               : {cfg.observation.event_type}",
            f"  Nights              : {cfg.observation.n_nights}",
            f"  Atmosphere          : 1D={not cfg.atmosphere.limb_asymmetries}",
            f"  Species (planet)    : {cfg.atmosphere.planet_model.species}",
            f"  EasyChem            : {cfg.atmosphere.planet_model.use_easychem}",
            f"  C/O                 : {cfg.atmosphere.planet_model.carbon_to_oxygen_ratio}",
            f"  Metallicity (log Z) : {cfg.atmosphere.planet_model.metallicity_wrt_solar}",
            f"  Pipeline            : {cfg.pipeline.name}",
            f"  SYSREM iterations   : {cfg.pipeline.sysrem_iterations}",
            f"  CCF v_max (km/s)    : {cfg.cross_correlation.velocity_max_kms}",
            f"  Telluric mask       : {cfg.tellurics.mask_threshold}",
            f"  Retrieval           : {cfg.retrieval.enabled}",
            f"  Output root         : {cfg.paths.output_root}",
        ]
        return "\n".join(lines)
