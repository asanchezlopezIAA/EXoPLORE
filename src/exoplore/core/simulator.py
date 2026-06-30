"""
exoplore.core.simulator
=======================

High-level simulation orchestrator.

:class:`ExoploreSimulator` takes a :class:`~exoplore.config.SimulationConfig`
and runs the full simulation workflow.

Block status
------------
- Block 1  Setup & instrument loading        : implemented (clean)
- Block 2  Event, timing, phase, airmass     : implemented (clean)
- Block 3  Atmosphere forward model (pRT)    : implemented (clean)
- Block 4  Stellar model & injection setup   : implemented (clean)
- Block 5  Tellurics, noise & SYSREM pipeline : implemented (clean)
- Block 6  CCF computation                   : implemented (clean)
- Block 7  CCF statistics / Kp-Vsys maps     : implemented (clean)
- Block 8  Save matrices & CCF products      : implemented (clean)
- Block 9  Retrieval                         : implemented (clean)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from exoplore.config.models import SimulationConfig
from exoplore.io.paths import build_output_tree


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from exoplore.atmosphere.prt import call_pRT, call_pRT_limbs, convolve
from exoplore.atmosphere.winds import (
    atmospheric_scale_height,
    convolve_spectrum_with_kernel,
    get_sflimbs,
    planet_rot_vel,
    rotation_kernel_Maguire24,
    wind_broadening_triangular_kernel,
)
from exoplore.ccf.compute import (
    call_ccf_literature,
    call_ccf_numba,
    call_ccf_numba_par_weighted,
    call_ccf_numba_par_matched_filter,
    call_ccf_numba_par_weighted_ordbord_opt,
    get_max_CCF_peak,
    get_shifted_ccf_matrix,
)
from exoplore.ccf.statistics import Welch_ttest_map
from exoplore.instruments.andes import (
    From1OrderTo1Detector,
    get_WaveGrid,
    pixel_snr_one_order,
)
from exoplore.io.stellar import LoadPhoenix, get_stellar_matrix, spec_to_mat_fraction
from exoplore.io.utils import find_nearest, format_number, save_compressed
from exoplore.observation.airmass import PWV_handling, get_airmass
from exoplore.observation.noise import add_throughput
from exoplore.observation.timing import (
    block_parameter,
    dayside_fraction,
    find_nights_with_extrema,
    get_event,
)
from exoplore.observation.velocity import get_V, get_V_eccentric
from exoplore.pipelines.masking import Correct_NaN, Remove_Outliers, merge_masks, _merge_masks
from exoplore.pipelines.prepare import injection, preparing_pipeline
from exoplore.pipelines.sysrem import (
    SYSREM_filtering_projector,
    SYSREM_filtering_projector_singleorder,
    filter_model_singleorder,
    get_SYSREM_its_ordbyord,
)
from exoplore.pipelines.tellurics import Load_Telluric_Transmittances
from exoplore.plotting.kpvsys import plot_1D_CCF, plot_Kp_Vrest
from exoplore.plotting.matrices import CCF_matrix_ERF
from exoplore.analysis.stats import statistical_study, plot_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_number(x: float) -> str:
    """Format a float for use in simulation names.

    Examples: 1.0 → "1",  1.2 → "1p20",  0.5 → "0p50"
    """
    if int(x) == x:
        return str(int(x))
    integer_part = int(x)
    decimal_part = int(round((x - integer_part), 1) * 100)
    return f"{integer_part}p{decimal_part:02d}"


def build_simulation_name(cfg: SimulationConfig) -> str:
    """Build the simulation run name.

    Matches the logic at lines 1073-1096 of EXOSIMS_2p0_ANDES.py::

        f"{preparing_pipeline}_{signal_flag}_{PCA_opt}{Kp_flag}{Opt_flag}"
        f"{n_nights}nights_{signif_flag}_{stack_flag}_{real_flag}_{noise_flag}_"
        f"stdnoisex{format_number(Noise_scaling_factor)}"
    """
    obs  = cfg.observation
    pipe = cfg.pipeline
    ccf  = cfg.cross_correlation
    stats = cfg.statistics

    signal_flag = "withsignal" if obs.simulate_planet else "withoutsignal"

    # Significance metric flag
    if ccf.all_significance_metrics:
        signif_flag = "AllMetrics"
    elif ccf.ccf_snr:
        signif_flag = "SNR"
    elif ccf.welch_ttest:
        signif_flag = "Welch"
    else:
        signif_flag = ""

    # Stacking flag
    sg = stats.stack_group_size
    stack_flag = f"comb{sg}" if sg is not None else "comb1"

    # Real vs simulated data
    real_flag = "realdata" if obs.use_real_data else "simdata"

    # Noise flag
    noise_flag = "noiseless" if obs.noiseless else "noisy"

    # Detrending flags. Only SYSREM/PCA-based pipelines carry a detrending
    # token in the run name. BL19 and Blain24 are polynomial fitting pipelines
    # and use neither SYSREM nor PCA, so they carry no such token.
    if pipe.name not in {"ASL19", "Gibson22", "Cheverall26"}:
        pca_flag = kp_flag = opt_flag = ""
    elif pipe.optimize_sysrem_order_by_order:
        pca_flag = "SYSREMopt_"
        if pipe.optimize_criterion == "DeltaSigma":
            kp_flag  = "DeltaSigma_"
            opt_flag = ""
        else:
            kp_flag  = "planetpos_" if pipe.kp_vrest_injection == [0.0, 0.0] \
                       else "otherpos_"
            opt_flag = "maximum_" if pipe.optimize_criterion == "Maximum" \
                       else "MaxDiff_"
    else:
        # Fixed-N detrending (no per-order optimisation): encode the number
        # of SYSREM/PCA components in the name so runs with different N get
        # distinct output folders rather than overwriting one another.
        pca_flag = f"N{pipe.sysrem_iterations}_"
        kp_flag = opt_flag = ""

    sf_str = _format_number(obs.noise_scaling_factor)

    return (
        f"{pipe.name}_{signal_flag}_"
        f"{pca_flag}{kp_flag}{opt_flag}"
        f"{obs.n_nights}nights_"
        f"{signif_flag}_{stack_flag}_{real_flag}_"
        f"{noise_flag}_"
        f"stdnoisex{sf_str}"
    )


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationSummary:
    """Human-readable summary of a configured simulation."""
    planet_name: str
    instrument: str
    event_type: str
    n_nights: int
    pipeline: str
    sysrem_iterations: int
    species: list
    use_easychem: bool
    carbon_to_oxygen: float
    metallicity: float
    velocity_max_kms: float
    retrieval_enabled: bool
    output_root: str

    def __post_init__(self):
        # Backward-compatibility alias
        object.__setattr__(self, "planet", self.planet_name)

    # Pipeline citation strings
    _PIPELINE_REFS = {
        "BL19":         "BL19          (Brogi & Line 2019, AJ, 157, 114)",
        "Blain24":      "Blain24       (Blain, Sanchez-Lopez & Molliere 2024, AJ, 167, 179)",
        "ASL19":        "ASL19         (Sanchez-Lopez et al. 2019, A&A, 630, A53)",
        "Gibson22":     "Gibson22      (Gibson et al. 2022, MNRAS, 512, 4618)",
        "Cheverall26":  "Cheverall26   (Cheverall et al. 2026, MNRAS, IGRINS pipeline)",
    }
    _SYSREM_PIPELINES = {"ASL19", "Gibson22", "Cheverall26"}

    def __str__(self) -> str:
        pipeline_str = self._PIPELINE_REFS.get(self.pipeline, self.pipeline)
        lines = [
            "",
            "  EXoPLORE Simulation Summary",
            "  ============================",
            f"  Planet              : {self.planet_name}",
            f"  Instrument          : {self.instrument}",
            f"  Event               : {self.event_type}",
            f"  Nights              : {self.n_nights}",
            f"  Pipeline            : {pipeline_str}",
        ]
        if self.pipeline in self._SYSREM_PIPELINES:
            lines.append(f"  SYSREM iterations   : {self.sysrem_iterations}")
        lines += [
            f"  Species             : {self.species}",
            f"  EasyChem            : {self.use_easychem}",
            f"  C/O                 : {self.carbon_to_oxygen}",
            f"  Metallicity (log Z) : {self.metallicity}",
            f"  CCF v_max (km/s)    : {self.velocity_max_kms}",
            f"  Retrieval           : {self.retrieval_enabled}",
            f"  Output root         : {self.output_root}",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class ExoploreSimulator:
    """High-level interface for running an EXoPLORE simulation.

    Parameters
    ----------
    config:
        A fully populated :class:`~exoplore.config.SimulationConfig`.

    Examples
    --------
    >>> from exoplore.config import SimulationConfig
    >>> from exoplore.core import ExoploreSimulator
    >>> cfg = SimulationConfig.from_json("configs/hd189733b_andes_transit_clean.json")
    >>> sim = ExoploreSimulator(cfg)
    >>> print(sim.summary())
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        cfg = self.config

        if not cfg.planet.name:
            raise ValueError("planet.name must not be empty.")

        if cfg.observation.event_type not in ("transit", "dayside"):
            raise ValueError(
                f"observation.event_type must be 'transit' or 'dayside', "
                f"got: {cfg.observation.event_type!r}"
            )

        if cfg.observation.n_nights < 1:
            raise ValueError("observation.n_nights must be >= 1.")

        if cfg.observation.exposure_time_seconds <= 0:
            raise ValueError("observation.exposure_time_seconds must be positive.")

        if cfg.cross_correlation.velocity_max_kms <= 0:
            raise ValueError("cross_correlation.velocity_max_kms must be positive.")

        vstep = cfg.cross_correlation.velocity_step_kms
        if vstep is not None and vstep <= 0:
            raise ValueError("cross_correlation.velocity_step_kms must be positive.")

    # ------------------------------------------------------------------
    # Simulation name
    # ------------------------------------------------------------------

    def simulation_name(self) -> str:
        """Return the run name."""
        return build_simulation_name(self.config)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> SimulationSummary:
        """Alias for :meth:`summary`."""
        return self.summary()

    def summary(self) -> SimulationSummary:
        cfg = self.config
        return SimulationSummary(
            planet_name=cfg.planet.name,
            instrument=cfg.instrument.name,
            event_type=cfg.observation.event_type,
            n_nights=cfg.observation.n_nights,
            pipeline=cfg.pipeline.name,
            sysrem_iterations=cfg.pipeline.sysrem_iterations,
            species=cfg.atmosphere.planet_model.species,
            use_easychem=cfg.atmosphere.planet_model.use_easychem,
            carbon_to_oxygen=cfg.atmosphere.planet_model.carbon_to_oxygen_ratio,
            metallicity=cfg.atmosphere.planet_model.metallicity_wrt_solar,
            velocity_max_kms=cfg.cross_correlation.velocity_max_kms,
            retrieval_enabled=cfg.retrieval.enabled,
            output_root=cfg.paths.output_root,
        )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    def output_dirs(
        self,
        simulation_name: Optional[str] = None,
        create: bool = False,
    ) -> dict:
        """Return (and optionally create) the output directory tree."""
        if simulation_name is None:
            simulation_name = self.simulation_name()
        return build_output_tree(self.config, simulation_name, create=create)

    # ------------------------------------------------------------------
    # run(), Block 1: Setup, directories, instrument loading
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full simulation.

        Currently implements Block 1 (setup and instrument loading).
        Subsequent blocks will be added incrementally.

        Block 1 covers:
          - Build the simulation name
          - Create all output directories
          - Load planet parameters
          - Call Load_Instrumental_Info → resolve, file paths, gaps
          - Load wavelength grid (get_WaveGrid)
          - Optionally average SNR over time (Use_Mean_SNR)
          - Load PHOENIX stellar spectrum (LoadPhoenix)
        """

        # ----------------------------------------------------------------
        # Block 1a, setup, simulation name, output directories
        # ----------------------------------------------------------------
        import time as _time
        cfg = self.config
        _t_run_start = _time.time()
        _t_blocks = {}   # timing store: block_name -> elapsed_s
        sim_name = self.simulation_name()

        print(f"\n  Running simulation: {sim_name}")

        # Create output directory tree
        from exoplore.io.paths import build_output_tree
        dirs = build_output_tree(cfg, sim_name, create=True)

        # ----------------------------------------------------------------
        # Block 1b, planet parameters
        # ----------------------------------------------------------------
        from exoplore.planets import load_planet

        planet_file = cfg.planet.parameter_file
        if not planet_file:
            planet_file = str(
                Path(cfg.paths.planet_parameter_dir) / f"{cfg.planet.name}.json"
            )

        planet_path = Path(planet_file)
        if not planet_path.exists():
            raise FileNotFoundError(
                f"Planet parameter file not found: {planet_path}\n"
                f"  Set planet.parameter_file in your config, or place "
                f"{cfg.planet.name}.json in {cfg.paths.planet_parameter_dir}/"
            )

        planet = load_planet(planet_path)
        print(f"  Planet          : {planet.name}  "
              f"(Kp={planet.kp_kms:.1f} km/s, "
              f"T14={planet.transit_duration_hours:.2f} h)")

        _pre_h  = (cfg.observation.pre_event_hours
                   if cfg.observation.pre_event_hours > 0
                   else planet.transit_duration_hours / 2.0)
        _post_h = (cfg.observation.post_event_hours
                   if cfg.observation.post_event_hours > 0
                   else planet.transit_duration_hours / 2.0)

        # ----------------------------------------------------------------
        # Block 1c, instrument info and wavelength grid
        # ----------------------------------------------------------------

        # Set pRT input data path if configured
        if cfg.paths.prt_input_data:
            os.environ["pRT_input_data_path"] = cfg.paths.prt_input_data

        kp_range = np.arange(-cfg.cross_correlation.kp_max_kms,
                             cfg.cross_correlation.kp_max_kms + 1)

        # Load instrument resolving power and file locations via the dispatcher
        from exoplore.instruments import load_instrument_v2, get_WaveGrid_v2
        inst = load_instrument_v2(cfg)
        n_orders_total = inst.n_orders_total

        # Compute order_selection from cfg (mirrors what Block 3b does)
        _oi_b1 = cfg.instrument.order_indices
        _order_sel_b1 = (np.arange(n_orders_total)
                         if not _oi_b1 else np.asarray(_oi_b1))
        _n_orders_b1  = len(_order_sel_b1)

        print(f"  Orders          : {_n_orders_b1}  "
              f"(indices {_order_sel_b1[0]}, "
              f"{_order_sel_b1[-1]})")
        print(f"  Resolving power : R = {inst.res:.0f}")

        # Determine whether this is an ANDES YJHK full-band run
        _is_andes_yjhk = cfg.instrument.name in ("ANDES_YJHK", "ANDES")

        # Load wavelength grid (and SNR / JD / airmass reference arrays)
        _wg = get_WaveGrid_v2(cfg, inst, n_orders_total)
        _wg = _wg + (None,) * (7 - len(_wg))   # normalise to 7-tuple
        wave_star, n_pixels, sig_og, snr_og, JD_og, airmass_og, wave_mid_og = _wg

        # Average SNR over time if requested (non-ANDES instruments)
        _is_andes_any = cfg.instrument.name.startswith("ANDES")
        if not _is_andes_any:
            if (snr_og is not None
                    and not cfg.observation.use_real_data
                    and not cfg.observation.different_nights):
                if snr_og.ndim == 3 and cfg.noise.use_mean_snr:
                    snr_og = np.mean(snr_og, axis=0)

        print(f"  Wavelength grid : {n_pixels} pixels per order")

        # ----------------------------------------------------------------
        # Block 1d, PHOENIX stellar spectrum
        # ----------------------------------------------------------------
        _pw = cfg.paths.phoenix_wave_file or None
        _pf = cfg.paths.phoenix_flux_file or None
        _phoenix = (_pw, _pf) if (_pw and _pf) else None
        if _phoenix is not None:
            spec_star_ins = LoadPhoenix(
                _phoenix,
                wave_star,
                inst.res,
            )
            print(f"  Phoenix model   : loaded")
        else:
            spec_star_ins = None
            if cfg.observation.event_type == "dayside":
                raise ValueError(
                    "Phoenix stellar model required for dayside simulations. "
                    "Set paths.phoenix_wave_file and paths.phoenix_flux_file."
                )

        # ----------------------------------------------------------------
        # Store state for subsequent blocks
        # ----------------------------------------------------------------
        self._state = dict(
            sim_name=sim_name,
            dirs=dirs,
            inst=inst,
            planet=planet,
            wave_star=wave_star,
            n_pixels=n_pixels,
            sig_og=sig_og,
            snr_og=snr_og,
            JD_og=JD_og,
            airmass_og=airmass_og,
            wave_mid_og=wave_mid_og,
            spec_star_ins=spec_star_ins,
            kp_range=kp_range,
            wave_file=inst.wave_file,
        )

        _t_blocks["Block 1, setup & instrument loading"] = _time.time() - _t_run_start
        print("\n  Block 1 complete, setup and instrument loading done.")
        if cfg.timing:
            print(f"  [timing] Block 1: {_t_blocks['Block 1, setup & instrument loading']:.1f} s")

        # ----------------------------------------------------------------
        # Block 2a, event timing (syn_jd, in-event / out-event indices)
        # ----------------------------------------------------------------
        # When different_nights=True and specific_event=False (fully
        # synthetic, no JD files), get_event expects JD_og[n] per night
        # but JD_og is None.  Synthesise per-night JD arrays here so
        # get_event can index them.  Each night is placed at T0 + n*Period.
        if (cfg.observation.different_nights
                and not cfg.observation.specific_event
                and JD_og is None):
            _jd_step = (cfg.observation.exposure_time_seconds
                        + cfg.observation.overhead_time_seconds
                        + cfg.observation.readout_time_seconds) / 86400.0
            _td_d = planet.transit_duration_hours / 24.0
            _pre_d = _pre_h / 24.0
            _post_d = _post_h / 24.0
            JD_og = []
            for _n in range(cfg.observation.n_nights):
                _tmid_n = (planet.transit_epoch_bjd
                           + _n * planet.orbital_period_days)
                _jd_n = np.arange(
                    _tmid_n - _td_d / 2.0 - _pre_d,
                    _tmid_n + _td_d / 2.0 + _post_d + _jd_step,
                    _jd_step,
                )
                JD_og.append(_jd_n)

        from exoplore.observation import get_event_v2
        syn_jd, with_signal, without_signal, transit_mid_JD = \
            get_event_v2(cfg, planet, JD_og)

        # Per-night cadence override: if exposure_time_seconds_per_night is
        # set with different_nights=True, regenerate synthetic JD grids using
        # each night's own DIT. This enables different cadences per night
        # without requiring per-night JD FITS files.
        _exp_per_night = cfg.observation.exposure_time_seconds_per_night
        if _exp_per_night is not None and cfg.observation.different_nights:
            _readout  = cfg.observation.readout_time_seconds
            _overhead = cfg.observation.overhead_time_seconds
            # _pre_h / _post_h already set at Block 1b

            _new_jd, _new_with, _new_without = [], [], []
            _new_airmass = []
            for _n, _dit in enumerate(_exp_per_night):
                _step = (_dit + _readout + _overhead) / 86400.0
                _tmid = (transit_mid_JD[_n]
                         if isinstance(transit_mid_JD, list)
                         else transit_mid_JD)
                _jd_n = np.arange(
                    _tmid - _pre_h / 24.0,
                    _tmid + _post_h / 24.0,
                    _step,
                )
                _tb = _tmid - planet.transit_duration_hours / 48.0
                _te = _tmid + planet.transit_duration_hours / 48.0
                _new_with.append(
                    np.where(np.logical_and(_jd_n > _tb, _jd_n < _te))[0])
                _new_without.append(
                    np.where(np.logical_or(_jd_n <= _tb, _jd_n >= _te))[0])
                _new_jd.append(_jd_n)
                _am_limits_n = (
                    cfg.tellurics.airmass_limits_per_night[_n]
                    if (cfg.tellurics.airmass_limits_per_night is not None
                        and _n < len(cfg.tellurics.airmass_limits_per_night))
                    else cfg.tellurics.airmass_limits
                )
                _new_airmass.append(
                    get_airmass(cfg.tellurics.airmass_evolution, _jd_n,
                                _am_limits_n)
                    if cfg.tellurics.include_tellurics else None
                )
            syn_jd       = _new_jd
            with_signal  = _new_with
            without_signal = _new_without
            # Update airmass_og so Block 2b's "airmass = airmass_og" picks
            # up the per-night arrays matched to the new JD grids.
            if cfg.tellurics.include_tellurics:
                airmass_og = _new_airmass

            # Resize per-night SNR / signal arrays to match new n_spectra.
            # Interpolate from original JD_og[n] → new syn_jd[n].
            from scipy.interpolate import interp1d as _interp_cadence
            def _resize_per_night(arr_list, jd_list, new_jd_list):
                if arr_list is None:
                    return None
                out = []
                for _n2, _arr in enumerate(arr_list):
                    if _arr is None:
                        out.append(None)
                        continue
                    _jd_src = np.asarray(jd_list[_n2], dtype=float)
                    _arr = np.asarray(_arr, dtype=float)
                    _f = _interp_cadence(
                        _jd_src, _arr, axis=0, assume_sorted=True,
                        bounds_error=False,
                        fill_value=(_arr[0], _arr[-1]),
                    )
                    out.append(_f(new_jd_list[_n2]))
                return out
            snr_og = _resize_per_night(snr_og, JD_og, _new_jd)
            sig_og = _resize_per_night(sig_og, JD_og, _new_jd)

        if not cfg.observation.specific_event:
            from astropy.io import fits as _fits
            jd_path = dirs["matrices"] / "syn_jd.fits"
            if not jd_path.exists():
                hdu = _fits.PrimaryHDU(syn_jd)
                hdu.writeto(str(jd_path), overwrite=True)

        # ----------------------------------------------------------------
        # Block 2b, orbital phase and airmass
        # ----------------------------------------------------------------
        if not cfg.observation.specific_event and not cfg.observation.different_nights:
            phase = (syn_jd - planet.transit_epoch_bjd) / planet.orbital_period_days

            if cfg.tellurics.include_tellurics and not cfg.tellurics.use_full_skycalc:
                airmass = get_airmass(
                    cfg.tellurics.airmass_evolution,
                    syn_jd,
                    cfg.tellurics.airmass_limits,
                )
            else:
                airmass = None

        elif cfg.observation.different_nights:
            phase = []
            for n in range(cfg.observation.n_nights):
                phase.append(
                    (syn_jd[n] - transit_mid_JD[n]) / planet.orbital_period_days
                )
            # airmass_og may be None for fully synthetic different_nights
            # (no reference files).  For use_full_skycalc=True, airmass is
            # embedded in the per-night SkyCalc files and is not needed here;
            # use a list of None as a safe placeholder.
            if airmass_og is None:
                # Synthetic different_nights: no reference airmass files.
                # Mode 2 (use_full_skycalc): airmass is embedded in per-night
                # SkyCalc files, placeholder list of None is fine.
                # Mode 1 (airmass scaling): Beer-Lambert needs a real airmass
                # array per night, compute from the parabolic model using
                # each night's own JD grid.
                if (cfg.tellurics.include_tellurics
                        and not cfg.tellurics.use_full_skycalc):
                    _amlpn = cfg.tellurics.airmass_limits_per_night
                    airmass = [
                        get_airmass(
                            cfg.tellurics.airmass_evolution,
                            syn_jd[n],
                            (_amlpn[n] if _amlpn is not None
                             and n < len(_amlpn)
                             else cfg.tellurics.airmass_limits),
                        )
                        for n in range(cfg.observation.n_nights)
                    ]
                else:
                    airmass = [None] * cfg.observation.n_nights
            else:
                airmass = airmass_og

        else:  # specific_event, single night
            phase = (syn_jd - cfg.observation.specific_T0_bjd) / planet.orbital_period_days
            airmass = airmass_og

        # ----------------------------------------------------------------
        # Block 2c, n_spectra and phase serialisation
        # ----------------------------------------------------------------
        if not cfg.observation.different_nights:
            n_spectra = len(phase)
        else:
            n_spectra = np.zeros(cfg.observation.n_nights, int)
            for n in range(cfg.observation.n_nights):
                n_spectra[n] = int(len(syn_jd[n]))

        if not cfg.observation.different_nights:
            syn_jd = np.asarray(syn_jd, dtype=np.float64)

        if not cfg.observation.different_nights:
            from astropy.io import fits as _fits
            phase_path = dirs["matrices"] / "phase.fits"
            if not phase_path.exists():
                hdu = _fits.PrimaryHDU(phase)
                hdu.writeto(str(phase_path), overwrite=True)
        else:
            phase_path = dirs["matrices"] / "phase.npz"
            if not phase_path.exists():
                np.savez(
                    str(phase_path),
                    **{f"phase_{i}": m for i, m in enumerate(phase)},
                )

        # ----------------------------------------------------------------
        # Block 2d, interpolate SNR / signal arrays to syn_jd time grid
        # ----------------------------------------------------------------
        _oi = cfg.instrument.order_indices
        _n_sel_orders = inst.n_orders_total if not _oi else len(_oi)
        sig_all: np.ndarray | None = None
        if cfg.noise.fixed_snr is None:
            if not cfg.observation.specific_event:
                if snr_og is not None and snr_og.ndim == 3:
                    from scipy.interpolate import interp1d as _interp1d_snr
                    _f_snr = _interp1d_snr(
                        JD_og, snr_og, axis=0, assume_sorted=True,
                        bounds_error=False,
                        fill_value=(snr_og[0], snr_og[-1]),
                    )
                    snr_all = _f_snr(syn_jd)
                    if sig_og is not None:
                        _f_sig = _interp1d_snr(
                            JD_og, sig_og, axis=0, assume_sorted=True,
                            bounds_error=False,
                            fill_value=(sig_og[0], sig_og[-1]),
                        )
                        sig_all = _f_sig(syn_jd)
                    else:
                        sig_all = np.zeros_like(snr_all)
                else:
                    # Static SNR cube (n_orders, n_pixels), ANDES or similar.
                    # For different_nights n_spectra is an array; use its max
                    # as the placeholder size (Block 5a fills real SNR per night).
                    _ns_init = (int(np.max(n_spectra))
                                if hasattr(n_spectra, '__len__') else n_spectra)
                    snr_all = np.zeros(
                        (_ns_init, _n_sel_orders, n_pixels), np.float64
                    )
                    sig_all = np.zeros_like(snr_all)
            else:
                snr_all = snr_og
                sig_all = sig_og
        else:
            snr_all = np.ones_like(wave_star, dtype=np.float64) * cfg.noise.fixed_snr
            sig_all = None

        # ----------------------------------------------------------------
        # Block 2e, CARMENES-specific array transpositions
        # ----------------------------------------------------------------
        carmenes = cfg.instrument.name in ("CARMENES_NIR", "CARMENES_VIS")
        if carmenes:
            wave_star = wave_star.T
            if cfg.noise.fixed_snr is not None and snr_all is not None:
                snr_all = snr_all.T

        # ----------------------------------------------------------------
        # Block 2f, external planet model (optional)
        # ----------------------------------------------------------------
        syn_model: np.ndarray | None = None
        if cfg.observation.external_planet_model:
            ext_file = cfg.observation.external_planet_model_file or ""
            if not ext_file:
                raise FileNotFoundError(
                    "External planet model requested "
                    "(observation.external_planet_model=true) but no file "
                    "path was given.  Set "
                    "observation.external_planet_model_file in your config."
                )
            syn_model = np.loadtxt(ext_file, skiprows=1)
            print(f"  External model  : {Path(ext_file).name} "
                  f"({syn_model.shape[0]} rows)")

        # ----------------------------------------------------------------
        # Store Block 2 state
        # ----------------------------------------------------------------
        self._state.update(dict(
            syn_jd=syn_jd,
            with_signal=with_signal,
            without_signal=without_signal,
            transit_mid_JD=transit_mid_JD,
            phase=phase,
            airmass=airmass,
            n_spectra=n_spectra,
            snr_all=snr_all,
            sig_all=sig_all,
            syn_model=syn_model,
        ))

        self._state["wave_star"] = wave_star

        _dn = cfg.observation.different_nights
        print(f"  n_spectra       : {list(n_spectra) if _dn else n_spectra}")
        _ph_flat = np.hstack(phase) if _dn else np.asarray(phase)
        print(f"  Phase range     : [{_ph_flat.min():.4f}, {_ph_flat.max():.4f}]")

        _t_blocks["Block 2, event timing, phase, airmass, SNR"] = (
            _time.time() - _t_run_start - sum(_t_blocks.values()))
        print("\n  Block 2 complete, event timing, phase, airmass, SNR done.")
        if cfg.timing:
            print(f"  [timing] Block 2: {_t_blocks['Block 2, event timing, phase, airmass, SNR']:.1f} s")

        # ----------------------------------------------------------------
        # Block 3, pressure grids and per-order atmospheric forward model
        # ----------------------------------------------------------------

        # 3a, build log-pressure grids from config
        # (master script: p = np.logspace(-6, 2, 100) at line ~250)
        pm = cfg.atmosphere.planet_model
        p = np.logspace(
            np.log10(pm.pressure_min_bar),
            np.log10(pm.pressure_max_bar),
            pm.pressure_grid_size,
        )

        # CCF-template pressure grid (master script: p_cc at line ~532)
        cc = cfg.atmosphere.ccf_template
        p_cc = np.logspace(
            np.log10(cc.pressure_min_bar),
            np.log10(cc.pressure_max_bar),
            cc.pressure_grid_size,
        )

        # Limb pressure grids (only used if Limb_asymmetries == True)
        if cfg.atmosphere.limb_asymmetries:
            def _pgrid(region_cfg):
                return np.logspace(
                    np.log10(region_cfg.pressure_min_bar),
                    np.log10(region_cfg.pressure_max_bar),
                    region_cfg.pressure_grid_size,
                )
            p_morning_day   = _pgrid(cfg.atmosphere.morning_day)
            p_morning_night = _pgrid(cfg.atmosphere.morning_night)
            p_evening_day   = _pgrid(cfg.atmosphere.evening_day)
            p_evening_night = _pgrid(cfg.atmosphere.evening_night)

        # Store pressure grids for Block 8 (CCF template)
        self._state.update(dict(p=p, p_cc=p_cc))

        print(f"  Pressure grid   : {p.size} levels "
              f"[{p.min():.1e}, {p.max():.1e}] bar")

        # ----------------------------------------------------------------
        # 3b, per-order loop (Blocks 3-12 live inside this loop)
        # ----------------------------------------------------------------
        order_selection = (np.arange(inst.n_orders_total)
                           if not cfg.instrument.order_indices
                           else np.asarray(cfg.instrument.order_indices))
        n_orders = len(order_selection)

        # Mini dicts for the functions that still take an inp_dat-style
        # argument: each carries only the keys that function actually
        # reads, built directly from the config.
        _md = cfg.atmosphere.morning_day
        _ed = cfg.atmosphere.evening_day
        _mini_prt = {
            "event":          cfg.observation.event_type,
            "Gravity":        None,   # call_pRT computes from M/R
            "M_pl":           planet.M_pl,
            "R_pl":           planet.R_pl,
            "R_star":         planet.R_star,
            "T_star":         planet.stellar_teff_K,
            "conv":           cfg.instrument.convolve_to_resolution,
            "res":            inst.res,
            "T_int":          planet.t_int_K,
            "Limb_divisions": cfg.atmosphere.limb_divisions,
        }
        _mini_limbs = {
            **_mini_prt,
            "Limb_divisions":                    cfg.atmosphere.limb_divisions,
            "MMW":                               pm.mean_molecular_weight,
            "species_morning_day":               _md.species,
            "species_morning_night":             cfg.atmosphere.morning_night.species,
            "species_evening_day":               _ed.species,
            "species_evening_night":             cfg.atmosphere.evening_night.species,
            "vmr_morning_day":                   _md.vmr,
            "vmr_morning_night":                 cfg.atmosphere.morning_night.vmr,
            "vmr_evening_day":                   _ed.vmr,
            "vmr_evening_night":                 cfg.atmosphere.evening_night.vmr,
            "MMW_morning_day":                   _md.mean_molecular_weight,
            "MMW_morning_night":                 cfg.atmosphere.morning_night.mean_molecular_weight,
            "MMW_evening_day":                   _ed.mean_molecular_weight,
            "MMW_evening_night":                 cfg.atmosphere.evening_night.mean_molecular_weight,
            "p0_morning_day":                    _md.reference_pressure_bar,
            "p0_morning_night":                  cfg.atmosphere.morning_night.reference_pressure_bar,
            "p0_evening_day":                    _ed.reference_pressure_bar,
            "p0_evening_night":                  cfg.atmosphere.evening_night.reference_pressure_bar,
            "use_easyCHEM_morning_day":          _md.use_easychem,
            "use_easyCHEM_morning_night":        cfg.atmosphere.morning_night.use_easychem,
            "use_easyCHEM_evening_day":          _ed.use_easychem,
            "use_easyCHEM_evening_night":        cfg.atmosphere.evening_night.use_easychem,
            "Metallicity_wrt_solar_morning_day": _md.metallicity_wrt_solar,
            "Metallicity_wrt_solar_morning_night": cfg.atmosphere.morning_night.metallicity_wrt_solar,
            "Metallicity_wrt_solar_evening_day": _ed.metallicity_wrt_solar,
            "Metallicity_wrt_solar_evening_night": cfg.atmosphere.evening_night.metallicity_wrt_solar,
            "C_to_O_morning_day":                _md.carbon_to_oxygen_ratio,
            "C_to_O_morning_night":              cfg.atmosphere.morning_night.carbon_to_oxygen_ratio,
            "C_to_O_evening_day":                _ed.carbon_to_oxygen_ratio,
            "C_to_O_evening_night":              cfg.atmosphere.evening_night.carbon_to_oxygen_ratio,
            # Temperature structure keys (needed by calculate_temperature_structure_limbs)
            "T_equ_morning_day":                 _md.equilibrium_temperature_K,
            "T_equ_morning_night":               cfg.atmosphere.morning_night.equilibrium_temperature_K,
            "T_equ_evening_day":                 _ed.equilibrium_temperature_K,
            "T_equ_evening_night":               cfg.atmosphere.evening_night.equilibrium_temperature_K,
            "T_int":                             planet.t_int_K,
            "isothermal_morning_day":            _md.isothermal,
            "isothermal_morning_night":          cfg.atmosphere.morning_night.isothermal,
            "isothermal_evening_day":            _ed.isothermal,
            "isothermal_evening_night":          cfg.atmosphere.evening_night.isothermal,
            "isothermal_T_value_morning_day":    _md.isothermal_temperature_K,
            "isothermal_T_value_morning_night":  cfg.atmosphere.morning_night.isothermal_temperature_K,
            "isothermal_T_value_evening_day":    _ed.isothermal_temperature_K,
            "isothermal_T_value_evening_night":  cfg.atmosphere.evening_night.isothermal_temperature_K,
            "Kappa_IR_morning_day":              _md.kappa_ir,
            "Kappa_IR_morning_night":            cfg.atmosphere.morning_night.kappa_ir,
            "Kappa_IR_evening_day":              _ed.kappa_ir,
            "Kappa_IR_evening_night":            cfg.atmosphere.evening_night.kappa_ir,
            "Gamma_morning_day":                 _md.gamma_guillot,
            "Gamma_morning_night":               cfg.atmosphere.morning_night.gamma_guillot,
            "Gamma_evening_day":                 _ed.gamma_guillot,
            "Gamma_evening_night":               cfg.atmosphere.evening_night.gamma_guillot,
            "two_point_T_morning_day":           _md.two_point,
            "two_point_T_morning_night":         cfg.atmosphere.morning_night.two_point,
            "two_point_T_evening_day":           _ed.two_point,
            "two_point_T_evening_night":         cfg.atmosphere.evening_night.two_point,
            "p_points_morning_day":              _md.two_point_pressures_bar,
            "p_points_morning_night":            cfg.atmosphere.morning_night.two_point_pressures_bar,
            "p_points_evening_day":              _ed.two_point_pressures_bar,
            "p_points_evening_night":            cfg.atmosphere.evening_night.two_point_pressures_bar,
            "t_points_morning_day":              _md.two_point_temperatures_K,
            "t_points_morning_night":            cfg.atmosphere.morning_night.two_point_temperatures_K,
            "t_points_evening_day":              _ed.two_point_temperatures_K,
            "t_points_evening_night":            cfg.atmosphere.evening_night.two_point_temperatures_K,
        }
        _mini_rotvel = {
            "Period": planet.orbital_period_days,
            "R_pl":   planet.R_pl,
        }
        _mini_sflimbs = {
            "a":          planet.a,
            "incl":       planet.inc,
            "R_pl":       planet.R_pl,
            "R_star":     planet.R_star,
            "T_duration": (planet.transit_duration_hours / 24.0
                           if planet.transit_duration_hours is not None
                           else None),
            "Period":     planet.orbital_period_days,
        }

        # Diagnostic: assert mini-dicts match inp_dat for every key

        # Limb scaling factors are computed once (h==0) and reused each order
        _sf_morning  = None
        _sf_evening  = None
        _ingress_idx = None
        _egress_idx  = None

        # Kernel stores for two-limb modes (h==0 first, then reused per-order)
        _kernel_wind_morning_store = []
        _kernel_wind_evening_store = []
        _kernel_rot_morning_store  = []
        _kernel_rot_evening_store  = []
        _delta_v_windkernel        = []
        _delta_v_rotkernel         = []

        # Block 4 loop-persistent variables (initialized before loop,
        # set at h==0 and then reused for all subsequent orders)
        T_0               = None   # transit/eclipse mid-time used in spec_to_mat
        v_planet          = None   # planet velocity array (or list per night)
        berv              = None   # BERV scalar or per-night list
        fraction          = None   # BATMAN transit fraction (or list per night)
        berv_store        = None   # copy for different_nights
        v_planet_store    = None   # copy for different_nights
        # Per-limb rotation-convolved spectra for spec_to_mat_fraction
        syn_spec_morning_rot = None
        syn_spec_evening_rot = None

        per_order_results = []

        # Block 5 pre-loop: accumulated arrays that span all orders.
        # Allocated at h==0, b==0 inside the per-order loop; declared
        # here to prevent NameError if they are referenced early.
        _mask_store                         = None   # (n_nights, n_orders, n_pixels)
        # Block 6 pre-loop: CCF velocity grid (set once at h==0, b==0)
        _ccf_iterations = None
        _v_ccf          = None
        _ccf_v_step     = None
        _useful_spectral_points_store       = None
        _mask_snr_store                     = None
        _useful_spectral_points_snr_store   = None
        _mask_inter_store                   = None
        _useful_spectral_points_inter_store = None
        _U_sysrem                           = None   # (n_nights, n_spectra, sysrem_its)
        _sysrem_passes_per_order            = None   # (n_orders,) actual passes used
        _mat_star_forfile                   = None   # only Different_nights

        # use_real_data + noiseless consistency check (fires once, before loop)
        if cfg.observation.use_real_data and not cfg.observation.noiseless:
            import warnings as _warnings
            _warnings.warn(
                "\n\n"
                "  [EXoPLORE] use_real_data=True but observation.noiseless=False\n"
                "  ---------------------------------------------------------------\n"
                "  Real observed spectra already contain observational noise.\n"
                "  Adding synthetic Gaussian noise on top is almost certainly not\n"
                "  what you want, and would also trigger a silent pipeline bug\n"
                "  (mat_noise left uninitialised).\n"
                "  EXoPLORE is auto-enforcing noiseless=True for this run.\n\n"
                "  → To silence this warning: set \"noiseless\": true in your config.\n"
                "  → To intentionally add synthetic noise to real spectra (e.g. for\n"
                "    injection-recovery studies): keep \"noiseless\": false, this\n"
                "    warning will still appear as a reminder.\n",
                UserWarning,
                stacklevel=2,
            )
            cfg.observation.noiseless = True   # force-noiseless for real-data run

        # Accumulate fully-masked orders across the whole run so we can
        # report them all at once rather than stopping at the first one.
        _fully_masked_orders: list = []
        _tel_coverage_checked = False  # run wavelength-coverage check once

        for h in range(n_orders):
            # ----------------------------------------------------------
            # 3c, per-order wavelength and SNR setup
            # ----------------------------------------------------------
            wave_ins = wave_star[order_selection[h], :].astype(np.float64)
            if spec_star_ins is not None:
                spec_star_phoenix = spec_star_ins[order_selection[h], :]
            else:
                spec_star_phoenix = None

            # Wavelength bounds wider than array to avoid edge effects
            wvl_min = wave_ins[0]  - 1.0e-3
            wvl_max = wave_ins[-1] + 1.0e-3

            # Per-order SNR and signal arrays
            if not _is_andes_any:
                if not cfg.observation.different_nights:
                    if snr_all.ndim > 2:
                        if cfg.noise.fixed_snr is None:
                            snr = (snr_all[:, order_selection[h], :]
                                   + cfg.noise.snr_correction)
                        else:
                            snr = np.full((n_spectra, n_pixels),
                                          cfg.noise.fixed_snr)
                        sig = (sig_all[:, order_selection[h], :]
                               .astype(float))
                    else:
                        if cfg.noise.fixed_snr is None:
                            snr = (snr_all[order_selection[h], :]
                                   + cfg.noise.snr_correction)
                        else:
                            snr = np.full((n_pixels), cfg.noise.fixed_snr)
                        sig = None
                else:  # different_nights
                    snr = []
                    sig = []
                    for n in range(cfg.observation.n_nights):
                        snr.append(snr_og[n][:, order_selection[h], :])
                        if sig_all is not None:
                            sig.append(
                                sig_all[n][:, order_selection[h], :]
                                .astype(float)
                            )
            else:
                # ANDES: SNR comes from ETC + per-order tellurics, computed
                # in Block 5a after the telluric transmission is known.
                # For single-night: init 1-D zeros, filled in Block 5a.
                # For different_nights: init empty lists; Block 5a appends
                # one (n_spectra_n, n_pix) array per night.
                if cfg.observation.different_nights:
                    snr = []
                    sig = []
                else:
                    snr = np.zeros_like(wave_ins, float)
                    sig = np.zeros_like(wave_ins, float)

            # ----------------------------------------------------------
            # 3d, atmospheric forward model
            # ----------------------------------------------------------
            if cfg.observation.simulate_planet:

                if not cfg.atmosphere.limb_asymmetries:
                    # ---- 1D case ----
                    from petitRADTRANS.radtrans import Radtrans
                    atmosphere = Radtrans(
                        pressures=p,
                        line_species=pm.species[2:],    # skip H2, He
                        rayleigh_species=["H2", "He"],
                        gas_continuum_contributors=["H2--H2", "H2--He"],
                        wavelength_boundaries=[
                            wave_ins.min() - 0.01,
                            wave_ins.max() + 0.01,
                        ],
                        line_opacity_mode="lbl",
                    )
                    (wave_pRT, syn_spec, mass_fractions,
                     MMW, syn_star, temperature) = call_pRT(
                        _mini_prt, p, atmosphere,
                        pm.species, pm.vmr,
                        pm.mean_molecular_weight, pm.reference_pressure_bar,
                        pm.isothermal, pm.isothermal_temperature_K,
                        pm.two_point, pm.two_point_pressures_bar,
                        pm.two_point_temperatures_K, pm.kappa_ir,
                        pm.gamma_guillot, pm.equilibrium_temperature_K,
                        pm.metallicity_wrt_solar, pm.carbon_to_oxygen_ratio,
                        use_easyCHEM=pm.use_easychem,
                        P_cloud=pm.cloud_pressure_bar,
                        easychem_CtoO_ret=pm.use_easychem,
                    )

                    # External planet model override
                    if cfg.observation.external_planet_model:
                        syn_star = np.interp(wave_ins, wave_pRT, syn_star)
                        wave_pRT = np.copy(wave_ins)
                        syn_spec = np.interp(
                            wave_ins, syn_model[:, 0], syn_model[:, 1]
                        )
                        syn_spec = convolve(wave_ins, syn_spec, inst.res)
                        syn_spec *= (planet.R_pl / planet.R_star) ** 2.0

                else:
                    # ---- Limb-asymmetry case ----
                    from petitRADTRANS.radtrans import Radtrans
                    from petitRADTRANS import physical_constants as cst

                    if cfg.atmosphere.limb_divisions in (
                            "gradual", "asymmetric", "simplified_step"):
                        morning_species_differ = (
                            _md.species[2:] != _ed.species[2:]
                        )
                        p_grids_differ = not np.array_equal(
                            p_morning_day, p_evening_day
                        )
                        if morning_species_differ or p_grids_differ:
                            atmosphere_morning = Radtrans(
                                pressures=p_morning_day,
                                line_species=_md.species[2:],
                                rayleigh_species=["H2", "He"],
                                gas_continuum_contributors=[
                                    "H2--H2", "H2--He"],
                                wavelength_boundaries=[
                                    wave_ins.min() - 0.01,
                                    wave_ins.max() + 0.01,
                                ],
                                line_opacity_mode="lbl",
                            )
                            atmosphere_evening = Radtrans(
                                pressures=p_evening_day,
                                line_species=_ed.species[2:],
                                rayleigh_species=["H2", "He"],
                                gas_continuum_contributors=[
                                    "H2--H2", "H2--He"],
                                wavelength_boundaries=[
                                    wave_ins.min() - 0.01,
                                    wave_ins.max() + 0.01,
                                ],
                                line_opacity_mode="lbl",
                            )
                        else:
                            atmosphere_morning = Radtrans(
                                pressures=p_morning_day,
                                line_species=_md.species[2:],
                                rayleigh_species=["H2", "He"],
                                gas_continuum_contributors=[
                                    "H2--H2", "H2--He"],
                                wavelength_boundaries=[
                                    wave_ins.min() - 0.01,
                                    wave_ins.max() + 0.01,
                                ],
                                line_opacity_mode="lbl",
                            )
                            atmosphere_evening = atmosphere_morning

                        (wave_pRT,
                         syn_spec_morning, _,
                         syn_spec_evening, _,
                         mass_fractions_morning, _,
                         MMW_morning, _,
                         mass_fractions_evening, _,
                         MMW_evening, _,
                         syn_star,
                         t_morning, _, t_evening, _,
                         ) = call_pRT_limbs(
                            _mini_limbs,
                            p_morning_day, None,
                            p_evening_day, None,
                            atmosphere_morning, None,
                            atmosphere_evening, None,
                            mode="full", easychem_CtoO_ret=True,
                        )

                        if h == 0:
                            _kernel_wind_morning_store.clear()
                            _kernel_wind_evening_store.clear()
                            _kernel_rot_morning_store.clear()
                            _kernel_rot_evening_store.clear()
                            _delta_v_windkernel.clear()
                            _delta_v_rotkernel.clear()

                        kernel_wind_morning, delta_v = \
                            wind_broadening_triangular_kernel(
                                planet.systemic_velocity_kms,
                                _md.wind_velocity_kms,
                                wave_pRT, max_delta_v=100,
                            )
                        _kernel_wind_morning_store.append(kernel_wind_morning)
                        _delta_v_windkernel.append(delta_v)

                        kernel_wind_evening, _ = \
                            wind_broadening_triangular_kernel(
                                planet.systemic_velocity_kms,
                                _ed.wind_velocity_kms,
                                wave_pRT, max_delta_v=100,
                            )
                        _kernel_wind_evening_store.append(kernel_wind_evening)

                        syn_spec_morning_wc = \
                            convolve_spectrum_with_kernel(
                                wave_pRT, syn_spec_morning,
                                kernel_wind_morning, delta_v,
                            )
                        syn_spec_evening_wc = \
                            convolve_spectrum_with_kernel(
                                wave_pRT, syn_spec_evening,
                                kernel_wind_evening, delta_v,
                            )

                        c_kms = cst.c / 1e5
                        wave_wm = wave_pRT * (
                            1.0 + _md.wind_velocity_kms / c_kms)
                        syn_spec_morning_wind = np.interp(
                            wave_pRT, wave_wm, syn_spec_morning_wc)
                        wave_we = wave_pRT * (
                            1.0 + _ed.wind_velocity_kms / c_kms)
                        syn_spec_evening_wind = np.interp(
                            wave_pRT, wave_we, syn_spec_evening_wc)

                        r1   = planet.R_pl * 1e-5
                        g_pl = (cst.G * planet.M_pl
                                / planet.R_pl ** 2) * 1e-2

                        d_morning = (5.0 * atmospheric_scale_height(
                            _md.equilibrium_temperature_K,
                            np.mean(MMW_morning), g_pl,
                        ) / r1)
                        kernel_morning_rot, delta_v_rot = \
                            rotation_kernel_Maguire24(
                                planet_rot_vel(_mini_rotvel), r1,
                                d_morning, wave_pRT,
                                max_delta_v=100, mode="morning",
                            )
                        _kernel_rot_morning_store.append(kernel_morning_rot)
                        _delta_v_rotkernel.append(delta_v_rot)

                        d_evening = (5.0 * atmospheric_scale_height(
                            _ed.equilibrium_temperature_K,
                            np.mean(MMW_evening), g_pl,
                        ) / r1)
                        kernel_evening_rot, _ = \
                            rotation_kernel_Maguire24(
                                planet_rot_vel(_mini_rotvel), r1,
                                d_evening, wave_pRT,
                                max_delta_v=100, mode="evening",
                            )
                        _kernel_rot_evening_store.append(kernel_evening_rot)

                        syn_spec_morning_rot = \
                            convolve_spectrum_with_kernel(
                                wave_pRT, syn_spec_morning_wind,
                                kernel_morning_rot, delta_v_rot,
                            )
                        syn_spec_evening_rot = \
                            convolve_spectrum_with_kernel(
                                wave_pRT, syn_spec_evening_wind,
                                kernel_evening_rot, delta_v_rot,
                            )

                        syn_spec_rot = 0.5 * (
                            syn_spec_morning_rot + syn_spec_evening_rot
                        )
                        syn_spec = convolve(
                            wave_pRT, syn_spec_rot, inst.res
                        )

                        if h == 0:
                            (sf_morning, _sf_ign,
                             sf_evening, _sf_ign2,
                             _ingress_idx, _egress_idx,
                             ) = get_sflimbs(
                                _mini_sflimbs, with_signal, without_signal,
                                phase, syn_jd,
                                mode=cfg.atmosphere.limb_divisions,
                            )
                            _sf_morning = sf_morning
                            _sf_evening = sf_evening

            else:
                # ---- No planet signal ----
                if cfg.observation.event_type == "transit":
                    wave_pRT = wave_ins
                    syn_spec = np.zeros_like(wave_ins)
                    syn_star = np.zeros_like(wave_ins)
                elif cfg.observation.event_type == "dayside":
                    raise NotImplementedError(
                        "Dayside simulation without planet signal requires "
                        "a low-res pRT Phoenix model. Update this code "
                        "path before running dayside simulations."
                    )
                else:
                    wave_pRT = wave_ins
                    syn_spec = np.zeros_like(wave_ins)
                    syn_star = np.zeros_like(wave_ins)

            # ----------------------------------------------------------
            # Collect Block 3 per-order results
            # ----------------------------------------------------------
            _order_result = {
                "h": h,
                "wave_ins": wave_ins,
                "spec_star_phoenix": spec_star_phoenix,
                "snr": snr,
                "sig": sig,
                "wave_pRT": wave_pRT,
                "syn_spec": syn_spec,
                "syn_star": syn_star,
            }
            # Limb-specific results (only if applicable)
            if cfg.atmosphere.limb_asymmetries and cfg.observation.simulate_planet:
                _order_result["sf_morning"]  = _sf_morning
                _order_result["sf_evening"]  = _sf_evening
                _order_result["ingress_idx"] = _ingress_idx
                _order_result["egress_idx"]  = _egress_idx

            per_order_results.append(_order_result)

            # Mini-dict for spec_to_mat_fraction (6 keys only)
            _mini_stmf = {
                "event":               cfg.observation.event_type,
                "Scale_inj":           cfg.observation.scale_injection,
                "Inject_Scale_Factor": cfg.pipeline.inject_scale_factor,
                "Limb_asymmetries":    cfg.atmosphere.limb_asymmetries,
                "Limb_divisions":      cfg.atmosphere.limb_divisions,
                "res":                 inst.res,
            }

            # ----------------------------------------------------------
            # Block 4a, stellar matrix (every order)
            # ----------------------------------------------------------
            from astropy.io import fits as _fits
            _berv_ref = (cfg.observation.berv_kms
                         if cfg.observation.berv_kms is not None else 0.0)
            if not cfg.observation.different_nights:
                v_star = get_V(
                    planet.stellar_rv_semiamplitude_kms, phase,
                    _berv_ref, planet.systemic_velocity_kms, 0,
                )
                mat_star = get_stellar_matrix(
                    spec_star_phoenix, v_star, wave_ins,
                )
            else:
                v_star = []
                mat_star = []
                for nn in range(cfg.observation.n_nights):
                    v_star.append(get_V(
                        planet.stellar_rv_semiamplitude_kms, phase[nn],
                        _berv_ref, planet.systemic_velocity_kms, 0,
                    ))
                    mat_star.append(get_stellar_matrix(
                        spec_star_phoenix, v_star[nn], wave_ins,
                    ))

            # ----------------------------------------------------------
            # Block 4b, planet velocity (h==0 only; reused per order)
            # ----------------------------------------------------------
            v_wind = (pm.wind_velocity_kms
                      if not cfg.atmosphere.limb_asymmetries else 0.0)

            if not cfg.observation.different_nights and h == 0:
                berv = _berv_ref

                if not cfg.observation.significant_eccentricity:
                    v_planet = get_V(
                        planet.kp_kms, phase, berv,
                        planet.systemic_velocity_kms, v_wind,
                    )
                else:
                    v_planet = get_V_eccentric(
                        planet.kp_kms, phase,
                        planet.eccentricity,
                        planet.argument_of_periastron_deg,
                        berv, planet.systemic_velocity_kms, v_wind,
                    )

            elif h == 0:   # different_nights, first pass
                v_planet = []
                berv = []
                _exp_pn = cfg.observation.exposure_time_seconds_per_night
                for n in range(cfg.observation.n_nights):
                    if (_exp_pn is not None and not cfg.observation.use_real_data
                            or not cfg.observation.specific_event
                            and not cfg.observation.use_real_data):
                        # Synthetic different_nights (no reference BERV files):
                        # broadcast constant berv_kms for each night.
                        berv.append(
                            np.full(int(n_spectra[n]),
                                    cfg.observation.berv_kms, dtype=float)
                        )
                    else:
                        _berv_file = (
                            f"{cfg.paths.inputs_dir}reference_night/"
                            f"observations_berv_{n}.fits"
                        )
                        berv.append(_fits.open(_berv_file)[0].data)
                    if not cfg.observation.significant_eccentricity:
                        v_planet.append(get_V(
                            planet.kp_kms, phase[n], berv[n],
                            planet.systemic_velocity_kms, v_wind,
                        ))
                    else:
                        v_planet.append(get_V_eccentric(
                            planet.kp_kms, phase[n],
                            planet.eccentricity,
                            planet.argument_of_periastron_deg,
                            berv[n], planet.systemic_velocity_kms, v_wind,
                        ))
                berv_store     = berv.copy()
                v_planet_store = v_planet.copy()

            # ----------------------------------------------------------
            # Block 4c, BATMAN transit / dayside fraction (h==0 only)
            # ----------------------------------------------------------
            _use_lc = (cfg.observation.signal_uses_light_curve
                       and not np.isnan(planet.argument_of_periastron_deg))
            if h == 0:
                if not cfg.observation.different_nights:
                    fraction = np.zeros(n_spectra, float)
                    if _use_lc:
                        if cfg.observation.event_type == "transit":
                            T_0 = (cfg.observation.specific_T0_bjd
                                   if cfg.observation.specific_event
                                   else planet.transit_epoch_bjd)
                            fraction = block_parameter(
                                syn_jd, T_0,
                                planet.orbital_period_days,
                                planet.R_pl,
                                planet.a * 1e5,          # km → cm
                                planet.R_star,
                                planet.inclination_deg,
                                planet.limb_darkening_coeffs,
                                e=planet.eccentricity,
                                omega=planet.argument_of_periastron_deg,
                            )
                        else:  # dayside
                            fraction = dayside_fraction(
                                syn_jd, without_signal,
                            )
                    else:
                        T_0 = (cfg.observation.specific_T0_bjd
                               if cfg.observation.specific_event
                               else planet.transit_epoch_bjd)
                        fraction[with_signal] = 1.0

                else:   # different_nights
                    if _use_lc and cfg.observation.event_type == "transit":
                        fraction = []
                        for nn in range(cfg.observation.n_nights):
                            _jdsyn = np.asarray(syn_jd[nn])
                            T_0    = transit_mid_JD[nn]
                            fraction.append(block_parameter(
                                _jdsyn - T_0, 0.0,
                                planet.orbital_period_days,
                                planet.R_pl,
                                planet.a * 1e5,
                                planet.R_star,
                                planet.inclination_deg,
                                planet.limb_darkening_coeffs,
                                e=planet.eccentricity,
                                omega=planet.argument_of_periastron_deg,
                            ))
                    else:
                        T_0 = (transit_mid_JD
                               if cfg.observation.specific_event
                               else planet.transit_epoch_bjd)
                        fraction = []
                        for nn in range(cfg.observation.n_nights):
                            _aux = np.zeros_like(phase[nn])
                            _aux[with_signal[nn]] = 1.0
                            fraction.append(_aux)

            # ----------------------------------------------------------
            # Block 4d, spectral matrix (planet signal time-series)
            # ----------------------------------------------------------
            if not cfg.atmosphere.limb_asymmetries:
                if not cfg.observation.different_nights:
                    spec_mat, spec_mat_shift = \
                        spec_to_mat_fraction(
                            _mini_stmf, syn_jd, T_0, v_planet,
                            wave_ins, wave_pRT, syn_spec, mat_star,
                            with_signal, without_signal, fraction,
                            include_star=False,
                        )
                else:
                    spec_mat       = []
                    spec_mat_shift = []
                    for n in range(cfg.observation.n_nights):
                        _a, _b = spec_to_mat_fraction(
                            _mini_stmf, syn_jd[n], transit_mid_JD[n],
                            v_planet[n], wave_ins, wave_pRT, syn_spec,
                            mat_star[n], with_signal[n], without_signal[n],
                            fraction[n], include_star=False,
                        )
                        spec_mat.append(_a)
                        spec_mat_shift.append(_b)

            else:   # limb asymmetries
                spec_mat, spec_mat_shift = \
                    spec_to_mat_fraction(
                        _mini_stmf, syn_jd, T_0, v_planet,
                        wave_ins, wave_pRT, syn_spec, mat_star,
                        with_signal, without_signal, fraction,
                        syn_spec_morning_rot, None,
                        syn_spec_evening_rot, None,
                        _sf_evening, None, _sf_morning, None,
                    )

            # ----------------------------------------------------------
            # Update per-order results with Block 4 outputs
            # ----------------------------------------------------------
            _order_result.update({
                "mat_star":       mat_star,
                "v_star":         v_star,
                "spec_mat":       spec_mat,
                "spec_mat_shift": spec_mat_shift,
            })

            # ----------------------------------------------------------
            # Block 5a, telluric transmittances
            # ----------------------------------------------------------
            _tell_ref_file = (
                cfg.tellurics.reference_telluric_file
                or (f"{cfg.paths.inputs_dir}tellurics/"
                    f"tell_ref_airmass_{cfg.tellurics.reference_airmass:.1f}.fits")
            )
            tell_ref   = None
            tell_trans = None

            # One-time check: does the telluric reference file cover
            # the instrument wavelength range?
            if (cfg.tellurics.include_tellurics
                    and not cfg.tellurics.use_full_skycalc
                    and not _tel_coverage_checked
                    and _tell_ref_file):
                _tel_coverage_checked = True
                try:
                    from astropy.io import fits as _fits_telcheck
                    with _fits_telcheck.open(_tell_ref_file) as _th:
                        _tel_lam = _th[1].data['lam'] * 1e-3  # nm → µm
                    _inst_min = float(wave_star.min())
                    _inst_max = float(wave_star.max())
                    _tel_min  = float(_tel_lam.min())
                    _tel_max  = float(_tel_lam.max())
                    _overlap  = min(_inst_max, _tel_max) - max(_inst_min, _tel_min)
                    if _overlap <= 0:
                        print(
                            f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            f"  WARNING: telluric reference file covers\n"
                            f"    {_tel_min:.3f} to {_tel_max:.3f} µm\n"
                            f"  but the instrument covers\n"
                            f"    {_inst_min:.3f} to {_inst_max:.3f} µm.\n"
                            f"  There is NO wavelength overlap, all orders will\n"
                            f"  likely be fully masked. Generate a telluric reference\n"
                            f"  covering the instrument range:\n"
                            f"    python scripts/generate_skycalc_inputs.py \\\n"
                            f"        <your_config.json> --ref-only --mode synthetic\n"
                            f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        )
                    elif _overlap < 0.5 * (_inst_max - _inst_min):
                        print(
                            f"\n  WARNING: telluric reference file covers "
                            f"{_tel_min:.3f} to {_tel_max:.3f} µm but instrument "
                            f"covers {_inst_min:.3f} to {_inst_max:.3f} µm, "
                            f"partial overlap only. Some orders may be fully masked.\n"
                        )
                except Exception:
                    pass  # non-fatal; do not interrupt the run

            if cfg.tellurics.include_tellurics and not cfg.observation.use_real_data:
                # Real-data analysis substitutes the observed matrices before the
                # preparing pipeline, so simulated telluric transmittances are
                # irrelevant and must not be loaded (tell_ref/tell_trans stay None,
                # set above).  include_tellurics still routes the data-driven
                # pipeline (e.g. Cheverall26) via telluric_variation.
                _pwv_subdir = ("Fixed_PWV" if cfg.tellurics.constant_pwv
                               else "Variable_PWV")
                _filepath_tel = (
                    f"{cfg.paths.inputs_dir}"
                    f"Skycalc_{cfg.observation.flag_event}/"
                    f"{_pwv_subdir}/"
                )

                if not cfg.observation.different_nights:
                    _pwv_values = PWV_handling(
                        cfg.tellurics.constant_pwv, cfg.tellurics.pwv_mm,
                        n_spectra,
                        f"{_filepath_tel}pwv_values.fits",
                    )
                    tell_ref, tell_trans = Load_Telluric_Transmittances(
                        snr, cfg.tellurics.include_tellurics,
                        cfg.tellurics.use_full_skycalc,
                        _tell_ref_file, _filepath_tel,
                        inst.res, syn_jd, wave_ins, spec_mat, airmass,
                    )
                    spec_mat = spec_mat * tell_trans

                    # ANDES: compute per-pixel SNR from ETC + tellurics
                    if cfg.instrument.name in (
                        "ANDES", "ANDES_YJHK", "ANDES_YJH",
                        "ANDES_K", "ANDES_RIZ", "ANDES_UBV",
                    ):
                        if not _is_andes_yjhk:
                            from astropy.io import fits as _fits_b5
                            _center_pix = np.where(
                                wave_ins == find_nearest(
                                    wave_ins,
                                    _fits_b5.open(inst.wave_file)[2].data[
                                        order_selection[h]
                                    ] * 1e-3,
                                )
                            )[0][0]
                            snr = pixel_snr_one_order(
                                wave_star[h], tell_trans, snr_og[h],
                                px_per_resel=cfg.instrument.pixels_per_resolution_element,
                                center_pix=_center_pix,
                            )
                        else:
                            _order_idx_b5 = order_selection[h]
                            _lambda_mid   = wave_mid_og[_order_idx_b5] * 1e-3
                            _center_pix   = int(
                                np.argmin(np.abs(wave_star[h] - _lambda_mid))
                            )
                            snr = pixel_snr_one_order(
                                wave_star[h], tell_trans,
                                snr_og[_order_idx_b5],
                                px_per_resel=cfg.instrument.pixels_per_resolution_element,
                                center_pix=_center_pix,
                            )

                else:   # Different_nights
                    _tell_ref_list  = []
                    _tell_trans_list = []
                    _spec_mat_aux   = spec_mat.copy()
                    spec_mat = []

                    # ANDES: pre-compute order index and blaze-centre pixel
                    # once, they are wavelength-grid properties, identical
                    # for all nights.
                    if _is_andes_any:
                        _order_idx_dn = order_selection[h]
                        if _is_andes_yjhk:
                            _lambda_mid_dn = wave_mid_og[_order_idx_dn] * 1e-3
                            _cpix_dn = int(
                                np.argmin(np.abs(wave_star[h] - _lambda_mid_dn))
                            )
                        else:
                            from astropy.io import fits as _fits_dn
                            _cpix_dn = np.where(
                                wave_ins == find_nearest(
                                    wave_ins,
                                    _fits_dn.open(wave_file)[2].data[
                                        _order_idx_dn
                                    ] * 1e-3,
                                )
                            )[0][0]

                    for _nn in range(cfg.observation.n_nights):
                        # Per-night SkyCalc directory when use_full_skycalc
                        # is enabled: Skycalc_{event}/night_{N}/Fixed_PWV/.
                        # This allows each night to have its own telluric
                        # spectra (different PWV, different airmass sequence)
                        # generated by generate_skycalc_inputs.py --night N.
                        # For Mode 1 (airmass scaling, use_full_skycalc=False)
                        # a single reference file is shared across all nights.
                        if cfg.tellurics.use_full_skycalc:
                            _filepath_tel_nn = (
                                f"{cfg.paths.inputs_dir}"
                                f"Skycalc_{cfg.observation.flag_event}/"
                                f"night_{_nn}/{_pwv_subdir}/"
                            )
                        else:
                            _filepath_tel_nn = _filepath_tel

                        # Per-night PWV: use pwv_mm_per_night[n] when set.
                        _pwv_nn = (
                            cfg.tellurics.pwv_mm_per_night[_nn]
                            if cfg.tellurics.pwv_mm_per_night is not None
                            else cfg.tellurics.pwv_mm
                        )
                        _pwv_values = PWV_handling(
                            cfg.tellurics.constant_pwv, _pwv_nn,
                            int(n_spectra[_nn]),
                            f"{_filepath_tel_nn}pwv_values.fits",
                        )
                        # Load_Telluric_Transmittances uses snr only to check
                        # ndim (1-D → reference mode, 2-D → per-exposure mode).
                        # For ANDES the real SNR is computed after tellurics;
                        # pass a dummy 2-D array so the function takes the
                        # per-exposure path.
                        _snr_arg = (
                            snr[_nn] if not _is_andes_any
                            else np.ones((int(n_spectra[_nn]), 1), float)
                        )
                        _tr, _tt = Load_Telluric_Transmittances(
                            _snr_arg, cfg.tellurics.include_tellurics,
                            cfg.tellurics.use_full_skycalc,
                            _tell_ref_file, _filepath_tel_nn,
                            inst.res, syn_jd[_nn], wave_ins,
                            _spec_mat_aux[_nn], airmass[_nn],
                        )
                        _tell_ref_list.append(_tr)
                        _tell_trans_list.append(_tt)
                        spec_mat.append(_spec_mat_aux[_nn] * _tt)

                        # ANDES: compute per-pixel SNR from ETC + per-night
                        # telluric transmittance, then store for this night.
                        if _is_andes_any:
                            _snr_nn = pixel_snr_one_order(
                                wave_star[h], _tt,
                                snr_og[_order_idx_dn],
                                px_per_resel=(
                                    cfg.instrument.pixels_per_resolution_element
                                ),
                                center_pix=_cpix_dn,
                            )
                            snr.append(_snr_nn)
                            sig.append(np.zeros_like(_snr_nn))

                    tell_ref  = _tell_ref_list
                    tell_trans = _tell_trans_list
                    del _spec_mat_aux

            # ----------------------------------------------------------
            # Block 5b, CCF template spectrum
            # ----------------------------------------------------------
            if cfg.atmosphere.cc_with_true_model:
                wave_pRT_cc = wave_pRT
                spec_cc     = syn_spec
            else:
                from petitRADTRANS.radtrans import Radtrans as _Radtrans_cc
                _atmosphere_cc = _Radtrans_cc(
                    pressures=p,
                    line_species=cc.species[2:],
                    rayleigh_species=['H2', 'He'],
                    gas_continuum_contributors=['H2--H2', 'H2--He'],
                    wavelength_boundaries=[
                        wave_ins.min() - 0.01,
                        wave_ins.max() + 0.01,
                    ],
                    line_opacity_mode='lbl',
                )
                (wave_pRT_cc, spec_cc, _, _, _, _) = call_pRT(
                    _mini_prt, p_cc, _atmosphere_cc,
                    cc.species, cc.vmr,
                    cc.mean_molecular_weight, cc.reference_pressure_bar,
                    cc.isothermal, cc.isothermal_temperature_K,
                    cc.two_point, cc.two_point_pressures_bar,
                    cc.two_point_temperatures_K, cc.kappa_ir,
                    cc.gamma_guillot, cc.equilibrium_temperature_K,
                    cc.metallicity_wrt_solar, cc.carbon_to_oxygen_ratio,
                    use_easyCHEM=cc.use_easychem,
                    easychem_CtoO_ret=cc.use_easychem,
                )

            # ----------------------------------------------------------
            # Block 5c, Different_nights: snapshot per-night stores
            # ----------------------------------------------------------
            if cfg.observation.different_nights:
                _phase_store5        = phase.copy()
                _n_spectra_store_b5  = n_spectra.copy()
                _with_signal_store5  = with_signal.copy()
                _without_signal_store5 = without_signal.copy()
                _airmass_store5      = airmass.copy()
                _fraction_store5     = fraction.copy()
                _spec_mat_store5     = spec_mat.copy()
                _mat_star_store5     = mat_star.copy()
                _syn_jd_store5       = syn_jd.copy()
                del (phase, syn_jd, n_spectra, with_signal, without_signal,
                     airmass, fraction, spec_mat, mat_star)
            else:
                _n_spectra_store_b5 = None   # not needed in single-night path

            # ----------------------------------------------------------
            # Block 5d, per-night simulation loop
            # ----------------------------------------------------------
            # Per-order results to be filled by the for-b loop
            _mat_noise_b5    = None
            _std_noise_b5    = None
            _mat_res_b5      = None
            _propag_noise_b5 = None
            _sysrem_pass_b5  = None
            _mat_cc_b5       = None
            _mat_back_b5     = None
            _syn_mat_res_b5  = None
            _v_cc_b5         = None
            _useful_spectral_points_b5 = None
            _mask_b5                   = None
            _ccf_store_b6              = None   # (n_nights, ccf_iters, n_spectra)
            _ccf_flag                  = cfg.cross_correlation.cc_metric

            from petitRADTRANS import physical_constants as _cst_b5

            for b in range(cfg.observation.n_nights):

                # Restore per-night variables (Different_nights only)
                if cfg.observation.different_nights:
                    phase       = np.asarray(_phase_store5[b],       dtype=np.float64)
                    berv        = np.asarray(berv_store[b],           dtype=np.float64)
                    n_spectra   = int(_n_spectra_store_b5[b])
                    with_signal = np.asarray(_with_signal_store5[b],  dtype=int)
                    without_signal = np.asarray(
                        _without_signal_store5[b], dtype=int)
                    airmass     = np.asarray(_airmass_store5[b],      dtype=np.float64)
                    fraction    = np.asarray(_fraction_store5[b],     dtype=np.float64)
                    spec_mat    = np.asarray(_spec_mat_store5[b],     dtype=np.float64)
                    mat_star    = np.asarray(_mat_star_store5[b],     dtype=np.float64)
                    syn_jd      = np.asarray(_syn_jd_store5[b],       dtype=np.float64)

                # 5d.1, array allocation at b==0
                _dn  = cfg.observation.different_nights
                _nl  = cfg.observation.noiseless
                _urd = cfg.observation.use_real_data
                _nn  = cfg.observation.n_nights
                if b == 0:
                    if not _urd and not _dn:
                        if not _nl:
                            _gauss_noise_b5 = np.zeros(
                                (_nn, n_spectra, n_pixels), float)
                        _mat_noise_b5 = np.empty(
                            (_nn, n_spectra, n_pixels), float)
                        _std_noise_b5 = np.empty(
                            (_nn, n_spectra, n_pixels), float)
                    elif _urd and not _dn:
                        _mat_noise_b5 = np.empty(
                            (_nn, n_spectra, n_pixels), float)
                        _std_noise_b5 = np.empty(
                            (_nn, n_spectra, n_pixels), float)
                    else:   # Different_nights
                        _max_sp = int(np.max(_n_spectra_store_b5))
                        if not _nl:
                            _gauss_noise_b5 = np.zeros(
                                (_nn, _max_sp, n_pixels), float)
                        _mat_noise_b5 = np.zeros(
                            (_nn, _max_sp, n_pixels), float)
                        _std_noise_b5 = np.zeros(
                            (_nn, _max_sp, n_pixels), float)

                if _dn:
                    _mat_noise_b5[b, n_spectra:, :] = np.nan
                    _std_noise_b5[b, n_spectra:, :] = np.nan
                    if not _nl:
                        _gauss_noise_b5[b, n_spectra:, :] = np.nan

                # 5d.2, throughput variations
                if not _nl and cfg.observation.add_throughput_variations:
                    _spec_tp = add_throughput(F=spec_mat, jitter_frac=0.02)
                else:
                    _spec_tp = np.copy(spec_mat)

                # 5d.3, noise standard-deviation matrix
                _nc = cfg.noise.noise_choice
                _nf = cfg.observation.noise_scaling_factor
                _sc = cfg.noise.snr_correction
                if not _urd and not _nl and _nc == 'SNR' and not _dn:
                    snr = np.maximum(snr, 1.0)
                    _std_noise_b5[b, :] = _nf / snr + _sc
                elif not _urd and not _nl and _nc == 'SNR' and _dn:
                    snr[b] = np.maximum(snr[b], 1.0)
                    _std_noise_b5[b, :n_spectra, :] = _nf / snr[b] + _sc
                elif _urd and _dn and _nl:
                    _std_noise_b5[b, :n_spectra, :] = np.asarray(
                        sig[b], dtype=np.float64)
                elif not _urd and _dn and _nl:
                    _sig_b = np.asarray(sig[b], dtype=np.float64)
                    if np.all(_sig_b == 0):
                        # sig[b] is uninformative (ANDES has no real signal
                        # array); use ETC-based noise so the SNR mask does
                        # not reject every pixel.
                        snr[b] = np.maximum(snr[b], 1.0)
                        _std_noise_b5[b, :n_spectra, :] = _nf / snr[b] + _sc
                    else:
                        _std_noise_b5[b, :n_spectra, :] = _sig_b
                elif np.all(sig) is None or np.sum(sig) == 0:
                    snr = np.maximum(snr, 1.0)
                    _std_noise_b5[b, :] = _nf / snr + _sc
                else:
                    _std_noise_b5[b, :] = sig

                # 5d.4, telluric scaling of noise + Gaussian noise draw
                _nseed = cfg.noise.noise_seed if _nn == 1 else None
                # Cheverall26 statistics route: make the multi-night noise
                # realisations exactly reproducible from the (tiny) base seed
                # while staying independent across orders (h) and nights (b).
                # Gated to Cheverall26 so every other pipeline keeps its
                # default behaviour (per-night seed=None when n_nights>1).
                if (cfg.pipeline.name == "Cheverall26" and _nn != 1
                        and cfg.noise.noise_seed is not None):
                    _nseed = (int(cfg.noise.noise_seed) * 1_000_003
                              + h * 10_007 + b)
                if not _nl and not _dn:
                    if (cfg.tellurics.include_tellurics
                            and np.ndim(snr) == 1
                            and tell_trans is not None
                            and tell_ref is not None):
                        _std_noise_b5[b, :] *= np.sqrt(
                            tell_trans / tell_ref)
                    _rng = np.random.default_rng(seed=_nseed)
                    _gauss_noise_b5[b] = _rng.normal(
                        loc=0.0, scale=_std_noise_b5[b],
                        size=(n_spectra, n_pixels),
                    )
                    if cfg.observation.first_night_noiseless and b == 0:
                        _gauss_noise_b5[b, :, :] = 0.0
                elif not _nl and _dn:
                    if (cfg.tellurics.include_tellurics
                            and np.ndim(snr[b]) == 1
                            and tell_trans is not None
                            and tell_ref is not None):
                        _std_noise_b5[b, :n_spectra, :] *= np.sqrt(
                            tell_trans[b] / tell_ref[b])
                    _rng = np.random.default_rng(seed=_nseed)
                    _max_sp_b = int(np.max(_n_spectra_store_b5))
                    _gauss_noise_b5[b] = _rng.normal(
                        loc=0.0, scale=_std_noise_b5[b],
                        size=(_max_sp_b, n_pixels),
                    )
                    if cfg.observation.first_night_noiseless and b == 0:
                        _gauss_noise_b5[b, :n_spectra, :] = 0.0

                # 5d.5, SNR mask
                if not _dn:
                    _mask_snr_bool = (snr / _nf) < cfg.pipeline.snr_mask_threshold
                    _mask_snr_bool |= (_std_noise_b5[b, :] < 1e-9)
                    _mask_snr_indices = np.argwhere(_mask_snr_bool)
                else:
                    _mask_snr_bool = (
                        (snr[b] / _nf) < cfg.pipeline.snr_mask_threshold)
                    _mask_snr_bool |= (
                        _std_noise_b5[b, :n_spectra, :] < 1e-9)
                    _mask_snr_indices = np.argwhere(_mask_snr_bool)

                # 5d.6, real data: load from disk + NaN/outlier correction
                if _urd:
                    from astropy.io import fits as _fits_b5d
                    _idir = cfg.paths.inputs_dir
                    if cfg.instrument.split_detectors:
                        _fn = (f"{_idir}reference_night/"
                               f"observations_order_"
                               f"{order_selection[h // 2]}.fits")
                        _spec_tp = From1OrderTo1Detector(
                            _fits_b5d.open(_fn)[0].data, h)
                    else:
                        _fn = (f"{_idir}reference_night/"
                               f"observations_night_{b}_order_"
                               f"{order_selection[h]}.fits")
                        _spec_tp = _fits_b5d.open(_fn)[0].data

                    if h == 0 and b == 0 and not _dn:
                        berv = _fits_b5d.open(
                            f"{_idir}reference_night/observations_berv_0.fits"
                        )[0].data
                        # berv is consumed by Block 7's _mini_b7 init ("BERV": berv),
                        # which runs later; no need to sync a dict that does not
                        # exist yet on this real-data path.

                    if not _dn:
                        _spec_tp, _std_noise_b5[b, :] = \
                            Correct_NaN(_spec_tp, _std_noise_b5[b, :])
                        _spec_tp, _std_noise_b5[b, :] = \
                            Remove_Outliers(_spec_tp, _std_noise_b5[b, :])
                    else:
                        _spec_tp, _std_noise_b5[b, :n_spectra, :] = \
                            Correct_NaN(
                                _spec_tp, _std_noise_b5[b, :n_spectra, :])
                        _spec_tp, _std_noise_b5[b, :n_spectra, :] = \
                            Remove_Outliers(
                                _spec_tp, _std_noise_b5[b, :n_spectra, :])

                # 5d.6b, real-data injection-recovery: imprint the planet
                # transmission (spec_mat = 1 - scale*depth*fraction in-transit,
                # 1 out-of-transit) onto the loaded REAL spectra.  Only when
                # analysing real data AND a planet is requested (simulate_planet).
                # For ordinary real-data analysis (simulate_planet=false) this
                # is skipped and the real spectra are used unchanged.
                if _urd and cfg.observation.simulate_planet:
                    _spec_tp = _spec_tp * spec_mat

                # 5d.7, final noisy matrix
                if not _nl and not _urd and not _dn:
                    _mat_noise_b5[b, :] = _spec_tp + _gauss_noise_b5[b, :, :]
                elif not _nl and not _urd and _dn:
                    _mat_noise_b5[b, :n_spectra, :] = (
                        _spec_tp + _gauss_noise_b5[b, :n_spectra, :])
                elif _nl and _dn:
                    _mat_noise_b5[b, :n_spectra, :] = np.copy(_spec_tp)
                elif _nl and not _dn:
                    _mat_noise_b5[b] = np.copy(_spec_tp)
                elif _urd and not _nl and _dn:
                    _mat_noise_b5[b, :n_spectra, :] = np.copy(_spec_tp)
                elif _urd and not _nl and not _dn:
                    _mat_noise_b5[b] = np.copy(_spec_tp)

                # Apply SNR mask
                if _mask_snr_indices.T[1].shape != (0,):
                    if not _dn:
                        for _j in _mask_snr_indices.T[1]:
                            _mask_snr_bool[:, _j] = True
                            _mat_noise_b5[b][:][:, _j] = 1
                    else:
                        for _j in _mask_snr_indices.T[1]:
                            _mask_snr_bool[:, _j] = True
                            _mat_noise_b5[b][:][:n_spectra, _j] = 1

                if np.where(_mask_snr_bool)[0].shape != (0,):
                    if not _dn:
                        _mask_snr_1d = np.where(
                            _mat_noise_b5[0, with_signal[0], :] == 1)[0]
                    else:
                        _mask_snr_1d = np.where(
                            _mat_noise_b5[b, with_signal[0], :] == 1)[0]
                    _useful_snr = np.setdiff1d(
                        np.arange(n_pixels), _mask_snr_1d)
                else:
                    _mask_snr_1d = np.asarray([], dtype=int)
                    _useful_snr  = np.arange(n_pixels)

                # 5d.8, negative value masking
                if np.any(_mat_noise_b5 < 0):
                    if not _dn:
                        _bad_cols = np.any(_mat_noise_b5[b, :] <= 0., axis=0)
                        _mat_noise_b5[b, :, _bad_cols] = 1
                        _mask_snr_1d, _useful_snr = _merge_masks(
                            _mask_snr_1d, np.argwhere(_bad_cols), n_pixels)
                    else:
                        _bad_cols = np.any(
                            _mat_noise_b5[b, :n_spectra, :] <= 0., axis=0)
                        _mat_noise_b5[b, :n_spectra, _bad_cols] = 1
                        _mask_snr_1d, _useful_snr = _merge_masks(
                            _mask_snr_1d, np.where(_bad_cols)[0], n_pixels)

                # 5d.9, detector gap masking
                if inst.gaps is not None:
                    assert len(inst.gaps) % 2 == 0
                    _mask_gap = np.zeros(n_pixels, dtype=bool)
                    for _gs, _ge in zip(inst.gaps[::2], inst.gaps[1::2]):
                        _mask_gap |= (wave_ins >= _gs) & (wave_ins <= _ge)
                    _mat_noise_b5[b, :, _mask_gap] = 1

                # 5d.10, allocate mask_store + U arrays (h==0, b==0 only)
                _si = cfg.pipeline.sysrem_iterations
                if h == 0 and b == 0:
                    if _dn:
                        _max_sp_alloc = int(np.max(_n_spectra_store_b5))
                        _U_sysrem = np.zeros((_nn, _max_sp_alloc, _si), float)
                    else:
                        _U_sysrem = np.zeros((_nn, n_spectra, _si), float)
                    # Only SYSREM/PCA pipelines have a per-order component count;
                    # BL19 and Blain24 are polynomial fitting pipelines (no SYSREM,
                    # no PCA), so this stays None and is saved as an empty array.
                    _sysrem_passes_per_order = (
                        np.full(n_orders, _si, int)
                        if cfg.pipeline.name in {"ASL19", "Gibson22", "Cheverall26"}
                        else None
                    )
                    _mask_store = np.full(
                        (_nn, n_orders, n_pixels), False, dtype=bool)
                    _useful_spectral_points_store = np.full(
                        (_nn, n_orders, n_pixels), False, dtype=bool)
                    _mask_snr_store = np.full(
                        (_nn, n_orders, n_pixels), False, dtype=bool)
                    _useful_spectral_points_snr_store = np.full(
                        (_nn, n_orders, n_pixels), False, dtype=bool)
                    _mask_inter_store = np.full(
                        (_nn, n_orders, n_pixels), False, dtype=bool)
                    _useful_spectral_points_inter_store = np.full(
                        (_nn, n_orders, n_pixels), False, dtype=bool)

                if _dn:
                    _U_sysrem[b, n_spectra:, :] = np.nan

                # Update SNR mask store
                _mask_snr_1d_int = np.asarray(_mask_snr_1d, dtype=int)
                if _mask_snr_1d_int.shape != (0,):
                    _mask_snr_store[b, h, _mask_snr_1d_int] = True
                _useful_spectral_points_snr_store[b, h, _useful_snr] = True

                # 5d.11, preparing_pipeline (SYSREM / PCA)
                # Build mini-dicts here (inside loop) so berv reflects
                # the current night's value after Different_nights restore
                _opt = cfg.pipeline.optimize_sysrem_order_by_order
                _mini_inj = {
                    "Kp_Vrest_inj": cfg.pipeline.kp_vrest_injection,
                    "BERV":         berv,
                    "V_sys":        planet.systemic_velocity_kms,
                    "event":              cfg.observation.event_type,
                    "Scale_inj":          cfg.observation.scale_injection,
                    "Inject_Scale_Factor": cfg.pipeline.inject_scale_factor,
                    "Limb_asymmetries":   cfg.atmosphere.limb_asymmetries,
                    "Limb_divisions":     cfg.atmosphere.limb_divisions,
                    "res":                inst.res,
                }
                _mini_pp = {
                    "preparing_pipeline":    cfg.pipeline.name,
                    "continuum_method":      cfg.pipeline.continuum_method,
                    "detrend_method":        cfg.pipeline.detrend_method,
                    "BERV":                  berv,
                    "V_sys":                 planet.systemic_velocity_kms,
                    "sysrem_its":            _si,
                    "Opt_PCA_its_ord_by_ord": _opt,
                    "Opt_criterion":         cfg.pipeline.optimize_criterion,
                    "sysrem_delta_sigma_threshold": cfg.pipeline.sysrem_delta_sigma_threshold,
                    "SYSREM_robust_halt":    cfg.pipeline.sysrem_robust_halt,
                    "Use_real_data":         _urd,
                    "Kp_Vrest_inj":          cfg.pipeline.kp_vrest_injection,
                    "telluric_variation":    cfg.tellurics.include_tellurics,
                    "telluric_mask":         cfg.tellurics.mask_threshold,
                    "safety_window":         cfg.tellurics.safety_window_pixels,
                    "Different_nights":      _dn,
                    "n_nights":              _nn,
                    "n_orders":              n_orders,
                }
                _mini_sysrem = {
                    "Different_nights": _dn,
                    "Kp_Vrest_inj":     cfg.pipeline.kp_vrest_injection,
                    "n_nights":         _nn,
                    "n_orders":         n_orders,
                    "sysrem_its":       _si,
                }

                # Optional: inject test signal for order-by-order optimisation.
                # DeltaSigma is model-independent, no injection needed.
                _opt_needs_injection = (
                    _opt and
                    cfg.pipeline.optimize_criterion != "DeltaSigma"
                )
                if _opt_needs_injection:
                    _mat_noise_inj_b5 = np.copy(_mat_noise_b5)
                    if not _dn:
                        # mat_star is (n_spectra, n_pixels) on the single-night
                        # path (Block 4a) and must be passed whole, exactly as
                        # the Block 4d injection does (line ~1476).  Indexing
                        # mat_star[b] here would collapse it to 1D.
                        _mat_noise_inj_b5[b, :] = injection(
                            _mini_inj, wave_ins,
                            _mat_noise_inj_b5[b, :], wave_pRT, syn_spec,
                            with_signal, without_signal, fraction, phase,
                            mat_star, T_0, syn_jd,
                        )
                    else:
                        _mat_noise_inj_b5[b, :] = injection(
                            _mini_inj, wave_ins,
                            _mat_noise_inj_b5[b, :n_spectra, :],
                            wave_pRT, syn_spec,
                            with_signal, without_signal, fraction, phase,
                            mat_star, transit_mid_JD[b], syn_jd,
                        )
                    _inj_slice_b5 = (
                        _mat_noise_inj_b5[b, :n_spectra, :] if _dn
                        else _mat_noise_inj_b5[b, :]
                    )
                else:
                    _mat_noise_inj_b5 = None
                    _inj_slice_b5     = None
                    if _opt and cfg.pipeline.optimize_criterion == "DeltaSigma":
                        _mat_noise_inj_b5 = None

                # mat_res / propag_noise allocation (b==0 per order).
                # Injection-based criteria (Maximum/Max_Diff) need a 5D array
                # (with injection × SYSREM iteration axes); DeltaSigma uses
                # standard 3D, same shape as the non-opt path.
                if b == 0:
                    if not _dn:
                        _mr_shape = ((_nn, n_spectra, n_pixels, 2, _si)
                                     if _opt_needs_injection
                                     else (_nn, n_spectra, n_pixels))
                        _mat_res_b5      = np.zeros(_mr_shape, float)
                        _propag_noise_b5 = np.zeros(
                            (_nn, n_spectra, n_pixels), float)
                    else:
                        _max_sp2 = int(np.max(_n_spectra_store_b5))
                        _mr_shape = ((_nn, _max_sp2, n_pixels, 2, _si)
                                     if _opt_needs_injection
                                     else (_nn, _max_sp2, n_pixels))
                        _mat_res_b5      = np.zeros(_mr_shape, float)
                        _propag_noise_b5 = np.zeros(
                            (_nn, _max_sp2, n_pixels), float)
                        _mat_star_forfile = np.zeros(
                            (_nn, _max_sp2, n_pixels), float)

                if _dn:
                    _mat_star_forfile[b, :n_spectra, :] = mat_star
                    _mat_res_b5[b, n_spectra:, :] = np.nan
                    _propag_noise_b5[b, n_spectra:, :] = np.nan
                    _mat_star_forfile[b, n_spectra:, :] = np.nan

                if len(_useful_snr) == 0:
                    _wc = float(wave_ins[n_pixels // 2])
                    _fully_masked_orders.append((h, b, _wc))
                    print(
                        f"  [mask] Order {h} night {b} "
                        f"(λ ≈ {_wc:.3f} µm): fully masked, no usable pixels "
                        f"after telluric/SNR masking. "
                        + ("Check warnings folder for full report."
                           if len(_fully_masked_orders) == 1 else "")
                    )
                    if _mat_res_b5 is not None:
                        if _dn:
                            _mat_res_b5[b, :n_spectra, :]      = np.nan
                            _propag_noise_b5[b, :n_spectra, :] = np.nan
                        else:
                            _mat_res_b5[b, :]      = np.nan
                            _propag_noise_b5[b, :] = np.nan
                    continue

                # Call preparing_pipeline
                _sl = slice(None, n_spectra) if _dn else slice(None)
                _pp_result = preparing_pipeline(
                    _mini_pp, _mat_noise_b5[b, _sl],
                    _std_noise_b5[b, _sl], wave_ins,
                    _useful_snr, _mask_snr_1d, airmass, phase,
                    without_signal, None, _inj_slice_b5,
                    tell_mask_threshold_Blain24=0.8,
                    max_fit_BL19=False, sysrem_division=False,
                    masks=True, correct_uncertainties=True,
                )
                if not cfg.pipeline.sysrem_robust_halt:
                    (
                        _mat_res_b5[b, _sl],
                        _propag_noise_b5[b, _sl],
                        _useful_spectral_points_b5,
                        _mask_b5, _n_passes_b5,
                        _inter_mask_b5, _inter_useful_b5,
                        _cor_b5,
                        _U_sysrem[b, _sl],
                    ) = _pp_result
                    _sysrem_pass_b5 = None
                    # Record actual passes used for this order (may be < _si
                    # when DeltaSigma halting is active).
                    if (_opt and _n_passes_b5 is not None
                            and _n_passes_b5 > 0
                            and _sysrem_passes_per_order is not None
                            and b == 0):
                        _sysrem_passes_per_order[h] = int(_n_passes_b5)
                else:
                    (
                        _mat_res_b5[b, _sl],
                        _propag_noise_b5[b, _sl],
                        _useful_spectral_points_b5,
                        _mask_b5, _sysrem_pass_b5,
                        _inter_mask_b5, _inter_useful_b5,
                    ) = _pp_result

                # Update mask stores
                if _mask_b5.shape != (0,):
                    _mask_store[b, h, _mask_b5] = True
                _useful_spectral_points_store[b, h,
                    _useful_spectral_points_b5] = True
                if _inter_mask_b5.shape != (0,):
                    _mask_inter_store[b, h, _inter_mask_b5] = True
                _useful_spectral_points_inter_store[b, h,
                    _inter_useful_b5] = True

                # Consistency check
                if not np.array_equal(
                        _mask_b5,
                        np.where(_mask_store[b, h, :])[0]):
                    raise RuntimeError(
                        f"Block 5: mask inconsistency at order {h}, "
                        f"night {b}")

                # 5d.12, CCF template matrix (v_cc, mat_cc, mat_back)
                _vcc_wind = cc.wind_velocity_kms
                if cfg.cross_correlation.cc_metric:
                    if not _dn:
                        if b == 0:
                            if not cfg.observation.significant_eccentricity:
                                _v_cc_b5 = get_V(
                                    planet.kp_kms, phase, berv,
                                    planet.systemic_velocity_kms, _vcc_wind,
                                )
                            else:
                                _v_cc_b5 = get_V_eccentric(
                                    planet.kp_kms, phase,
                                    planet.eccentricity,
                                    planet.argument_of_periastron_deg,
                                    berv,
                                    planet.systemic_velocity_kms, _vcc_wind,
                                )
                            _mat_cc_b5   = np.zeros(
                                (n_spectra, n_pixels), float)
                            _mat_back_b5 = np.zeros_like(_mat_cc_b5)
                            # Build the template for ALL
                            # exposures (no lightcurve weighting) so out-of-
                            # transit columns carry the fixed template and give
                            # real-noise CCFs for the duration test.  Other
                            # pipelines keep in-transit-only.
                            _ws_ccf0 = (np.arange(n_spectra)
                                        if cfg.pipeline.name == "Cheverall26"
                                        else with_signal)
                            _mat_cc_b5, _ = spec_to_mat_fraction(
                                _mini_stmf, syn_jd,
                                planet.transit_epoch_bjd, _v_cc_b5,
                                wave_ins, wave_pRT_cc, spec_cc, mat_star,
                                _ws_ccf0, without_signal,
                                np.ones_like(fraction),
                                include_star=False, ccf_setup=True,
                            )

                            if cfg.pipeline.prepare_template:
                                # ASL19 and Gibson22: use Gibson22 projector
                                # (Gibson et al. 2022, Eq. 7) to apply the same
                                # SYSREM filter to the template without running
                                # SYSREM on it directly (which would destroy it).
                                # BL19/Blain24: run preparing_pipeline as normal.
                                if cfg.pipeline.name in ("ASL19", "Gibson22"):
                                    # Use actual passes for this order (may be
                                    # < _si when DeltaSigma halting was active).
                                    _n_proj = (
                                        int(_sysrem_passes_per_order[h])
                                        if _sysrem_passes_per_order is not None
                                        else _si
                                    )
                                    _mini_sysrem_h = dict(_mini_sysrem)
                                    _mini_sysrem_h["sysrem_its"] = _n_proj
                                    _syn_mat_res_b5 = np.ones(
                                        (n_spectra, n_pixels), float)
                                    _Pf = np.zeros(
                                        (1, 1, n_spectra, n_spectra), float)
                                    _Nf = np.zeros(
                                        (1, 1, n_spectra, n_pixels), float)
                                    _Uf = np.zeros(
                                        (1, 1, n_spectra, _n_proj), float)
                                    _Nf[0, :, :, :] = _propag_noise_b5
                                    _Uf[0, :, :, :] = _U_sysrem[:, :, :_n_proj]
                                    _Pf[0, 0, :, :] = (
                                        SYSREM_filtering_projector_singleorder(
                                            _mini_sysrem_h, n_spectra,
                                            _Nf, _Uf,
                                        ))
                                    _syn_mat_res_b5 = (
                                        filter_model_singleorder(
                                            _Pf[0, 0, :, :], _mat_cc_b5,
                                            _useful_spectral_points_b5,
                                        ))
                                    del _Pf, _Uf, _Nf
                                else:
                                    _syn_mat_res_b5 = preparing_pipeline(
                                        _mini_pp, _mat_cc_b5,
                                        _std_noise_b5[b, :], wave_ins,
                                        _useful_spectral_points_b5,
                                        _mask_b5, airmass, phase,
                                        without_signal, _sysrem_pass_b5,
                                        None,
                                        tell_mask_threshold_Blain24=0.8,
                                        max_fit_BL19=False,
                                        sysrem_division=False,
                                        masks=False,
                                        correct_uncertainties=False,
                                    )
                            elif cfg.pipeline.name == "Cheverall26":
                                # Cheverall+26 Sec 2.5: "model spectra ...
                                # normalized in the same way as the spectral
                                # data".  Apply the SAME rescale + 2nd-order
                                # continuum normalisation as the data pipeline
                                # (but NOT the detrending, no model reprocessing
                                # for the CCF).  Gated to Cheverall26; other
                                # pipelines reprocess via the branch above.
                                from exoplore.pipelines.cheverall26 import (
                                    chev26_rescale as _ch_resc,
                                    chev26_normalise as _ch_norm,
                                    chev26_normalise_polyfit as _ch_normpf,
                                )
                                _gp_t = _useful_spectral_points_b5
                                _dummy_n = np.ones_like(_mat_cc_b5)
                                _t_norm, _ = _ch_resc(_mat_cc_b5, _dummy_n, _gp_t)
                                _normfn_t = (
                                    _ch_normpf
                                    if cfg.pipeline.continuum_method == "polyfit"
                                    else _ch_norm)
                                _t_norm, _ = _normfn_t(
                                    wave_ins, _t_norm, _dummy_n, _gp_t)
                                _syn_mat_res_b5 = _t_norm
                                # Do NOT flatten out-of-transit
                                #, keep the fixed (chev26-normalised) template on
                                # every exposure so the duration test gets real
                                # out-of-transit noise CCFs.
                            else:
                                _syn_mat_res_b5 = _mat_cc_b5
                                _syn_mat_res_b5[without_signal, :] = 1.01

                            for _i in range(n_spectra):
                                _mat_back_b5[_i, :] = np.interp(
                                    wave_ins,
                                    wave_ins * (
                                        1. - _v_cc_b5[_i]
                                        / (_cst_b5.c / 1e5)
                                    ),
                                    _syn_mat_res_b5[_i, :],
                                )

                    else:   # Different_nights
                        if not cfg.observation.significant_eccentricity:
                            _v_cc_b5 = get_V(
                                planet.kp_kms, phase,
                                np.asarray(berv, dtype=np.float64),
                                planet.systemic_velocity_kms, _vcc_wind,
                            )
                        else:
                            _v_cc_b5 = get_V_eccentric(
                                planet.kp_kms, phase,
                                planet.eccentricity,
                                planet.argument_of_periastron_deg,
                                berv,
                                planet.systemic_velocity_kms, _vcc_wind,
                            )
                        _mat_cc_b5   = np.zeros((n_spectra, n_pixels), float)
                        _mat_back_b5 = np.zeros_like(_mat_cc_b5)
                        _mat_cc_b5, _ = spec_to_mat_fraction(
                            _mini_stmf, syn_jd,
                            planet.transit_epoch_bjd, _v_cc_b5,
                            wave_ins, wave_pRT_cc, spec_cc, mat_star,
                            with_signal, without_signal,
                            fraction,
                            include_star=False, ccf_setup=True,
                        )

                        if cfg.pipeline.prepare_template:
                            if cfg.pipeline.name in ("ASL19", "Gibson22"):
                                _syn_mat_res_b5 = np.ones(
                                    (n_spectra, n_pixels), float)
                                _Pf = np.zeros(
                                    (1, 1, n_spectra, n_spectra), float)
                                _Nf = np.zeros(
                                    (1, 1, n_spectra, n_pixels), float)
                                _Uf = np.zeros(
                                    (1, 1, n_spectra, _si), float)
                                _Nf[0, 0, :, :] = (
                                    _propag_noise_b5[b, :n_spectra, :])
                                _Uf[0, 0, :, :] = (
                                    _U_sysrem[b, :n_spectra, :])
                                _Pf[0, 0, :, :] = (
                                    SYSREM_filtering_projector_singleorder(
                                        _mini_sysrem, n_spectra, _Nf, _Uf,
                                        no_night_loop=True,
                                    ))
                                _syn_mat_res_b5 = (
                                    filter_model_singleorder(
                                        _Pf[0, 0, :, :], _mat_cc_b5,
                                        _useful_spectral_points_b5,
                                    ))
                                del _Pf, _Uf, _Nf
                            else:
                                _syn_mat_res_b5 = preparing_pipeline(
                                    _mini_pp, _mat_cc_b5,
                                    _std_noise_b5[b, :n_spectra, :],
                                    wave_ins,
                                    _useful_spectral_points_b5,
                                    _mask_b5, airmass, phase,
                                    without_signal, _sysrem_pass_b5, None,
                                    tell_mask_threshold_Blain24=0.8,
                                    max_fit_BL19=False,
                                    sysrem_division=False,
                                    masks=False,
                                    correct_uncertainties=False,
                                )
                        else:
                            _syn_mat_res_b5 = np.copy(_mat_cc_b5)
                            _syn_mat_res_b5[without_signal, :] = 1.01

                        for _i in range(n_spectra):
                            _mat_back_b5[_i, :] = np.interp(
                                wave_ins,
                                wave_ins * (
                                    1. - _v_cc_b5[_i]
                                    / (_cst_b5.c / 1e5)
                                ),
                                _syn_mat_res_b5[_i, :],
                            )

                # ----------------------------------------------------------
                # Block 6, CCF computation
                # ----------------------------------------------------------
                _ccf_flag = cfg.cross_correlation.cc_metric
                if _ccf_flag:

                    # 6a, velocity grid
                    if _ccf_iterations is None:
                        _vstep_cfg = cfg.cross_correlation.velocity_step_kms
                        if _vstep_cfg is None:
                            _wave_cm    = wave_ins * 1e-4
                            _step_v     = (_cst_b5.c
                                           * np.diff(_wave_cm) / _wave_cm[:-1])
                            _ccf_v_step = np.round(
                                np.mean(_step_v) / 1e5, 1)
                        else:
                            _ccf_v_step = _vstep_cfg
                        _ccf_v_interval = cfg.cross_correlation.velocity_max_kms
                        _ccf_iterations = (
                            int(round(2 * _ccf_v_interval / _ccf_v_step))
                            + 1
                        )
                        if _ccf_iterations % 2 == 0:
                            _ccf_iterations += 1
                        _v_ccf = np.linspace(
                            -_ccf_v_interval, _ccf_v_interval,
                            num=_ccf_iterations, dtype=float,
                        )

                    # 6b, ccf_store allocation (b==0 per order)
                    if b == 0:
                        if not _dn:
                            _ccf_store_b6 = np.zeros(
                                ((_nn, _ccf_iterations, n_spectra, 2, _si)
                                 if _opt_needs_injection
                                 else (_nn, _ccf_iterations, n_spectra)),
                                float,
                            )
                        else:
                            _max_sp_ccf = int(np.max(_n_spectra_store_b5))
                            _ccf_store_b6 = np.zeros(
                                ((_nn, _ccf_iterations, _max_sp_ccf, 2, _si)
                                 if _opt_needs_injection
                                 else (_nn, _ccf_iterations, _max_sp_ccf)),
                                float,
                            )

                    if _dn:
                        _ccf_store_b6[b, :, n_spectra:] = np.nan

                    if _nn > 20 and b % 10 == 0:
                        print(f"  CCF order {order_selection[h]}, night {b}")

                    # 6c, CCF call (instrument-dependent)
                    _big_inst = cfg.instrument.name in [
                        'CARMENES_NIR', 'CARMENES_VIS', 'ANDES', 'CRIRES',
                        'IGRINS']
                    _norm_ccf = cfg.cross_correlation.normalized
                    # Kernel form for the weighted (large-instrument) path:
                    # normalised Pearson CCF (default) or the un-normalised
                    # matched filter Σ R·M/E² (Nortmann+24 Eq. 1).
                    _ccf_fn = (
                        call_ccf_numba_par_matched_filter
                        if cfg.cross_correlation.ccf_kernel == "matched_filter"
                        else call_ccf_numba_par_weighted)

                    # Error for the CCF inverse-variance weighting.  Default is
                    # the propagated per-pixel noise; "mad_residual" uses the
                    # per-wavelength-channel MAD of the time-series residuals
                    # (Gibson+20; Nortmann+24; Cheverall+26).  Applied on the
                    # standard (non-optimisation, 3D residual) path.
                    _ccf_unc_b5 = _propag_noise_b5
                    if (cfg.cross_correlation.ccf_error_estimate == "mad_residual"
                            and _mat_res_b5 is not None
                            and _mat_res_b5.ndim == 3):
                        _ccf_unc_b5 = np.full_like(_propag_noise_b5, np.inf)
                        for _bb in range(_propag_noise_b5.shape[0]):
                            _res = _mat_res_b5[_bb]            # (n_spec, n_pix)
                            _med = np.nanmedian(_res, axis=0)  # per channel
                            _mad = np.nanmedian(
                                np.abs(_res - _med[None, :]), axis=0)
                            _sig = 1.4826 * _mad
                            _sig[~np.isfinite(_sig) | (_sig <= 0)] = np.inf
                            _ccf_unc_b5[_bb] = _sig[None, :]

                    if not _dn:
                        if _big_inst:
                            if not _opt_needs_injection:
                                if _norm_ccf:
                                    _ccf_store_b6[b, :] = (
                                        _ccf_fn(
                                            lag=_v_ccf,
                                            n_spectra=n_spectra,
                                            obs=_mat_res_b5[b, :],
                                            ccf_iterations=_ccf_iterations,
                                            wave=wave_ins,
                                            wave_CC=wave_ins,
                                            template=_mat_back_b5,
                                            uncertainties=_ccf_unc_b5[
                                                b, :],
                                            with_signal=with_signal,
                                        ))
                                else:
                                    _ccf_store_b6[b, :] = (
                                        call_ccf_literature(
                                            lag=_v_ccf,
                                            n_spectra=n_spectra,
                                            obs=_mat_res_b5[b, :],
                                            ccf_iterations=_ccf_iterations,
                                            wave=wave_ins,
                                            wave_CC=wave_ins,
                                            template=_mat_back_b5,
                                            uncertainties=_ccf_unc_b5[
                                                b, :],
                                            with_signal=with_signal,
                                        ))
                            else:  # Opt_PCA_its_ord_by_ord
                                _ccf_store_b6[b, :] = (
                                    call_ccf_numba_par_weighted_ordbord_opt(
                                        _si,
                                        lag=_v_ccf,
                                        n_spectra=n_spectra,
                                        obs=_mat_res_b5[b, :],
                                        ccf_iterations=_ccf_iterations,
                                        wave=wave_ins,
                                        wave_CC=wave_ins,
                                        template=_mat_back_b5,
                                        uncertainties=_ccf_unc_b5[b, :],
                                    ))
                        else:
                            _ccf_store_b6[b, :] = call_ccf_numba(
                                lag=_v_ccf,
                                n_spectra=n_spectra,
                                obs=_mat_res_b5[b, :],
                                ccf_iterations=_ccf_iterations,
                                wave=wave_ins,
                                wave_CC=wave_ins,
                                template=_mat_back_b5,
                            )

                        # 6d, median subtraction
                        _ccf_store_b6[b, :, :] -= np.median(
                            _ccf_store_b6[b, :], axis=0)

                        # 6e, dayside v_rotsini masking
                        if (cfg.observation.event_type == 'dayside'
                                and cfg.observation.mask_v_rotsini):
                            _v_rot = planet.v_rotsini_kms or 0.0
                            _mv = np.where(np.logical_and(
                                _v_ccf > -_v_rot,
                                _v_ccf < _v_rot,
                            ))[0]
                            _ccf_store_b6[b, _mv, :] = np.median(
                                _ccf_store_b6[b, :])

                        # 6f, NaN / all-zero guard
                        if (not np.isfinite(_ccf_store_b6[b, :]).any()
                                or np.all(_ccf_store_b6[b, :] == 0)):
                            _ccf_store_b6[b, :, :] *= 0
                            _wdir = str(dirs["warnings"]) + "/"
                            os.makedirs(_wdir, exist_ok=True)
                            _wfn = f"{_wdir}ccf_values_{sim_name}.fits"
                            with open(_wfn, 'w') as _wfh:
                                _wfh.write(
                                    f"Full CCF_Values matrix was NaN for"
                                    f" order {order_selection[h]}"
                                    f" in night {b}. Spectral matrix fully"
                                    f" masked? Check."
                                )

                    else:  # Different_nights
                        if _big_inst:
                            if not _opt_needs_injection:
                                if _norm_ccf:
                                    _ccf_store_b6[b, :, :n_spectra] = (
                                        _ccf_fn(
                                            lag=_v_ccf,
                                            n_spectra=n_spectra,
                                            obs=_mat_res_b5[b, :n_spectra, :],
                                            ccf_iterations=_ccf_iterations,
                                            wave=wave_ins,
                                            wave_CC=wave_ins,
                                            template=_mat_back_b5,
                                            uncertainties=_ccf_unc_b5[
                                                b, :n_spectra, :],
                                            with_signal=with_signal,
                                        ))
                                else:
                                    _ccf_store_b6[b, :, :n_spectra] = (
                                        call_ccf_literature(
                                            lag=_v_ccf,
                                            n_spectra=n_spectra,
                                            obs=_mat_res_b5[b, :n_spectra, :],
                                            ccf_iterations=_ccf_iterations,
                                            wave=wave_ins,
                                            wave_CC=wave_ins,
                                            template=_mat_back_b5,
                                            uncertainties=_ccf_unc_b5[
                                                b, :n_spectra, :],
                                            with_signal=with_signal,
                                        ))
                            else:  # Opt_PCA_its_ord_by_ord
                                _ccf_store_b6[b, :, :n_spectra] = (
                                    call_ccf_numba_par_weighted_ordbord_opt(
                                        _si,
                                        lag=_v_ccf,
                                        n_spectra=n_spectra,
                                        obs=_mat_res_b5[b, :n_spectra, :],
                                        ccf_iterations=_ccf_iterations,
                                        wave=wave_ins,
                                        wave_CC=wave_ins,
                                        template=_mat_back_b5,
                                        uncertainties=_ccf_unc_b5[
                                            b, :n_spectra, :],
                                    ))
                        else:
                            _ccf_store_b6[b, :, :n_spectra] = (
                                call_ccf_numba(
                                    lag=_v_ccf,
                                    n_spectra=n_spectra,
                                    obs=_mat_res_b5[b, :n_spectra, :],
                                    ccf_iterations=_ccf_iterations,
                                    wave=wave_ins,
                                    wave_CC=wave_ins,
                                    template=_mat_back_b5,
                                ))

                        # 6d, median subtraction (Different_nights)
                        _ccf_store_b6[b, :, :n_spectra] -= np.median(
                            _ccf_store_b6[b, :, :n_spectra], axis=0)

                        # 6e, dayside v_rotsini masking
                        if (cfg.observation.event_type == 'dayside'
                                and cfg.observation.mask_v_rotsini):
                            _v_rot = planet.v_rotsini_kms or 0.0
                            _mv = np.where(np.logical_and(
                                _v_ccf > -_v_rot,
                                _v_ccf < _v_rot,
                            ))[0]
                            _ccf_store_b6[b, _mv, :n_spectra] = np.median(
                                _ccf_store_b6[b, :, :n_spectra])

                        # 6f, NaN / all-zero guard
                        if (not np.isfinite(
                                    _ccf_store_b6[b, :, :n_spectra]).any()
                                or np.all(
                                    _ccf_store_b6[b, :, :n_spectra] == 0)):
                            _ccf_store_b6[b, :, :n_spectra] *= 0
                            _wdir = str(dirs["warnings"]) + "/"
                            os.makedirs(_wdir, exist_ok=True)
                            _wfn = f"{_wdir}ccf_values_{sim_name}.fits"
                            with open(_wfn, 'w') as _wfh:
                                _wfh.write(
                                    f"Full CCF_Values matrix was NaN for"
                                    f" order {order_selection[h]}"
                                    f" in night {b}. Spectral matrix fully"
                                    f" masked? Check."
                                )

                # End of Block 6 per-night loop iteration

            # -- end for b -------------------------------------------------

            # 6g, save ccf_store per-order to disk (in-memory accumulation for
            # the Kp-Vsys co-add is separate and unaffected by cfg.output)
            if _ccf_flag and cfg.output.save_ccf_store:
                _base_dir_b6 = str(dirs["matrices"])
                os.makedirs(_base_dir_b6, exist_ok=True)
                np.savez_compressed(
                    f"{_base_dir_b6}/ccf_store_order_"
                    f"{order_selection[h]}_{sim_name}",
                    a=_ccf_store_b6,
                )

            # 6h, restore Different_nights per-night arrays for next order
            if _dn:
                n_spectra      = _n_spectra_store_b5.copy()
                phase          = _phase_store5[:]
                with_signal    = _with_signal_store5[:]
                without_signal = _without_signal_store5[:]
                airmass        = _airmass_store5[:]
                fraction       = _fraction_store5[:]
                syn_jd         = _syn_jd_store5[:]
                if berv_store is not None:
                    berv       = berv_store[:]

            # Update per-order results with Block 5 + Block 6 outputs
            _order_result.update({
                "mat_noise":    _mat_noise_b5,
                # Deterministic noiseless observed matrix (mat_noise minus the
                # Gaussian draw), i.e. the same data without noise, for the
                # pipeline-steps diagnostic.
                "mat_noiseless_obs": (
                    _mat_noise_b5 - _gauss_noise_b5
                    if (not cfg.observation.noiseless
                        and not cfg.observation.use_real_data)
                    else _mat_noise_b5),
                "std_noise":    _std_noise_b5,
                "mat_res":      _mat_res_b5,
                "propag_noise": _propag_noise_b5,
                "mask_snr":     _mask_snr_1d,
                "useful_spectral_points_snr": _useful_snr,
                "spec_cc":      spec_cc,
                "mat_cc":       _mat_cc_b5,
                "mat_back":     _mat_back_b5,
                "syn_mat_res":  _syn_mat_res_b5 if _ccf_flag else None,
                "v_cc":         _v_cc_b5,
                # Block 6
                "ccf_store":    _ccf_store_b6 if _ccf_flag else None,
                "v_ccf":        _v_ccf,
                # SYSREM eigenvectors for retrieval projector (ASL19/Gibson22)
                "U_sysrem": (np.copy(_U_sysrem)
                             if _U_sysrem is not None else None),
            })

            if (h + 1) % max(1, n_orders // 5) == 0 or h == n_orders - 1:
                print(f"  Order {h + 1:3d}/{n_orders}, Blocks 3-6 done")

        # End of per-order loop

        # ----------------------------------------------------------------
        # Pipeline-steps diagnostic plot (Fig. 5 style)
        # Generated once per run using the first processed order.
        # ----------------------------------------------------------------
        try:
            from exoplore.plotting.steps import plot_pipeline_steps
            # Displayed order: the user's pipeline_steps_order if it was
            # processed in this run, otherwise the first processed order.
            _h_steps = 0
            _ord_req = cfg.plotting.pipeline_steps_order
            if _ord_req is not None:
                _osel = list(order_selection)
                if _ord_req in _osel:
                    _h_steps = _osel.index(_ord_req)
                # else: requested order not processed -> first order
            _or_steps = per_order_results[_h_steps]
            _wave_steps = _or_steps["wave_ins"]
            _usp_steps  = np.where(
                _useful_spectral_points_store[0, _h_steps, :])[0]
            _mn_steps = _or_steps.get("mat_noiseless_obs")
            _my_steps = _or_steps.get("mat_noise")
            _mr_steps = _or_steps.get("mat_res")
            if (_mn_steps is not None and _my_steps is not None
                    and _mr_steps is not None
                    and len(_usp_steps) > 0):
                # For different_nights, take night 0 slice
                _mn_steps = _mn_steps[0] if _mn_steps.ndim == 3 else _mn_steps
                _my_steps = _my_steps[0] if _my_steps.ndim == 3 else _my_steps
                _mr_steps = _mr_steps[0] if _mr_steps.ndim == 3 else _mr_steps
                _ph_steps  = phase[0] if _dn else phase
                _ws_steps  = with_signal[0] if _dn else with_signal
                _wo_steps  = without_signal[0] if _dn else without_signal
                _ord_lbl   = f"order {order_selection[_h_steps]}"

                # SYSREM pipelines: reconstruct the per-iteration residuals so
                # the diagnostic becomes a stacked iteration waterfall.  Uses
                # apply_sysrem on the displayed order only (no change to the
                # production preparation).
                _sysrem_stages = None
                _sysrem_iters = None
                if cfg.pipeline.name in ("ASL19", "Gibson22"):
                    try:
                        from exoplore.plotting.steps import (
                            reconstruct_sysrem_stages)
                        _noise_steps = _or_steps.get("std_noise")
                        if _noise_steps is not None:
                            _noise_steps = (_noise_steps[0]
                                            if _noise_steps.ndim == 3
                                            else _noise_steps)
                            _sysrem_iters = list(
                                cfg.plotting.pipeline_steps_sysrem_iterations)
                            _sysrem_stages = reconstruct_sysrem_stages(
                                _wave_steps, _my_steps, _noise_steps,
                                _usp_steps, _sysrem_iters,
                                use_normalised_errors=(
                                    cfg.pipeline.name == "ASL19"),
                            )
                    except Exception as _e_rec:
                        print(f"  NOTE: SYSREM stages skipped ({_e_rec})")
                        _sysrem_stages = None

                plot_pipeline_steps(
                    sim_name          = sim_name,
                    plots_dir         = str(dirs["plots"]) + "/",
                    wave_ins          = _wave_steps,
                    phase             = _ph_steps,
                    with_signal       = _ws_steps,
                    without_signal    = _wo_steps,
                    useful_spectral_points = _usp_steps,
                    mat_noiseless     = _mn_steps,
                    mat_noisy         = _my_steps,
                    mat_residual      = _mr_steps,
                    order_label       = _ord_lbl,
                    xlim_1d           = (tuple(cfg.plotting.pipeline_steps_xlim_um)
                                         if cfg.plotting.pipeline_steps_xlim_um
                                         else None),
                    sysrem_stages     = _sysrem_stages,
                    sysrem_iters      = _sysrem_iters,
                    use_real_data     = cfg.observation.use_real_data,
                    save_plot         = True,
                    show_plot         = False,
                )
        except Exception as _e_steps:
            print(f"  NOTE: pipeline_steps plot skipped ({_e_steps})")

        # ----------------------------------------------------------------
        # Fully-masked order handling
        # Orders where every pixel was masked (SNR + negative-value) on
        # at least one night are logged to a warnings file and excluded
        # from all downstream processing.  The simulation continues with
        # the remaining good orders.
        # ----------------------------------------------------------------
        # Compute the set of h-indices that are bad on ALL their nights
        # (i.e. ccf_store is still None, never computed, because every
        # `b` iteration hit the masked-order guard).
        _fully_masked_h_set: set = {
            h for h in range(n_orders)
            if per_order_results[h].get("ccf_store") is None
            and cfg.cross_correlation.cc_metric
        }

        if _fully_masked_orders:
            _warn_dir = str(dirs["warnings"]) + "/"
            os.makedirs(_warn_dir, exist_ok=True)
            _warn_path = os.path.join(
                _warn_dir, f"fully_masked_orders_{sim_name}.txt")
            with open(_warn_path, "w") as _wf:
                _wf.write(
                    f"EXoPLORE, fully masked orders report\n"
                    f"Simulation: {sim_name}\n"
                    f"{'=' * 60}\n\n"
                    f"The following (order, night) pairs had no usable pixels\n"
                    f"after SNR + negative-value masking.  They were excluded\n"
                    f"from CCF accumulation, retrievals, and significance maps.\n\n"
                )
                for h, b, wc in _fully_masked_orders:
                    _wf.write(
                        f"  Order index {h:3d}  (order_selection="
                        f"{order_selection[h]})  "
                        f"night {b}, wavelength centre {wc:.4f} µm\n"
                    )
                _wf.write(
                    f"\nTo suppress these orders entirely, remove their\n"
                    f"order_selection indices from 'order_indices' in the\n"
                    f"config, or lower 'snr_mask_threshold'.\n"
                )
            print(
                f"\n  WARNING: {len(_fully_masked_orders)} (order, night) "
                f"pair(s) fully masked, excluded from CCF and retrieval.\n"
                f"  Details written to: {_warn_path}\n"
            )

        # ----------------------------------------------------------------
        # Store Blocks 3 to 6 state
        # ----------------------------------------------------------------
        self._state.update(dict(
            per_order_results=per_order_results,
            # Block 3, limb scaling factors
            sf_morning=_sf_morning,
            sf_evening=_sf_evening,
            ingress_idx=_ingress_idx,
            egress_idx=_egress_idx,
            # Block 3, kernel stores
            kernel_wind_morning_store=_kernel_wind_morning_store,
            kernel_wind_evening_store=_kernel_wind_evening_store,
            kernel_rot_morning_store=_kernel_rot_morning_store,
            kernel_rot_evening_store=_kernel_rot_evening_store,
            # Block 4, loop-persistent variables
            T_0=T_0,
            v_planet=v_planet,
            berv=berv,
            fraction=fraction,
            berv_store=berv_store,
            v_planet_store=v_planet_store,
            # Block 5, accumulated mask stores (n_nights × n_orders × n_pixels)
            mask_store=_mask_store,
            useful_spectral_points_store=_useful_spectral_points_store,
            mask_snr_store=_mask_snr_store,
            useful_spectral_points_snr_store=_useful_spectral_points_snr_store,
            mask_inter_store=_mask_inter_store,
            useful_spectral_points_inter_store=_useful_spectral_points_inter_store,
            U_sysrem=_U_sysrem,
            mat_star_forfile=_mat_star_forfile,
            # Block 6, CCF velocity grid
            v_ccf=_v_ccf,
            ccf_iterations=_ccf_iterations,
        ))

        # Also store _ccf_v_step for downstream use
        self._state["ccf_v_step"] = _ccf_v_step

        _t_blocks["Blocks 3-6, atm forward model, matrices, pipeline, CCF"] = (
            _time.time() - _t_run_start - sum(_t_blocks.values()))
        print(
            f"\n  Blocks 3-6 complete, atmospheric model, stellar matrix, "
            f"tellurics, noise, SYSREM pipeline, CCF done ({n_orders} orders)."
        )
        if cfg.timing:
            print(f"  [timing] Blocks 3-6: {_t_blocks['Blocks 3-6, atm forward model, matrices, pipeline, CCF']:.1f} s  "
                  f"({_t_blocks['Blocks 3-6, atm forward model, matrices, pipeline, CCF']/60:.2f} min)")

        # ================================================================
        # Block 7, CCF statistics and Kp-Vsys maps
        # ================================================================
        # This block:
        #   7a  Assemble full ccf_store (in-memory, no disk round-trip)
        #   7b  CO-ADD spectral orders → ccf_nights
        #   7c  CO-ADD nights → ccf_complete  +  CCF_matrix_ERF plot
        #   7d  Build v_rest and kp_range
        #   7e  get_shifted_ccf_matrix → ccf_values_shift (Kp grid)
        #   7f  Significance evaluation
        #         CCF_SNR only → get_max_CCF_peak + plots
        #         Welch_ttest only → Welch_ttest_map + plots
        #         All_significance_metrics → both branches
        #   7g  Opt_PCA_its_ord_by_ord: optimise SYSREM iterations
        #   7h  Statistical study (n_nights > 1)
        #   NOTE: Study_velocity_ranges and Noise_statistical_study are
        #         specialised optional analyses that require gauss_noise to
        #         be persisted per-order. These are not yet wired; set
        #         inp_dat["Study_velocity_ranges"] = False and
        #         inp_dat["Noise_statistical_study"] = False to skip.
        # ================================================================

        # Mini-dict for Block 7/8/9 functions that still take inp_dat.
        # Covers all keys used by ccf/, plotting/, and analysis/ modules.
        # CCF_SNR and Welch_ttest start from config; All_significance_metrics
        # path temporarily flips them, update _mini_b7 at those points.
        _mini_b7: dict = {
            "BERV":                  berv,
            "CCF_SNR":               cfg.cross_correlation.ccf_snr,
            "CCF_SNR_exclude":       cfg.cross_correlation.snr_exclude_kms,
            "ccf_snr_exclude_around": ("point"
                                       if cfg.pipeline.name == "Cheverall26"
                                       else "peak"),
            "CCF_V_STEP":            cfg.cross_correlation.velocity_step_kms,
            "Different_nights":      _dn,
            "K_p":                   planet.kp_kms,
            "MAX_CCF_V_STD":         cfg.cross_correlation.noise_velocity_max_kms,
            "Opt_PCA_its_ord_by_ord": _opt,
            "Opt_crit":              cfg.pipeline.optimize_criterion,
            "PLOT_CCF_XSTEP":        cfg.cross_correlation.plot_ccf_xstep,
            "Simulation_name":       sim_name,
            "Stack_Group_Size":      cfg.statistics.stack_group_size,
            "V_sys":                 planet.systemic_velocity_kms,
            "V_wind":                pm.wind_velocity_kms,
            "Welch_ttest":           cfg.cross_correlation.welch_ttest,
            "All_significance_metrics": cfg.cross_correlation.all_significance_metrics,
            "SSIM_metric":           cfg.cross_correlation.ssim_metric,
            "Study_velocity_ranges": cfg.cross_correlation.study_velocity_ranges,
            "arg_periastron_w":      planet.argument_of_periastron_deg,
            "eccentricity":          planet.eccentricity,
            "event":                 cfg.observation.event_type,
            "first_night_noiseless": cfg.observation.first_night_noiseless,
            "in_trail_left_right":   cfg.cross_correlation.in_trail_left_right,
            "kp_max":                cfg.cross_correlation.kp_max_kms,
            "n_nights":              _nn,
            "n_orders":              n_orders,
            "order_selection":       order_selection,
            "plots_dir":             str(dirs["plots"]) + "/",
            "matrix_dir":            str(dirs["matrices"]) + "/",
            "home_dir":              str(dirs["root"]) + "/",
            "correlations_dir":      str(dirs["correlations"]) + "/",
            "warnings_dir":          str(dirs["warnings"]) + "/",
            "significant_eccentricity": cfg.observation.significant_eccentricity,
            "sysrem_its":            _si,
            "Kp_Vrest_inj":          cfg.pipeline.kp_vrest_injection,
            "Noise_scaling_factor":  cfg.observation.noise_scaling_factor,
            "T_equ":                 pm.equilibrium_temperature_K,
            "vmr":                   pm.vmr,
            "R":                     inst.res,
            "Resolving_power":       inst.res,
            "resolution":            inst.res,
            # Retrieval (Block 9)
            "Perform_retrieval":     cfg.retrieval.enabled,
            "Ret_dim":               cfg.retrieval.dimensionality,
            "logL_choice":           cfg.retrieval.log_likelihood,
            "ret_error_model":       getattr(cfg.retrieval, "error_model", "propagated"),
            # Normalise sampler string to internal canonical form
            "Sampler_choice":        (
                "Nested_sampling"
                if cfg.retrieval.sampler.lower().replace("-", "_")
                   in ("nested_sampling", "multinest", "nested")
                else "MCMC"
            ),
            "Retrieval_choice":      cfg.retrieval.retrieval_choice,
            "Multinest_live_points": cfg.retrieval.live_points,
            "Multinest_Constant_Eff_Mode": cfg.retrieval.constant_efficiency_mode,
            "n_walkers":             cfg.retrieval.n_walkers,
            "n_steps_MCMC":          cfg.retrieval.n_steps,
            "MCMC_burnin":           cfg.retrieval.burnin,
            "PRIOR_BOUNDS":          cfg.retrieval.prior_bounds,
            "Limb_asymmetries":      cfg.atmosphere.limb_asymmetries,
            "SYSREM_robust_halt":    cfg.pipeline.sysrem_robust_halt,
            "prepare_template":      cfg.pipeline.prepare_template,
            "preparing_pipeline":    cfg.pipeline.name,
            # Keys required by preparing_pipeline when called inside loglike
            "telluric_variation":    cfg.tellurics.include_tellurics,
            "telluric_mask":         cfg.tellurics.mask_threshold,
            "safety_window":         cfg.tellurics.safety_window_pixels,
            "Use_real_data":         cfg.observation.use_real_data,
            "Opt_PCA_its_ord_by_ord": cfg.pipeline.optimize_sysrem_order_by_order,
            "Opt_criterion":         cfg.pipeline.optimize_criterion,
            "sysrem_delta_sigma_threshold": cfg.pipeline.sysrem_delta_sigma_threshold,
            "T_equ_morning_day":     _md.equilibrium_temperature_K,
            "T_equ_evening_day":     _ed.equilibrium_temperature_K,
            "species_ret_morning":   _md.species,
            "species_ret_evening":   _ed.species,
            "vmr_morning_day":       _md.vmr,
            "vmr_evening_day":       _ed.vmr,
            # Retrieval atmosphere (ret_atm = ret.atmosphere or planet_model)
            "species_ret":           (cfg.retrieval.atmosphere.species
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.species),
            "MMW_ret":               (cfg.retrieval.atmosphere.mean_molecular_weight
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.mean_molecular_weight),
            "p0_ret":                (cfg.retrieval.atmosphere.reference_pressure_bar
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.reference_pressure_bar),
            "isothermal_ret":        (cfg.retrieval.atmosphere.isothermal
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.isothermal),
            "two_point_T_ret":       (cfg.retrieval.atmosphere.two_point
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.two_point),
            "p_points_ret":          (cfg.retrieval.atmosphere.two_point_pressures_bar
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.two_point_pressures_bar),
            "t_points_ret":          (cfg.retrieval.atmosphere.two_point_temperatures_K
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.two_point_temperatures_K),
            "Kappa_IR_ret":          (cfg.retrieval.atmosphere.kappa_ir
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.kappa_ir),
            "Gamma_ret":             (cfg.retrieval.atmosphere.gamma_guillot
                                      if cfg.retrieval.atmosphere is not None
                                      else pm.gamma_guillot),
        }

        # Pre-Block-7 outputs (initialised to None; set only when paths run)
        ccf_values_shift = None
        v_rest            = None
        _kp_range         = None
        _n_kp             = None
        _pix_lr_b7        = None
        sysrem_it_opt     = None
        ccf_complete      = None
        _ccf_all_orders   = None
        stats             = None
        shuffled_nights   = None
        _night_min        = 0
        _night_max        = 0
        ccf_tot_sig       = None
        max_sig           = None
        max_kp_idx        = None
        max_v_wind        = None
        v_rest_sigma      = None

        if cfg.cross_correlation.ssim_metric:
            print("  SSIM_metric is True, skipping Block 7 (Kp-Vsys maps).")

        elif _ccf_flag:

            # -------------------------------------------------------
            # 7a, Assemble full ccf_store from per_order_results
            # -------------------------------------------------------
            _base_dir_b7 = str(dirs["matrices"])
            os.makedirs(_base_dir_b7, exist_ok=True)

            # Guard: if every (order, night) pair was fully masked the
            # velocity grid was never computed and _ccf_iterations is
            # still None.  This means no CCF data exists, there is
            # nothing to assemble or save.  This can happen when the
            # telluric mask thresholds are very aggressive, the SNR
            # floor is set too high, or the wavelength range falls
            # entirely inside deep telluric absorption bands.
            # Check the warnings file for the list of masked orders,
            # then review CCF_SNR_THRESHOLD / CCF_V_MAX / order_indices
            # in your config.
            if _ccf_iterations is None:
                raise RuntimeError(
                    "\n\n  All spectral orders are fully masked, no CCF "
                    "was computed and Block 7 cannot proceed.\n"
                    "  This is not a code bug: every order/night pair "
                    "failed the SNR or negative-value mask.\n"
                    "  Likely causes:\n"
                    "    · CCF_SNR_THRESHOLD is too high for this dataset\n"
                    "    · order_indices selects only deep telluric bands\n"
                    "    · input spectra contain unexpected NaN / zero rows\n"
                    "  Masked orders are listed in:\n"
                    f"    {str(dirs['warnings'])}/"
                    f"fully_masked_orders_{sim_name}.txt\n"
                )

            if not _dn:
                _ccf_shape7 = ((n_orders, _nn, _ccf_iterations, n_spectra, 2, _si)
                               if _opt_needs_injection
                               else (n_orders, _nn, _ccf_iterations, n_spectra))
                _ccf_all_orders = np.full(_ccf_shape7, np.nan, float)
                for _h7 in range(n_orders):
                    if _h7 in _fully_masked_h_set:
                        continue
                    _ccf_all_orders[_h7] = per_order_results[_h7]["ccf_store"]
            else:
                _max_nspec_b7 = int(np.max(n_spectra))
                _ccf_shape7 = ((n_orders, _nn, _ccf_iterations, _max_nspec_b7, 2, _si)
                               if _opt_needs_injection
                               else (n_orders, _nn, _ccf_iterations, _max_nspec_b7))
                _ccf_all_orders = np.full(_ccf_shape7, np.nan, float)
                for _h7 in range(n_orders):
                    if _h7 in _fully_masked_h_set:
                        continue
                    _ccf_all_orders[_h7] = per_order_results[_h7]["ccf_store"]
                for _h7 in range(n_orders):
                    for _nn7 in range(_nn):
                        _ccf_all_orders[
                            _h7, _nn7, :, int(n_spectra[_nn7]):
                        ] = np.nan

            # 7b, CO-ADD ALL SPECTRAL ORDERS
            if not _dn:
                ccf_nights = np.nansum(_ccf_all_orders, 0)
            else:
                ccf_nights = np.zeros(
                    (_nn, _ccf_iterations, _max_nspec_b7), float)
                _ord_dn = _mini_b7.get("order_selection_diffnights")
                for _b7 in range(_nn):
                    if _ord_dn is not None:
                        _oi7 = [
                            np.where(order_selection == _v7)[0][0]
                            for _v7 in _ord_dn[_b7]
                        ]
                    else:
                        _oi7 = list(range(n_orders))
                    ccf_nights[_b7] = np.nansum(
                        _ccf_all_orders[_oi7, _b7, :], 0)

            # 7c, CO-ADD NIGHTS → ccf_complete + ERF plot
            _plots_subdir_b7 = str(dirs["plots"])
            os.makedirs(_plots_subdir_b7, exist_ok=True)

            if not _dn:
                ccf_complete = np.nansum(ccf_nights, 0)
                CCF_matrix_ERF(
                    _mini_b7, _v_ccf, phase, ccf_complete,
                    with_signal, without_signal, v_planet,
                    show_plot=False, save_plot=True, CCF_Noise=False,
                )
            else:
                ccf_complete = ccf_nights
                for _bb7 in range(_nn):
                    CCF_matrix_ERF(
                        _mini_b7, _v_ccf, phase[_bb7],
                        ccf_complete[
                            _bb7, :, :int(n_spectra[_bb7])
                        ],
                        with_signal[_bb7], without_signal[_bb7],
                        v_planet[_bb7],
                        show_plot=False, save_plot=False,
                        CCF_Noise=False,
                    )

            # -------------------------------------------------------
            # 7d, v_rest grid and kp_range
            _max_ccf_v_b7 = cfg.cross_correlation.noise_velocity_max_kms
            _pix_lr_b7    = int(_max_ccf_v_b7 / _ccf_v_step)
            v_rest        = _ccf_v_step * np.arange(
                -_pix_lr_b7, _pix_lr_b7 + 1
            )
            _kp_range     = np.arange(
                -cfg.cross_correlation.kp_max_kms,
                cfg.cross_correlation.kp_max_kms + 1
            )
            _n_kp         = len(_kp_range)

            # 7d.2, Cheverall26 route ONLY: in-run duration (N_in) timing map.
            # Produced for EVERY Cheverall26 run (single real night and each
            # statistics night) from the one tested source chev26_duration_map
            # (identical algorithm to the validated duration test), so the run
            # emits everything it needs and nothing is computed by an external
            # helper.  Saves the per-night crosshair (N_in = number of
            # in-transit exposures) + map-max metrics for the statistics corner,
            # the map array, and the Fig.4-right figure (single/first night).
            # Gated to Cheverall26 so no other pipeline is affected.
            _chev_tim_pos = None
            _chev_tim_max = None
            if (cfg.pipeline.name == "Cheverall26" and not _dn
                    and _ccf_all_orders is not None):
                try:
                    from exoplore.pipelines.cheverall26 import (
                        chev26_duration_map, chev26_plot_duration_map)
                    _kp_t  = float(_mini_b7["K_p"])
                    _vs_t  = float(_mini_b7["V_sys"])
                    _excl_t = float(cfg.cross_correlation.snr_exclude_kms)
                    _ncross = (int(np.size(with_signal))
                               if np.ndim(with_signal) == 1 else 14)
                    _chev_tim_pos = np.full(_nn, np.nan)
                    _chev_tim_max = np.full(_nn, np.nan)
                    for _bt in range(_nn):
                        _ccfe = np.nansum(_ccf_all_orders[:, _bt, :, :], axis=0)
                        _pht = phase[_bt] if np.ndim(phase) > 1 else phase
                        _bet = berv[_bt] if np.ndim(berv) > 1 else berv
                        _snm, _vsa, _iv0 = chev26_duration_map(
                            _ccfe, _v_ccf, _pht, _bet, _kp_t, _vs_t,
                            exclude=_excl_t,
                            vmax_noise=cfg.cross_correlation.noise_velocity_max_kms)
                        _kx = int(np.clip(_ncross - 1, 0, _snm.shape[0] - 1))
                        _chev_tim_pos[_bt] = (_snm[_kx, _iv0]
                                              if np.isfinite(_snm[_kx, _iv0])
                                              else np.nan)
                        _chev_tim_max[_bt] = (np.nanmax(_snm)
                                              if np.any(np.isfinite(_snm))
                                              else np.nan)
                        if _nn == 1 or _bt == 0:
                            np.savez(
                                f"{_base_dir_b7}/ccf_timing_map_"
                                f"{_mini_b7['Simulation_name']}.npz",
                                snmap=_snm, vsys_axis=_vsa,
                                vsys_expected=_vs_t, kp=_kp_t,
                                n_in_cross=_ncross)
                            chev26_plot_duration_map(
                                _snm, _vsa, _vs_t,
                                f"{_plots_subdir_b7}/duration_test_fig4_"
                                f"{_mini_b7['Simulation_name']}.png",
                                kp=_kp_t, n_in_expected=_ncross)
                    save_compressed(
                        _base_dir_b7, sim_name,
                        {'stats_timing_pos': _chev_tim_pos,
                         'stats_timing_max': _chev_tim_max})
                    print(f"  [Cheverall26] in-run duration/timing map + "
                          f"metrics produced (N_in_cross={_ncross})")
                except Exception as _e:
                    print(f"  [Cheverall26] timing map skipped: {_e}")

            # 7e, get_shifted_ccf_matrix (Kp-grid shift)
            if not _dn:
                ccf_values_shift = get_shifted_ccf_matrix(
                    _mini_b7, with_signal, v_rest, _v_ccf, _kp_range,
                    phase, planet.systemic_velocity_kms, berv,
                    _pix_lr_b7, _ccf_v_step, ccf_complete,
                    sysrem_opt=_opt_needs_injection,
                )
            else:
                ccf_values_shift = []
                for _b7 in range(_nn):
                    ccf_values_shift.append(
                        get_shifted_ccf_matrix(
                            _mini_b7, with_signal[_b7], v_rest,
                            _v_ccf, _kp_range, phase[_b7],
                            planet.systemic_velocity_kms, berv[_b7],
                            _pix_lr_b7, _ccf_v_step,
                            ccf_complete[_b7, :, :int(n_spectra[_b7])],
                            sysrem_opt=_opt_needs_injection,
                        )
                    )

            # 7f, Significance evaluation
            if (cfg.cross_correlation.ccf_snr
                    and not cfg.cross_correlation.all_significance_metrics):
                # --- S/N only ---
                if not _dn:
                    ccf_tot = np.nansum(ccf_values_shift, 1)
                    (ccf_tot_sig, max_sig, max_kp_idx,
                     max_v_wind, _) = get_max_CCF_peak(
                        _mini_b7, ccf_tot, v_rest, _kp_range,
                        b=None, stats=None,
                        sysrem_opt=_opt_needs_injection, CCF_Noise=False,
                    )
                    # Persist the Kp-Vrest S/N map (same as the all-metrics
                    # branch does) so the map array is available post-hoc for
                    # zoomed figures and significance sampling, CCF S/N only,
                    # no Welch.
                    try:
                        _snr_mat_dir = str(dirs["matrices"])
                        os.makedirs(_snr_mat_dir, exist_ok=True)
                        np.savez(
                            f"{_snr_mat_dir}/ccf_tot_sn_map_"
                            f"{_mini_b7['Simulation_name']}.npz",
                            ccf_tot_sn=ccf_tot_sig, v_rest=v_rest,
                            kp_range=_kp_range,
                            sysrem_opt=bool(_opt_needs_injection),
                            order_indices=np.asarray(order_selection),
                            n_components_per_order=(
                                np.asarray(_sysrem_passes_per_order)
                                if _sysrem_passes_per_order is not None
                                else np.array([])),
                        )
                    except Exception as _e:
                        print(f"  NOTE: S/N map save skipped ({_e})")
                    plot_Kp_Vrest(
                        _mini_b7, _kp_range, ccf_tot_sig, v_rest,
                        show_plot=False, save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    plot_1D_CCF(
                        _mini_b7, v_rest, ccf_tot_sig, max_kp_idx,
                        max_sig, _n_kp, max_v_wind, [-100, 100],
                        show_plot=False, save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    if cfg.statistics.detectability_maps:
                        if cfg.atmosphere.limb_asymmetries:
                            _feh_dm = _ed.metallicity_wrt_solar
                            _co_dm  = _ed.carbon_to_oxygen_ratio
                        else:
                            _feh_dm = pm.metallicity_wrt_solar
                            _co_dm  = pm.carbon_to_oxygen_ratio
                        _feh_str = f"{_feh_dm:+.2f}".replace(".", "p")
                        _co_str  = f"{_co_dm:.2f}".replace(".", "p")
                        os.makedirs(_base_dir_b7, exist_ok=True)
                        _dmap_f = (f"{_base_dir_b7}/detectability_"
                                   f"FeH{_feh_str}_CO{_co_str}.txt")
                        _vr_idx = int(np.where(v_rest == max_v_wind)[0])
                        with open(_dmap_f, "w") as _dmfh:
                            _dmfh.write(
                                f"{_feh_dm:.4f} {_co_dm:.4f} "
                                f"{ccf_tot_sig[_vr_idx, max_kp_idx]:.4f}\n"
                            )
                else:  # Different_nights
                    # Per-night Kp-Vsys maps (each saved with _nightN suffix)
                    _night_1d_slices = []
                    for _nn7 in range(_nn):
                        _mini_b7["Simulation_name"] = (
                            f"{sim_name}_night{_nn7}")
                        ccf_tot = np.nansum(ccf_values_shift[_nn7], 1)
                        (ccf_tot_sig, max_sig, max_kp_idx,
                         max_v_wind, _) = get_max_CCF_peak(
                            _mini_b7, ccf_tot, v_rest, _kp_range,
                            b=None, stats=None,
                            sysrem_opt=_opt_needs_injection, CCF_Noise=False,
                        )
                        plot_Kp_Vrest(
                            _mini_b7, _kp_range, ccf_tot_sig, v_rest,
                            show_plot=False, save_plot=True,
                            sysrem_opt=_opt_needs_injection,
                        )
                        plot_1D_CCF(
                            _mini_b7, v_rest, ccf_tot_sig, max_kp_idx,
                            max_sig, _n_kp, max_v_wind, [-100, 100],
                            show_plot=False, save_plot=True,
                            sysrem_opt=_opt_needs_injection,
                        )
                        _night_1d_slices.append(
                            (ccf_tot_sig[:, int(max_kp_idx)], float(max_sig))
                        )
                    _mini_b7["Simulation_name"] = sim_name  # restore

                    # Combined Kp-Vsys map: co-add all nights
                    # nansum over spectra per night then sum across nights, 
                    # valid even when n_spectra differs per night.
                    _ccf_combined = sum(
                        np.nansum(ccf_values_shift[_n], 1)
                        for _n in range(_nn)
                    )
                    (ccf_tot_sig, max_sig, max_kp_idx,
                     max_v_wind, _) = get_max_CCF_peak(
                        _mini_b7, _ccf_combined, v_rest, _kp_range,
                        b=None, stats=None,
                        sysrem_opt=_opt_needs_injection, CCF_Noise=False,
                    )
                    plot_Kp_Vrest(
                        _mini_b7, _kp_range, ccf_tot_sig, v_rest,
                        show_plot=False, save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    plot_1D_CCF(
                        _mini_b7, v_rest, ccf_tot_sig, max_kp_idx,
                        max_sig, _n_kp, max_v_wind, [-100, 100],
                        show_plot=False, save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    # Overlay: all individual nights + combined on one figure
                    from exoplore.plotting.kpvsys import plot_multi_night_1D_CCF
                    plot_multi_night_1D_CCF(
                        _mini_b7, v_rest,
                        _night_1d_slices,
                        (ccf_tot_sig[:, int(max_kp_idx)], float(max_sig)),
                        xlims=[-100, 100],
                        show_plot=False, save_plot=True,
                    )

            elif (not cfg.cross_correlation.ccf_snr
                  and cfg.cross_correlation.welch_ttest
                  and not cfg.cross_correlation.all_significance_metrics):
                # --- Welch only ---
                ccf_tot = np.nansum(ccf_values_shift, 1)
                (ccf_tot_sig, _, _, v_rest_ttest, max_sig,
                 max_kp_idx, max_v_wind,
                 _, _, _, _, _, _) = Welch_ttest_map(
                    ccf_values_shift, v_rest, _kp_range,
                    _mini_b7, CCF_Noise=False,
                )
                # Persist the Welch Kp-Vrest map for completeness (distinct
                # name so it never collides with the CCF S/N map).
                try:
                    _snr_mat_dir = str(dirs["matrices"])
                    os.makedirs(_snr_mat_dir, exist_ok=True)
                    np.savez(
                        f"{_snr_mat_dir}/ccf_tot_welch_map_"
                        f"{_mini_b7['Simulation_name']}.npz",
                        ccf_tot_welch=ccf_tot_sig, v_rest=v_rest_ttest,
                        kp_range=_kp_range,
                        sysrem_opt=bool(_opt_needs_injection),
                        order_indices=np.asarray(order_selection),
                        n_components_per_order=(
                            np.asarray(_sysrem_passes_per_order)
                            if _sysrem_passes_per_order is not None
                            else np.array([])),
                    )
                except Exception as _e:
                    print(f"  NOTE: Welch map save skipped ({_e})")
                plot_Kp_Vrest(
                    _mini_b7, _kp_range, ccf_tot_sig, v_rest_ttest,
                    show_plot=False, save_plot=False, sysrem_opt=_opt_needs_injection,
                )
                plot_1D_CCF(
                    _mini_b7, v_rest_ttest, ccf_tot_sig, max_kp_idx,
                    max_sig, _n_kp, max_v_wind, [-100, 100],
                    show_plot=False, save_plot=False, sysrem_opt=_opt_needs_injection,
                )

            elif cfg.cross_correlation.all_significance_metrics:
                # --- Both S/N and Welch ---
                if not _dn:
                    ccf_tot = np.nansum(ccf_values_shift, 1)
                    # S/N branch
                    _mini_b7["CCF_SNR"]    = True
                    _mini_b7["Welch_ttest"] = False
                    (ccf_tot_sn, max_sn, max_kp_idx_sn,
                     max_v_wind_sn, _) = get_max_CCF_peak(
                        _mini_b7, ccf_tot, v_rest, _kp_range,
                        b=None, stats=None,
                        sysrem_opt=_opt_needs_injection, CCF_Noise=False,
                    )
                    # Persist the combined Kp-Vrest S/N map so a zoomed figure
                    # and the S/N at a chosen velocity can be produced post-hoc.
                    try:
                        _snr_mat_dir = str(dirs["matrices"])
                        os.makedirs(_snr_mat_dir, exist_ok=True)
                        np.savez(
                            f"{_snr_mat_dir}/ccf_tot_sn_map_"
                            f"{_mini_b7['Simulation_name']}.npz",
                            ccf_tot_sn=ccf_tot_sn, v_rest=v_rest,
                            kp_range=_kp_range,
                            sysrem_opt=bool(_opt_needs_injection),
                            order_indices=np.asarray(order_selection),
                            n_components_per_order=(
                                np.asarray(_sysrem_passes_per_order)
                                if _sysrem_passes_per_order is not None
                                else np.array([])),
                        )
                    except Exception as _e:
                        print(f"  NOTE: S/N map save skipped ({_e})")
                    plot_Kp_Vrest(
                        _mini_b7, _kp_range, ccf_tot_sn, v_rest,
                        show_plot=False, save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    plot_1D_CCF(
                        _mini_b7, v_rest, ccf_tot_sn, max_kp_idx_sn,
                        max_sn, _n_kp, max_v_wind_sn, [-100, 100],
                        show_plot=False, save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    # Welch branch, use a distinct Simulation_name so the
                    # Welch map doesn't overwrite the S/N map on disk.
                    _mini_b7["CCF_SNR"]    = False
                    _mini_b7["Welch_ttest"] = True
                    (ccf_tot_sig, _, _, v_rest_ttest, max_sig,
                     max_kp_idx, max_v_wind,
                     _, _, _, _, _, _) = Welch_ttest_map(
                        ccf_values_shift, v_rest, _kp_range,
                        _mini_b7, CCF_Noise=False,
                    )
                    _sim_name_b7 = _mini_b7["Simulation_name"]
                    _mini_b7["Simulation_name"] = f"{_sim_name_b7}_Welch"
                    # The Welch t-test map is collapsed to (n_v, n_kp) with a
                    # scalar max_kp and has no per-SYSREM-iteration axis, so it
                    # always uses the single-panel plot path (sysrem_opt=False),
                    # unlike the optimisation-aware CCF S/N map above.
                    plot_Kp_Vrest(
                        _mini_b7, _kp_range, ccf_tot_sig, v_rest_ttest,
                        show_plot=False, save_plot=True,
                        sysrem_opt=False,
                    )
                    plot_1D_CCF(
                        _mini_b7, v_rest_ttest, ccf_tot_sig, max_kp_idx,
                        max_sig, _n_kp, max_v_wind, [-100, 100],
                        show_plot=False, save_plot=True,
                        sysrem_opt=False,
                    )
                    _mini_b7["Simulation_name"] = _sim_name_b7


            # 7g, Injection-based per-order SYSREM iteration selection
            # (Maximum / Max_Diff only; DeltaSigma selects per-order in
            # prepare.py and does not need post-hoc CCF maximisation).
            if _opt_needs_injection:
                sysrem_it_opt = get_SYSREM_its_ordbyord(
                    _mini_b7, _ccf_all_orders, v_rest,
                    with_signal, phase,
                    berv, planet.systemic_velocity_kms,
                    _pix_lr_b7, _ccf_v_step, _v_ccf,
                ).astype(int)

                if (sysrem_it_opt.shape[0] == n_orders
                        and len(_ccf_all_orders.shape) == 6):
                    _ccf_opt = np.zeros(
                        _ccf_all_orders.shape[:4], float)
                    _crit7 = (
                        0 if cfg.pipeline.optimize_criterion == "Maximum"
                        else 1
                    )
                    for _b7 in range(_nn):
                        for _h7 in range(n_orders):
                            for _n7 in range(
                                _ccf_all_orders.shape[3]
                            ):
                                _ccf_opt[_h7, _b7, :, _n7] = (
                                    _ccf_all_orders[
                                        _h7, _b7, :, _n7,
                                        0,
                                        sysrem_it_opt[_h7, _b7, _crit7]
                                    ]
                                )
                    _ccf_opt = np.nansum(_ccf_opt, axis=0)  # orders
                    _ccf_opt = np.nansum(_ccf_opt, axis=0)  # nights
                    _cvs_opt = get_shifted_ccf_matrix(
                        _mini_b7, with_signal, v_rest, _v_ccf,
                        _kp_range, phase, planet.systemic_velocity_kms,
                        berv, _pix_lr_b7, _ccf_v_step,
                        _ccf_opt, sysrem_opt=False,
                    )
                    _ctot_opt = np.sum(_cvs_opt, 1)
                    (ccf_tot_sig, max_sig, max_kp_idx,
                     max_v_wind, _) = get_max_CCF_peak(
                        _mini_b7, _ctot_opt, v_rest, _kp_range,
                        b=None, stats=None,
                        sysrem_opt=False, CCF_Noise=False,
                    )
                    plot_Kp_Vrest(
                        _mini_b7, _kp_range, ccf_tot_sig, v_rest,
                        show_plot=False, save_plot=True,
                        sysrem_opt=False,
                    )
                    plot_1D_CCF(
                        _mini_b7, v_rest, ccf_tot_sig, max_kp_idx,
                        max_sig, _n_kp, max_v_wind, [-100, 100],
                        show_plot=False, save_plot=True,
                        sysrem_opt=False,
                    )
                    _sysrem_opt_file = (
                        f"{str(dirs['matrices'])}/"
                        f"sysrem_it_opt_{sim_name}"
                    )
                    np.savez_compressed(_sysrem_opt_file, a=sysrem_it_opt)

            # 7h, Statistical study  (n_nights > 1)
            if cfg.statistics.enabled and _nn != 1:

                def _save_stats_corner(stats_arr, sig_map, name_suffix=""):
                    """Render the noise-study corner with plot_stats (best-effort)."""
                    try:
                        _inp = dict(_mini_b7)
                        if name_suffix:
                            _inp["Simulation_name"] = f"{sim_name}{name_suffix}"
                        plot_stats(
                            stats_arr,
                            kp_lim_inf=-350, kp_lim_sup=350, kp_step=175,
                            vrest_lim_inf=-100, vrest_lim_sup=100, vrest_step=50,
                            sn_lim_inf=1, sn_lim_sup=13, sn_lim_step=2,
                            binwidth_sn=0.5, binwidth_kp=20, binwidth_v_rest=5,
                            significance_metric=sig_map, inp_dat=_inp, v_rest=v_rest,
                            auto_lims=True, mark_true_values=True,
                            show_dist_CC_values=False,
                            show_plot=False, save_plot=True, CCF_Noise=False,
                        )
                        print(f"  [stats] Corner plot saved: "
                              f"Corner_plot_{_inp['Simulation_name']}.png")
                    except Exception as _e:
                        print(f"  [stats] Could not save corner plot: {_e}")

                if not cfg.cross_correlation.all_significance_metrics:
                    (_, ccf_tot_sn_stat, _, _, stats,
                     stats_tvalue, stats_pvalue,
                     stats_planet_pos, stats_planet_area,
                     _, _, _, _,
                     _, _, shuffled_nights, v_rest_sigma) = (
                        statistical_study(
                            _mini_b7, _ccf_v_step,
                            _ccf_all_orders, _kp_range,
                            phase, _v_ccf, v_rest, with_signal,
                            _pix_lr_b7, sysrem_it_opt,
                            _ccf_iterations,
                            in_trail_pix=cfg.cross_correlation.in_trail_left_right,
                            auto_lims=True, input_stats=None,
                            verbose=True,
                            show_plot=False, save_plot=True,
                            CCF_Noise=False,
                        )
                    )
                    _night_min, _night_max = find_nights_with_extrema(
                        stats, cfg.observation.first_night_noiseless)
                    _sd_b7 = str(dirs["matrices"])
                    save_compressed(
                        _sd_b7, sim_name,
                        {'stats': stats,
                         'stats_planet_pos': stats_planet_pos,
                         'stats_planet_area': stats_planet_area,
                         'ccf_tot_sn_stat': ccf_tot_sn_stat},
                    )
                    if cfg.cross_correlation.ccf_snr:
                        save_compressed(
                            _sd_b7, sim_name, {'v_rest': v_rest})
                    elif cfg.cross_correlation.welch_ttest:
                        save_compressed(
                            _sd_b7, sim_name,
                            {'v_rest_ttest': v_rest_sigma,
                             'stats_tvalue': stats_tvalue,
                             'stats_pvalue': stats_pvalue},
                        )
                    save_compressed(
                        _sd_b7, sim_name,
                        {'kp_range': _kp_range},
                    )
                    _save_stats_corner(stats, ccf_tot_sn_stat)

                    # ---- Cheverall26 route only: add the timing (duration) map
                    # planet-pos + max per night, and a 4-metric corner plot.
                    # Reuses the per-night CCF cube (_ccf_all_orders) and the
                    # Kp-Vsys planet-pos/max already returned above.  Gated to
                    # Cheverall26 so no other pipeline/tutorial is affected.
                    if cfg.pipeline.name == "Cheverall26":
                        try:
                            from exoplore.pipelines.cheverall26 import (
                                chev26_timing_stats, chev26_fourmetric_corner)
                            # Reuse the per-night timing metrics already computed
                            # in-run by the 7d.2 block (single tested source);
                            # only recompute as a fallback if unavailable.
                            if _chev_tim_pos is not None:
                                _tim_pos, _tim_max = _chev_tim_pos, _chev_tim_max
                            else:
                                _excl = float(_mini_b7.get("CCF_SNR_exclude", 15.0))
                                _tim_pos, _tim_max = chev26_timing_stats(
                                    _ccf_all_orders, _v_ccf, phase, berv,
                                    _mini_b7["K_p"], _mini_b7["V_sys"], _nn,
                                    n_in_cross=14, exclude=_excl,
                                    vmax_noise=cfg.cross_correlation.noise_velocity_max_kms)
                                save_compressed(
                                    _sd_b7, sim_name,
                                    {'stats_timing_pos': _tim_pos,
                                     'stats_timing_max': _tim_max})
                            # Metric 1, Kp-Vsys S/N at the planet position:
                            # stats_planet_pos is (n_nights, 3) = [S/N, Kp,
                            # Vrest]; take the S/N column.
                            _kpv_pp = np.asarray(stats_planet_pos, float)
                            _kpv_pp = (_kpv_pp[:, 0] if _kpv_pp.ndim == 2
                                       else _kpv_pp.ravel())
                            # Metric 2, Kp-Vsys map MAX per night.  NOTE: the
                            # `stats` return is the local detection peak, NOT
                            # the global map max, so take the max of the saved
                            # full Kp-Vsys S/N map ccf_tot_sn_stat
                            # (n_vsys, n_kp, n_nights) over the map axes.
                            _kpv_map = np.asarray(ccf_tot_sn_stat, float)
                            if _kpv_map.ndim == 3:
                                _kpv_mx = np.nanmax(_kpv_map, axis=(0, 1))
                            else:
                                _kpv_mx = np.asarray(stats, float).ravel()
                            _metrics4 = {
                                "KpVsys_planetpos": _kpv_pp,
                                "KpVsys_max":       np.asarray(_kpv_mx, float).ravel(),
                                "timing_planetpos": np.asarray(_tim_pos, float).ravel(),
                                "timing_max":       np.asarray(_tim_max, float).ravel()}
                            # Plot goes in the run's plots/ folder (the npz
                            # data above stays in matrices/), matching the
                            # built-in Corner_plot convention.
                            _corner4 = (f"{str(dirs['plots'])}/Corner_4metric_"
                                        f"{_mini_b7['Simulation_name']}.png")
                            chev26_fourmetric_corner(
                                _metrics4, _corner4,
                                title=f"Cheverall26 significance metrics: {sim_name}")
                            print(f"  [stats] Cheverall26 timing stats + "
                                  f"4-metric corner saved: {_corner4}")
                        except Exception as _e:
                            print(f"  [stats] Cheverall26 timing/corner skipped: {_e}")

                else:
                    # All_significance_metrics: run both S/N and Welch
                    _mini_b7["CCF_SNR"]    = True
                    _mini_b7["Welch_ttest"] = False
                    _mini_b7["CCF_SNR"]    = True
                    _mini_b7["Welch_ttest"] = False
                    (_, ccf_tot_sn_stat, _, _,
                     stats_sn, _, _,
                     stats_planet_pos_sn, stats_planet_area_sn,
                     _, _, _, _,
                     _, _, shuffled_nights_sn, _) = (
                        statistical_study(
                            _mini_b7, _ccf_v_step,
                            _ccf_all_orders, _kp_range,
                            phase, _v_ccf, v_rest, with_signal,
                            _pix_lr_b7, sysrem_it_opt,
                            _ccf_iterations,
                            in_trail_pix=cfg.cross_correlation.in_trail_left_right,
                            auto_lims=True, input_stats=None,
                            verbose=True,
                            show_plot=False, save_plot=True,
                            CCF_Noise=False,
                        )
                    )
                    _mini_b7["CCF_SNR"]    = False
                    _mini_b7["Welch_ttest"] = True
                    (_, _, _, _,
                     stats_sig, _, _,
                     stats_planet_pos_sig, stats_planet_area_sig,
                     _, _, _, _,
                     _, _, shuffled_nights_sig, v_rest_sigma) = (
                        statistical_study(
                            _mini_b7, _ccf_v_step,
                            _ccf_all_orders, _kp_range,
                            phase, _v_ccf, v_rest, with_signal,
                            _pix_lr_b7, sysrem_it_opt,
                            _ccf_iterations,
                            in_trail_pix=cfg.cross_correlation.in_trail_left_right,
                            auto_lims=True, input_stats=None,
                            verbose=True,
                            show_plot=False, save_plot=True,
                            CCF_Noise=False,
                        )
                    )
                    _night_min_sn, _night_max_sn = find_nights_with_extrema(
                        stats_sn, cfg.observation.first_night_noiseless)
                    _night_max      = _night_max_sn
                    _night_min      = _night_min_sn
                    shuffled_nights = shuffled_nights_sn
                    _sd_b7 = str(dirs["matrices"])
                    save_compressed(
                        _sd_b7, sim_name,
                        {'stats': stats_sn, 'v_rest': v_rest,
                         'kp_range': _kp_range},
                    )
                    # Welch statistics share the single matrices/ directory; the
                    # distinct 'stats_welch' / 'v_rest_ttest' keys keep them from
                    # colliding with the S/N 'stats' / 'v_rest' files.
                    save_compressed(
                        _sd_b7, sim_name,
                        {'stats_welch': stats_sig,
                         'v_rest_ttest': v_rest_sigma},
                    )
                    _save_stats_corner(stats_sn, ccf_tot_sn_stat, "_SNR")
                    _save_stats_corner(stats_sig, ccf_tot_sn_stat, "_Welch")

            else:
                stats           = None
                shuffled_nights = None
                _night_min      = 0
                _night_max      = 0

            # -------------------------------------------------------
            # Note on skipped optional analyses:
            # * Study_velocity_ranges  (inp_dat["Study_velocity_ranges"])
            #   → requires iterating ccf_v_step grid; skipped pending
            #     dedicated analysis module
            # * Noise_statistical_study (inp_dat["Noise_statistical_study"])
            #   → requires gauss_noise per order which is not yet stored
            #     in per_order_results; skipped pending Block 5 update
            # -------------------------------------------------------

            _t_blocks["Block 7, CCF statistics, Kp-Vsys maps"] = (
                _time.time() - _t_run_start - sum(_t_blocks.values()))
            print(
                f"  Block 7 complete, CCF statistics, Kp-Vsys maps "
                f"({n_orders} orders)."
            )
            if cfg.timing:
                print(f"  [timing] Block 7: {_t_blocks['Block 7, CCF statistics, Kp-Vsys maps']:.1f} s")

        # ================================================================
        # Block 8, Save matrices and CCF products to disk
        #  and lines 5420 to 5436 [mat_back/propag_noise for retrieval])
        # ================================================================
        # Saves:
        #   8a  Per-order matrices: mat_res, propag_noise, mat_back,
        #       mat_noise, std_noise, mat_cc  (needed by Block 9 retrieval)
        #   8b  Global mask stores
        #   8c  U_sysrem and mat_star_forfile
        #   8d  CCF velocity grid and kp_range (convenience files)
        # ================================================================

        _base_dir_b8 = str(dirs["matrices"])
        os.makedirs(_base_dir_b8, exist_ok=True)
        _sfx8 = sim_name

        # 8a, Per-order matrices (each selected by cfg.output, see OutputConfig)
        _out8 = cfg.output
        for _h8, _ores8 in enumerate(per_order_results):
            _ord8 = order_selection[_h8]
            if _out8.save_mat_res:
                np.savez_compressed(
                    f"{_base_dir_b8}/mat_res_order_{_ord8}_{_sfx8}",
                    a=_ores8["mat_res"],
                )
            if _out8.save_propag_noise:
                np.savez_compressed(
                    f"{_base_dir_b8}/propag_noise_order_{_ord8}_{_sfx8}",
                    a=_ores8["propag_noise"],
                )
            if _out8.save_mat_back:
                np.savez_compressed(
                    f"{_base_dir_b8}/mat_back_order_{_ord8}_{_sfx8}",
                    a=_ores8["mat_back"],
                )
            if _out8.save_mat_noise:
                np.savez_compressed(
                    f"{_base_dir_b8}/mat_noise_order_{_ord8}_{_sfx8}",
                    a=_ores8["mat_noise"],
                )
            if _out8.save_std_noise:
                np.savez_compressed(
                    f"{_base_dir_b8}/std_noise_order_{_ord8}_{_sfx8}",
                    a=_ores8["std_noise"],
                )
            if _out8.save_mat_cc and _ores8.get("mat_cc") is not None:
                np.savez_compressed(
                    f"{_base_dir_b8}/mat_cc_order_{_ord8}_{_sfx8}",
                    a=_ores8["mat_cc"],
                )

        # 8b, Global mask stores
        if _mask_store is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/mask_{_sfx8}", a=_mask_store,
            )
        if _useful_spectral_points_store is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/useful_spectral_points_{_sfx8}",
                a=_useful_spectral_points_store,
            )
        if _mask_snr_store is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/mask_snr_{_sfx8}", a=_mask_snr_store,
            )
        if _useful_spectral_points_snr_store is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/useful_spectral_points_snr_{_sfx8}",
                a=_useful_spectral_points_snr_store,
            )
        if _mask_inter_store is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/mask_inter_{_sfx8}",
                a=_mask_inter_store,
            )
        if _useful_spectral_points_inter_store is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/useful_spectral_points_inter_{_sfx8}",
                a=_useful_spectral_points_inter_store,
            )

        # 8c, SYSREM basis and stellar matrix
        if _U_sysrem is not None and _out8.save_U_sysrem:
            np.savez_compressed(
                f"{_base_dir_b8}/U_sysrem_{_sfx8}", a=_U_sysrem,
            )
        if _mat_star_forfile is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/mat_star_forfile_{_sfx8}",
                a=_mat_star_forfile,
            )

        # 8d, CCF velocity grid and Kp range (convenience)
        if _ccf_flag and _v_ccf is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/v_ccf_{_sfx8}", a=_v_ccf,
            )
        if _kp_range is not None:
            np.savez_compressed(
                f"{_base_dir_b8}/kp_range_{_sfx8}", a=_kp_range,
            )

        # ----------------------------------------------------------------
        # Store Blocks 7 to 8 state
        # ----------------------------------------------------------------
        self._state.update(dict(
            # Block 7, Kp-Vsys maps
            ccf_all_orders=_ccf_all_orders,
            ccf_complete=ccf_complete,
            ccf_values_shift=ccf_values_shift,
            v_rest=v_rest,
            kp_range=_kp_range,
            n_kp=_n_kp,
            pixels_left_right=_pix_lr_b7,
            sysrem_it_opt=sysrem_it_opt,
            stats=stats,
            shuffled_nights=shuffled_nights,
            night_min=_night_min,
            night_max=_night_max,
            ccf_tot_sig=ccf_tot_sig,
            max_sig=max_sig,
            max_kp_idx=max_kp_idx,
            max_v_wind=max_v_wind,
            # Block 8, output directory
            output_matrix_dir=_base_dir_b8,
        ))

        print(
            f"\n  Blocks 7-8 complete, Kp-Vsys maps, matrices saved "
            f"to {_base_dir_b8}"
        )
        _t_blocks["Block 8, save matrices & CCF products"] = (
            _time.time() - _t_run_start - sum(_t_blocks.values()))
        if cfg.timing:
            print(f"  [timing] Block 8: {_t_blocks['Block 8, save matrices & CCF products']:.1f} s")
        print("  (Next: Block 9, Retrieval)\n")
        _t_b9_start = _time.time()

        # ======================================================================
        # BLOCK 9, RETRIEVAL
        # ======================================================================

        # Coupling validation: logL_choice and Ret_dim must be compatible.
        # The beta noise-scaling parameter is only part of '1D_Gibson22', using Gibson22
        # likelihood with any other dimensionality is undefined, and using
        # Blain24/BL19 with '1D_Gibson22' samples beta but never uses it.
        _VALID_LL_FOR_DIM = {
            "1D":          ("BL19", "Blain24"),
            "1D_Gibson22":      ("Gibson22",),
            "1D_CtoO_met": ("BL19", "Blain24"),
            "2D":          ("BL19", "Blain24"),
            # Amplitude-scaling retrieval: free Kp, V_rest, α on a fixed
            # nominal model (no pRT / abundance / T sampled).
            "1D_alpha":    ("Blain24", "BL19"),
            # Same fixed-model α detection, but with the Gibson22 β noise-rescaling
            # likelihood (free Kp, V_rest, α, β): β fits a single rescaling of the
            # propagated per-channel sigma from the data.
            "1D_alpha_Gibson22": ("Gibson22",),
        }
        _ret_dim = cfg.retrieval.dimensionality
        _logL    = cfg.retrieval.log_likelihood
        if _ret_dim in _VALID_LL_FOR_DIM:
            if _logL not in _VALID_LL_FOR_DIM[_ret_dim]:
                raise ValueError(
                    f"Incompatible retrieval configuration: "
                    f"dimensionality='{_ret_dim}' requires "
                    f"log_likelihood in {_VALID_LL_FOR_DIM[_ret_dim]}, "
                    f"got '{_logL}'. "
                    f"The beta noise-scaling parameter is only sampled in "
                    f"'1D_Gibson22', combining it with Blain24/BL19 produces "
                    f"meaningless posteriors. Use '1D_Gibson22' only with Gibson22."
                )

        # -- Parameter labels per Ret_dim --
        _RET_PARAM_LABELS_B9 = {
            "1D":   ["log(X$_{H_2O}$)", "$K_P$", "T$_{equ}$", "V$_{rest}$"],
            "1D_Gibson22": ["log(X$_{H_2O}$)", "$K_P$", "T$_{equ}$", "V$_{rest}$",
                       r"$\beta$"],
            "1D_CtoO_met": ["C/O", "[(C+O)/H]"],
            "2D": ["log(X$_{H_2O,\\,LL}$)", "log(X$_{H_2O,\\,TL}$)",
                   "$K_P$", "T$_{equ,\\,LL}$", "T$_{equ,\\,TL}$"],
            "1D_alpha": ["$K_P$", "V$_{rest}$", r"$\alpha$"],
            "1D_alpha_Gibson22": ["$K_P$", "V$_{rest}$", r"$\alpha$", r"$\beta$"],
        }
        # Prior bounds: use config values if provided, else built-in defaults.
        _cfg_pb = cfg.retrieval.prior_bounds or {}
        _PRIOR_DEFAULTS_B9 = {
            "1D":          [(-8., 0.), (85., 200.), (400., 1500.), (-25., 25.)],
            "1D_Gibson22":      [(-8., 0.), (85., 200.), (400., 1500.), (-25., 25.),
                            (0.01, 100.)],
            "1D_CtoO_met": [(0., 2.), (-2.5, 2.5)],
            "2D":          [(-8., 0.), (-8., 0.), (85., 200.), (400., 1500.),
                            (400., 1500.), (-20., 20.)],
            # K_p, V_rest, alpha  (alpha prior includes 0 so the posterior can
            # approach the null; config overrides for the target).
            "1D_alpha":    [(40., 100.), (-25., 25.), (0., 5.)],
            # K_p, V_rest, alpha, beta  (beta = Gibson22 noise-rescaling of the
            # propagated σ; config overrides alpha for model/null).
            "1D_alpha_Gibson22": [(40., 100.), (-25., 25.), (0., 5.), (0.01, 100.)],
        }
        _PRIOR_BOUNDS_B9 = {
            dim: [tuple(b) for b in _cfg_pb.get(dim, _PRIOR_DEFAULTS_B9.get(dim, []))]
                 or _PRIOR_DEFAULTS_B9.get(dim, [])
            for dim in _PRIOR_DEFAULTS_B9
        }
        retrieval_name = "retrieval"
        parameters = _RET_PARAM_LABELS_B9.get(_ret_dim, ["param"])
        n_params = len(parameters)

        if cfg.retrieval.enabled:
            import copy as _copy_b9
            import json as _json_b9
            # Augment _mini_b7 with all keys needed by call_pRT,
            # spec_to_mat_fraction, and preparing_pipeline inside the loglike.
            # The monolithic inp_dat is split here into mini-dicts by
            # purpose.  Merging _mini_prt and _mini_limbs here gives the
            # retrieval loglike the full parameter set without key errors.
            _mini_b7.update(_mini_prt)
            _mini_b7.update(_mini_limbs)
            _mini_b7.update({
                "Scale_inj":           cfg.pipeline.inject_scale_factor,
                "Inject_Scale_Factor": cfg.pipeline.inject_scale_factor,
                "Noise_statistical_study": False,
            })
            # Override species_ret to match the abundances array built inside
            # each loglike function.  This is set explicitly per ret_dim;
            # our code defaulted to all planet species → len mismatch in pRT.
            _ret_dim_b9 = cfg.retrieval.dimensionality
            if _ret_dim_b9 in ("1D", "1D_Gibson22"):
                _mini_b7["species_ret"] = ["H2", "He", "H2O"]
            elif _ret_dim_b9 in ("1D_alpha", "1D_alpha_Gibson22"):
                # Amplitude scaling: nominal model uses the config's species
                # opacity (the same line list as the CCF template).
                _mini_b7["species_ret"] = [
                    "H2", "He", cfg.atmosphere.ccf_template.species[-1]]
            elif _ret_dim_b9 == "1D_CtoO_met":
                _mini_b7["species_ret"] = ["H2", "He"]
            elif _ret_dim_b9 in ("1D_extended", "1D_extended_fast"):
                _mini_b7["species_ret"] = [
                    "H2", "He", "H2O", "CH4", "NH3", "CO", "CO2", "HCN"]
            elif _ret_dim_b9 == "2D":
                _mini_b7["species_ret"] = ["H2", "He", "H2O"]

            try:
                import pymultinest as _pymn
            except ImportError:
                _pymn = None
            try:
                import emcee as _emcee
            except ImportError:
                _emcee = None
            try:
                import corner as _corner
            except ImportError:
                _corner = None
            try:
                from petitRADTRANS.radtrans import Radtrans as _Radtrans
            except ImportError:
                _Radtrans = None

            _mini_b7["prepare_template"] = True

            # ------------------------------------------------------------------
            # 9a, Build retrieval matrices from per_order_results (in-memory)
            #      This version reads from RAM rather than disk.
            # ------------------------------------------------------------------
            _n_orders_b9 = n_orders
            _n_nights_b9 = _nn
            _n_px_b9     = n_pixels

            # ------------------------------------------------------------------
            # 9a pre-filter: remove fully-masked orders before stacking.
            #
            # Fully-masked orders (ccf_store never populated) must be excluded
            # from ALL retrieval structures before anything else in Block 9 runs.
            # The loglike uses hh as a SHARED index across:
            #   atmosphere_ret_list[hh]
            #   inp_dat["order_selection"][hh]
            #   mask_aux[night_index, hh, :]
            #   model_mat[hh]
            # All of these must be in sync.  The only safe place to achieve
            # that is here, before 9c builds _mat_res_ret and before 9d builds
            # wave_ret and loads the mask arrays.
            # ------------------------------------------------------------------
            if _fully_masked_h_set:
                _good_h_b9 = [
                    h for h in range(_n_orders_b9)
                    if h not in _fully_masked_h_set
                ]
                _n_orders_b9 = len(_good_h_b9)
                _mini_b7["order_selection"] = order_selection[
                    np.asarray(_good_h_b9, dtype=int)]
            else:
                _good_h_b9 = list(range(_n_orders_b9))

            # Keep _mini_b7["n_orders"] in sync with any dropped orders
            _mini_b7["n_orders"] = _n_orders_b9

            if not _dn:
                _n_sp_b9 = n_spectra
                # mat_res / propag / std_noise already have a night axis
                # from Block 5 allocation shape (_nn, n_spectra, n_pixels),
                # so stacking gives (n_orders, _nn, n_spectra, n_pixels). ✓
                # mat_star comes from get_stellar_matrix → (n_spectra, n_pixels),
                # so we must insert the night axis after stacking.
                mat_res_b9    = np.stack(
                    [per_order_results[h]["mat_res"]      for h in _good_h_b9])
                propag_b9     = np.stack(
                    [per_order_results[h]["propag_noise"] for h in _good_h_b9])
                std_noise_b9  = np.stack(
                    [per_order_results[h]["std_noise"]    for h in _good_h_b9])
                _ms_stack = np.stack(
                    [per_order_results[h]["mat_star"]     for h in _good_h_b9])
                if _ms_stack.ndim == 3:
                    # (n_orders, n_spectra, n_pixels) → add night dim
                    mat_star_b9 = _ms_stack[:, np.newaxis, :, :]
                else:
                    mat_star_b9 = _ms_stack
                _U_b9 = None
                if cfg.pipeline.name in ["ASL19", "Gibson22"]:
                    if per_order_results[0].get("U_sysrem") is not None:
                        _U_b9 = np.stack(
                            [per_order_results[h]["U_sysrem"]
                             for h in _good_h_b9])
            else:
                _max_sp_b9 = int(np.max(_n_spectra_store_b5))
                mat_res_b9   = np.full(
                    (_n_orders_b9, _n_nights_b9, _max_sp_b9, _n_px_b9), np.nan)
                propag_b9    = np.full_like(mat_res_b9, np.nan)
                std_noise_b9 = np.full_like(mat_res_b9, np.nan)
                mat_star_b9  = np.full_like(mat_res_b9, np.nan)
                for _hb9_idx, _hb9 in enumerate(_good_h_b9):
                    _mr_h = per_order_results[_hb9]["mat_res"]
                    _pn_h = per_order_results[_hb9]["propag_noise"]
                    _sn_h = per_order_results[_hb9]["std_noise"]
                    _ms_h = per_order_results[_hb9]["mat_star"]
                    mat_res_b9[_hb9_idx,   :, :_mr_h.shape[1], :] = _mr_h
                    propag_b9[_hb9_idx,    :, :_pn_h.shape[1], :] = _pn_h
                    std_noise_b9[_hb9_idx, :, :_sn_h.shape[1], :] = _sn_h
                    mat_star_b9[_hb9_idx,  :, :_ms_h.shape[1], :] = _ms_h
                _U_b9 = None
                if cfg.pipeline.name in ["ASL19", "Gibson22"]:
                    if "U_sysrem" in per_order_results[0]:
                        _U_b9 = np.full(
                            (_n_orders_b9, _n_nights_b9, _max_sp_b9,
                             _si), np.nan)
                        for _hb9_idx, _hb9 in enumerate(_good_h_b9):
                            _u_h = per_order_results[_hb9]["U_sysrem"]
                            _U_b9[_hb9_idx, :, :_u_h.shape[1], :] = _u_h
                # NaN-fill trailing spectra per night
                for _nn9 in range(_n_nights_b9):
                    _nsp9 = int(_n_spectra_store_b5[_nn9])
                    mat_res_b9[:,   _nn9, _nsp9:, :] = np.nan
                    propag_b9[:,    _nn9, _nsp9:, :] = np.nan
                    std_noise_b9[:, _nn9, _nsp9:, :] = np.nan
                    mat_star_b9[:,  _nn9, _nsp9:, :] = np.nan
                    if _U_b9 is not None:
                        _U_b9[:, _nn9, _nsp9:, :] = np.nan

            # ------------------------------------------------------------------
            # 9b, base directory
            # ------------------------------------------------------------------
            # Block 8 saved masks and matrices directly into _base_dir_b8.
            # Block 9 must load from the same directory.
            _base_dir_b9 = _base_dir_b8
            _sfx9 = _mini_b7["Simulation_name"]

            # ------------------------------------------------------------------
            # 9c, Night selection (lines 5757-5828)
            # ------------------------------------------------------------------
            _night_min_b9 = int(_night_min)
            _night_max_b9 = int(_night_max)

            if _mini_b7["Retrieval_choice"] == 1:
                print("Performing retrieval on the first night")
                _mr_n1 = mat_res_b9[:, 0, :, :].reshape(
                    1, *mat_res_b9[:, 0, :, :].shape)
                _mat_res_ret  = np.transpose(_mr_n1, (1, 0, 2, 3))
                _pn_n1 = propag_b9[:, 0, :, :].reshape(1, *propag_b9[:, 0, :, :].shape)
                _propag_ret   = np.transpose(_pn_n1, (1, 0, 2, 3))
                _ms_n1 = mat_star_b9[:, 0, :, :].reshape(
                    1, *mat_star_b9[:, 0, :, :].shape)
                _mat_star_ret = np.transpose(_ms_n1, (1, 0, 2, 3))
                _sn_n1 = std_noise_b9[:, 0, :, :].reshape(
                    1, *std_noise_b9[:, 0, :, :].shape)
                _std_noise_ret = np.transpose(_sn_n1, (1, 0, 2, 3))
                retrieved_nights = 1
            elif _mini_b7["Retrieval_choice"] == 2:
                print("Performing retrieval on night_max, night_min, and night_avg")
                _mean_night_b9 = int(np.argwhere(
                    stats[:, 0] == find_nearest(
                        stats[:, 0], np.mean(stats[:, 0])))[0][0])
                _sel9 = [_night_max_b9, _night_min_b9, _mean_night_b9]
                _mat_res_ret   = mat_res_b9[:,  _sel9, :, :]
                _propag_ret    = propag_b9[:,   _sel9, :, :]
                _mat_star_ret  = mat_star_b9[:, _sel9, :, :]
                _std_noise_ret = std_noise_b9[:,_sel9, :, :]
                retrieved_nights = 3
            elif _mini_b7["Retrieval_choice"] == 3:
                print("Performing retrieval on night_max and night_min")
                _sel9 = [_night_max_b9, _night_min_b9]
                _mat_res_ret   = mat_res_b9[:,  _sel9, :, :]
                _propag_ret    = propag_b9[:,   _sel9, :, :]
                _mat_star_ret  = mat_star_b9[:, _sel9, :, :]
                _std_noise_ret = std_noise_b9[:,_sel9, :, :]
                retrieved_nights = 2
            elif _mini_b7["Retrieval_choice"] in [4, 5, 6]:
                if _mini_b7["Retrieval_choice"] == 4:
                    print("Combining all retrievals")
                _mat_res_ret   = np.copy(mat_res_b9)
                _propag_ret    = np.copy(propag_b9)
                _mat_star_ret  = np.copy(mat_star_b9)
                _std_noise_ret = np.copy(std_noise_b9)
                retrieved_nights = _n_nights_b9

            # ------------------------------------------------------------------
            # 9d, Wave and mask loading (lines 5838-5858)
            # ------------------------------------------------------------------
            if _mini_b7["Different_nights"]:
                wave_ret = [
                    wave_star[
                        _mini_b7["order_selection_diffnights"][_b9], :
                    ].reshape(_n_px_b9 * len(_mini_b7["order_selection_diffnights"][_b9]))
                    for _b9 in range(_n_nights_b9)
                ]
            else:
                wave_ret = wave_star[_mini_b7["order_selection"], :].reshape(
                    _n_px_b9 * len(_mini_b7["order_selection"]))

            # Load masks from files saved in Block 8
            mask_ret_aux = np.load(
                f"{_base_dir_b9}/mask_{_sfx9}.npz")["a"]
            useful_spectral_points_aux = np.load(
                f"{_base_dir_b9}/useful_spectral_points_{_sfx9}.npz")["a"]
            mask_snr_ret_aux = np.load(
                f"{_base_dir_b9}/mask_snr_{_sfx9}.npz")["a"]
            useful_spectral_points_snr_aux = np.load(
                f"{_base_dir_b9}/useful_spectral_points_snr_{_sfx9}.npz")["a"]
            mask_inter_ret_aux = np.load(
                f"{_base_dir_b9}/mask_inter_{_sfx9}.npz")["a"]
            useful_spectral_points_inter_aux = np.load(
                f"{_base_dir_b9}/useful_spectral_points_inter_{_sfx9}.npz")["a"]

            # ------------------------------------------------------------------
            # 9d.5, Filter mask arrays for fully-masked orders
            # mat_res_b9 / propag_b9 / std_noise_b9 / mat_star_b9 were already
            # built from _good_h_b9 only in 9a.  _mini_b7["order_selection"] was
            # updated there too, so wave_ret (built in 9d) is already correct.
            # The only thing left is to filter the on-disk mask arrays, which
            # are shaped (n_nights, n_orders_ORIGINAL, n_pixels) and therefore
            # still carry the bad-order columns.
            # ------------------------------------------------------------------
            if _fully_masked_h_set:
                # Filter mask arrays along the order (axis-1) dimension.
                mask_ret_aux = mask_ret_aux[:, _good_h_b9, :]
                useful_spectral_points_aux = \
                    useful_spectral_points_aux[:, _good_h_b9, :]
                mask_snr_ret_aux = mask_snr_ret_aux[:, _good_h_b9, :]
                useful_spectral_points_snr_aux = \
                    useful_spectral_points_snr_aux[:, _good_h_b9, :]
                mask_inter_ret_aux = mask_inter_ret_aux[:, _good_h_b9, :]
                useful_spectral_points_inter_aux = \
                    useful_spectral_points_inter_aux[:, _good_h_b9, :]

            # ------------------------------------------------------------------
            # 9e, Reshape matrices (lines 5863-5969)
            # ------------------------------------------------------------------
            _n_ord_sel_b9 = _n_orders_b9  # updated above if orders were dropped
            if not _mini_b7["Different_nights"]:
                mask_ret = mask_ret_aux.reshape(
                    _n_nights_b9, _n_px_b9 * _n_ord_sel_b9)
                useful_spectral_points_ret = useful_spectral_points_aux.reshape(
                    _n_nights_b9, _n_px_b9 * _n_ord_sel_b9)
                mask_snr_ret = mask_snr_ret_aux.reshape(
                    _n_nights_b9, _n_px_b9 * _n_ord_sel_b9)
                useful_spectral_points_snr_ret = \
                    useful_spectral_points_snr_aux.reshape(
                        _n_nights_b9, _n_px_b9 * _n_ord_sel_b9)
                mask_inter_ret = mask_inter_ret_aux.reshape(
                    _n_nights_b9, _n_px_b9 * _n_ord_sel_b9)
                useful_spectral_points_inter_ret = \
                    useful_spectral_points_inter_aux.reshape(
                        _n_nights_b9, _n_px_b9 * _n_ord_sel_b9)

                # (n_nights, retrieved_nights, n_spectra, n_orders*n_pixels)
                _mat_res_ret  = np.transpose(_mat_res_ret,  (1, 2, 0, 3))
                _mat_res_ret  = _mat_res_ret.reshape(
                    _mat_res_ret.shape[0],  _mat_res_ret.shape[1],
                    _mat_res_ret.shape[2] * _mat_res_ret.shape[3])
                _propag_ret   = np.transpose(_propag_ret,   (1, 2, 0, 3))
                _propag_ret   = _propag_ret.reshape(
                    _propag_ret.shape[0],   _propag_ret.shape[1],
                    _propag_ret.shape[2] *  _propag_ret.shape[3])
                _mat_star_ret = np.transpose(_mat_star_ret, (1, 2, 0, 3))
                _mat_star_ret = _mat_star_ret.reshape(
                    _mat_star_ret.shape[0], _mat_star_ret.shape[1],
                    _mat_star_ret.shape[2] * _mat_star_ret.shape[3])
                _std_noise_ret = np.transpose(_std_noise_ret, (1, 2, 0, 3))
                _std_noise_ret = _std_noise_ret.reshape(
                    _std_noise_ret.shape[0], _std_noise_ret.shape[1],
                    _std_noise_ret.shape[2] * _std_noise_ret.shape[3])
            else:
                mask_ret                       = []
                useful_spectral_points_ret     = []
                mask_snr_ret                   = []
                useful_spectral_points_snr_ret = []
                mask_inter_ret                 = []
                useful_spectral_points_inter_ret = []

                _mat_res_ret_aux   = np.copy(_mat_res_ret)
                _propag_ret_aux    = np.copy(_propag_ret)
                _mat_star_ret_aux  = np.copy(_mat_star_ret)
                _std_noise_ret_aux = np.copy(_std_noise_ret)
                del _mat_res_ret, _propag_ret, _mat_star_ret, _std_noise_ret

                _mat_res_ret   = []
                _propag_ret    = []
                _std_noise_ret = []
                _mat_star_ret  = []

                if _mini_b7["Retrieval_choice"] == 1:
                    raise ValueError(
                        "Retrieval_choice==1 with Different_nights: "
                        "dimension mismatch.")

                for _b9 in range(_n_nights_b9):
                    _sel_b9  = _mini_b7["order_selection_diffnights"][_b9]
                    _idx_b9  = [
                        np.where(_mini_b7["order_selection"] == v)[0][0]
                        for v in _sel_b9
                    ]
                    mask_ret.append(
                        mask_ret_aux[_b9, _idx_b9].ravel())
                    useful_spectral_points_ret.append(
                        useful_spectral_points_aux[_b9, _idx_b9].ravel())
                    mask_snr_ret.append(
                        mask_snr_ret_aux[_b9, _idx_b9].ravel())
                    useful_spectral_points_snr_ret.append(
                        useful_spectral_points_snr_aux[_b9, _idx_b9].ravel())
                    mask_inter_ret.append(
                        mask_inter_ret_aux[_b9, _idx_b9].ravel())
                    useful_spectral_points_inter_ret.append(
                        useful_spectral_points_inter_aux[_b9, _idx_b9].ravel())
                    _blk = _mat_res_ret_aux[_idx_b9, _b9].transpose(1, 0, 2)
                    _mat_res_ret.append(_blk.reshape(_blk.shape[0], -1))
                    _blk = _propag_ret_aux[_idx_b9, _b9].transpose(1, 0, 2)
                    _propag_ret.append(_blk.reshape(_blk.shape[0], -1))
                    _blk = _std_noise_ret_aux[_idx_b9, _b9].transpose(1, 0, 2)
                    _std_noise_ret.append(_blk.reshape(_blk.shape[0], -1))
                    _blk = _mat_star_ret_aux[_idx_b9, _b9].transpose(1, 0, 2)
                    _mat_star_ret.append(_blk.reshape(_blk.shape[0], -1))

            # ------------------------------------------------------------------
            # 9f, Different_nights deep copies (lines 6008-6021)
            # ------------------------------------------------------------------
            if _mini_b7["Different_nights"]:
                T_0 = np.asarray(T_0)
                _syn_jd_store_b9 = np.full(
                    (_n_nights_b9, int(max(_n_spectra_store_b5))), np.nan)
                for _n_idx9, _n_aux9 in enumerate(n_spectra):
                    _syn_jd_store_b9[_n_idx9, :_n_aux9] = \
                        np.asarray(syn_jd[_n_idx9])
                    _syn_jd_store_b9[_n_idx9, _n_aux9:] = np.nan
                _std_noise_ret_store = _copy_b9.deepcopy(_std_noise_ret)
                _mat_res_ret_store   = _copy_b9.deepcopy(_mat_res_ret)
                _propag_ret_store    = _copy_b9.deepcopy(_propag_ret)
                _mat_star_ret_store  = _copy_b9.deepcopy(_mat_star_ret)
                _T_0_store_b9        = np.copy(T_0)
                del _mat_res_ret, _propag_ret, _mat_star_ret, \
                    _mat_res_ret_aux, _propag_ret_aux

            # ------------------------------------------------------------------
            # 9g, SYSREM filtering projector (lines 6024-6035)
            # ------------------------------------------------------------------
            P = None
            if (_mini_b7["preparing_pipeline"] in ["ASL19", "Gibson22"]
                    and _U_b9 is not None):
                if not _mini_b7["Different_nights"]:
                    P = SYSREM_filtering_projector(
                        _mini_b7, n_spectra, propag_b9, _U_b9)
                else:
                    P = SYSREM_filtering_projector(
                        _mini_b7, _n_spectra_store_b5, propag_b9, _U_b9)

            # ------------------------------------------------------------------
            # 9h, Sampler definitions (lines 6052-6087)
            # ------------------------------------------------------------------
            if _mini_b7["Sampler_choice"] == "Nested_sampling":
                def prior(cube, ndim, nparams):
                    # Use user-supplied prior_bounds from config if available,
                    # falling back to the hardcoded defaults otherwise.
                    _cfg_bounds = _mini_b7.get("PRIOR_BOUNDS", {})
                    _bds = (_cfg_bounds.get(_mini_b7["Ret_dim"])
                            or _PRIOR_BOUNDS_B9[_mini_b7["Ret_dim"]])
                    for _i9, (_lo9, _hi9) in enumerate(_bds):
                        cube[_i9] = _lo9 + (_hi9 - _lo9) * cube[_i9]
                    return cube
            # (log_prior/log_prob/emcee setup deferred, needs loglike defined first)

            # pressure grid for pRT retrievals
            p_ret = np.logspace(
                cfg.retrieval.pressure_log_min,
                cfg.retrieval.pressure_log_max,
                cfg.retrieval.pressure_n_levels,
            )

            # mutable context shared between loglike closure and outer loop
            _ctx = {
                "sysrem_pass":       None,
                "night_index":       0,
                "idx_interval":      None,
                "interval_tracker":  None,
                "atmosphere_ret_list":        None,
                "atmosphere_ret_list_morning": None,
                "atmosphere_ret_list_evening": None,
            }

            # ------------------------------------------------------------------
            # 9i, 1D retrieval branch (lines 6091-8453)
            # ------------------------------------------------------------------
            if _mini_b7["Ret_dim"] in [
                "1D", "1D_CtoO_met", "1D_Gibson22", "1D_alpha", "1D_alpha_Gibson22"
            ]:
                _mini_b7["Limb_asymmetries"] = False

                # ==============================================================
                # Single-night loglike  (Retrieval_choice != 4)
                # Master lines 6098-6667
                # ==============================================================
                def _loglike_1d_single(cube, ndim, nparams):
                    _sysrem_pass      = _ctx["sysrem_pass"]
                    _night_index      = _ctx["night_index"]
                    _idx_interval     = _ctx["idx_interval"]
                    _interval_tracker = _ctx["interval_tracker"]
                    _atm_list         = _ctx["atmosphere_ret_list"]

                    # ---- param extraction ----
                    if _mini_b7["Ret_dim"] == "1D":
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]
                    elif _mini_b7["Ret_dim"] == "1D_CtoO_met":
                        c_to_o = cube[0]; met = cube[1]
                    elif _mini_b7["Ret_dim"] == "1D_Gibson22":
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]; beta = cube[4]
                    elif _mini_b7["Ret_dim"] == "1D_alpha":
                        # Amplitude scaling: free K_p, V_rest, α on a fixed
                        # nominal model (no pRT / abundance / T sampled).
                        K_p = cube[0]; v_wind = cube[1]; alpha = cube[2]
                    elif _mini_b7["Ret_dim"] == "1D_alpha_Gibson22":
                        # Same fixed-model α detection + Gibson22 β noise-rescaling.
                        K_p = cube[0]; v_wind = cube[1]
                        alpha = cube[2]; beta = cube[3]

                    log_likelihood = 0.

                    # ---- abundances ----
                    if _mini_b7["Ret_dim"] in ["1D", "1D_Gibson22"]:
                        abundances = np.asarray(
                            [_mini_b7["vmr"][0], _mini_b7["vmr"][1], 10.**log10_X1])
                    else:
                        abundances = np.asarray(
                            [_mini_b7["vmr"][0], _mini_b7["vmr"][1]])

                    # ---- model matrix ----
                    if not _mini_b7["Different_nights"]:
                        _n_sp_ll  = n_spectra
                        _ph_ll    = phase
                        _berv_ll  = berv
                        _with_ll  = with_signal
                        _wout_ll  = without_signal
                        _frac_ll  = fraction
                        _sjd_ll   = syn_jd
                        _T0_ll    = T_0
                        model_mat = np.zeros(
                            (_n_ord_sel_b9, _n_sp_ll, n_pixels))
                        model_mat_prepared = np.zeros_like(model_mat)

                        _emulator_ll = _ctx.get("prt_emulator")
                        for hh in range(_n_orders_b9):
                            atm_r = _atm_list[hh]
                            if _mini_b7["Ret_dim"] in ("1D_alpha", "1D_alpha_Gibson22"):
                                # FIXED nominal model (computed once, pre-sampler);
                                # α scales its depth via Scale_inj below. No pRT.
                                wave_pRT_r = _ctx["nominal_wave"][hh]
                                syn_spec_r = _ctx["nominal_spec"][hh]
                            elif _emulator_ll is not None:
                                wave_pRT_r, syn_spec_r = _emulator_ll.predict(
                                    T_eq=T_equ,
                                    log10_x_h2o=log10_X1,
                                )
                            elif _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                wave_pRT_r, syn_spec_r, *_ = call_pRT(
                                    _mini_b7, p_ret, atm_r,
                                    _mini_b7["species_ret"],
                                    abundances, _mini_b7["MMW_ret"],
                                    _mini_b7["p0_ret"],
                                    _mini_b7["isothermal_ret"], T_equ,
                                    _mini_b7["two_point_T_ret"],
                                    _mini_b7["p_points_ret"],
                                    _mini_b7["t_points_ret"],
                                    _mini_b7["Kappa_IR_ret"],
                                    _mini_b7["Gamma_ret"], T_equ,
                                    None, None, use_easyCHEM=False,
                                )
                            else:  # 1D_CtoO_met
                                wave_pRT_r, syn_spec_r, *_ = call_pRT(
                                    _mini_b7, p_ret, atm_r,
                                    _mini_b7["species_ret"],
                                    abundances, _mini_b7["MMW_ret"],
                                    _mini_b7["p0_ret"],
                                    _mini_b7["isothermal_ret"],
                                    _mini_b7["T_equ"],
                                    _mini_b7["two_point_T_ret"],
                                    _mini_b7["p_points_ret"],
                                    _mini_b7["t_points_ret"],
                                    _mini_b7["Kappa_IR_ret"],
                                    _mini_b7["Gamma_ret"], _mini_b7["T_equ"],
                                    metallicity=met, C_to_O=c_to_o,
                                    use_easyCHEM=True, P_cloud=None,
                                    easychem_CtoO_ret=True,
                                )

                            # v_planet
                            if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                if not _mini_b7["significant_eccentricity"]:
                                    v_planet_r = get_V(
                                        K_p, _ph_ll, _berv_ll,
                                        _mini_b7["V_sys"], v_wind)
                                else:
                                    v_planet_r = get_V_eccentric(
                                        K_p, _ph_ll,
                                        _mini_b7["eccentricity"],
                                        _mini_b7["arg_periastron_w"],
                                        _berv_ll, _mini_b7["V_sys"], v_wind)
                            elif _mini_b7["Ret_dim"] == "1D_CtoO_met":
                                if not _mini_b7["significant_eccentricity"]:
                                    v_planet_r = get_V(
                                        _mini_b7["K_p"], _ph_ll, _berv_ll,
                                        _mini_b7["V_sys"], 0.)
                                else:
                                    v_planet_r = get_V_eccentric(
                                        _mini_b7["K_p"], _ph_ll,
                                        _mini_b7["eccentricity"],
                                        _mini_b7["arg_periastron_w"],
                                        _berv_ll, _mini_b7["V_sys"], 0.)
                            else:
                                if not _mini_b7["significant_eccentricity"]:
                                    v_planet_r = get_V(
                                        _mini_b7["K_p"], _ph_ll, _berv_ll,
                                        _mini_b7["V_sys"], _mini_b7["V_wind"])
                                else:
                                    v_planet_r = get_V_eccentric(
                                        _mini_b7["K_p"], _ph_ll,
                                        _mini_b7["eccentricity"],
                                        _mini_b7["arg_periastron_w"],
                                        _berv_ll, _mini_b7["V_sys"],
                                        _mini_b7["V_wind"])

                            # 1D_alpha[_Gibson22]: α scales the model depth via
                            # Scale_inj; all other modes keep the config value.
                            _smf_inp = (dict(_mini_b7, Scale_inj=alpha)
                                        if _mini_b7["Ret_dim"] in
                                           ("1D_alpha", "1D_alpha_Gibson22")
                                        else _mini_b7)
                            model_mat[hh], _ = spec_to_mat_fraction(
                                _smf_inp, _sjd_ll, _T0_ll, v_planet_r,
                                wave_star[_mini_b7["order_selection"][hh], :],
                                wave_pRT_r, syn_spec_r,
                                _mat_star_ret[_night_index][:, :n_pixels],
                                _with_ll, _wout_ll, _frac_ll,
                                include_star=False,
                            )

                            if _mini_b7["prepare_template"]:
                                if not _mini_b7["SYSREM_robust_halt"]:
                                    _sysrem_pass = None
                                if _mini_b7["preparing_pipeline"] != "Gibson22":
                                    model_mat_prepared[hh] = \
                                        preparing_pipeline(
                                            _mini_b7, model_mat[hh],
                                            _std_noise_ret[_night_index,
                                                           :_n_sp_ll,
                                                           hh*_n_px_b9:(hh+1)*_n_px_b9],
                                            wave_star[
                                                _mini_b7["order_selection"][hh],
                                                :],
                                            np.where(
                                                useful_spectral_points_snr_aux[
                                                    _night_index, hh, :])[0],
                                            np.where(
                                                mask_snr_ret_aux[
                                                    _night_index, hh, :])[0],
                                            _frac_ll, _ph_ll, _wout_ll,
                                            _sysrem_pass, None,
                                            tell_mask_threshold_Blain24=0.8,
                                            max_fit_BL19=False,
                                            sysrem_division=False,
                                            masks=False,
                                            correct_uncertainties=False,
                                            retrieval=True,
                                            mask_inter_retrieval=np.where(
                                                mask_inter_ret_aux[
                                                    _night_index, hh, :])[0],
                                            useful_spectral_points_inter_retrieval=np.where(
                                                useful_spectral_points_inter_aux[
                                                    _night_index, hh, :])[0],
                                        )
                                else:
                                    model_mat_prepared[hh] = \
                                        filter_model_singleorder(
                                            P[hh, _night_index, :, :],
                                            model_mat[hh],
                                            np.where(
                                                useful_spectral_points_inter_aux[
                                                    _night_index, hh, :])[0],
                                        )
                            else:
                                model_mat_prepared[hh] = np.copy(model_mat[hh])

                    else:
                        # Different_nights branch
                        _n_sp_ll  = n_spectra
                        _ph_ll    = phase
                        _berv_ll  = berv
                        _with_ll  = with_signal
                        _wout_ll  = without_signal
                        _frac_ll  = fraction
                        _sjd_ll   = syn_jd
                        _T0_ll    = T_0
                        _indices_ll = [
                            np.where(
                                _mini_b7["order_selection"] == v)[0][0]
                            for v in _mini_b7[
                                "order_selection_diffnights"][_night_index]
                        ]
                        model_mat = np.zeros(
                            (len(_indices_ll), _n_sp_ll, n_pixels))
                        model_mat_prepared = np.zeros_like(model_mat)

                        for _idx_ll, hh in enumerate(_indices_ll):
                            atm_r = _atm_list[hh]
                            if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                wave_pRT_r, syn_spec_r, *_ = call_pRT(
                                    _mini_b7, p_ret, atm_r,
                                    _mini_b7["species_ret"],
                                    abundances, _mini_b7["MMW_ret"],
                                    _mini_b7["p0_ret"],
                                    _mini_b7["isothermal_ret"], T_equ,
                                    _mini_b7["two_point_T_ret"],
                                    _mini_b7["p_points_ret"],
                                    _mini_b7["t_points_ret"],
                                    _mini_b7["Kappa_IR_ret"],
                                    _mini_b7["Gamma_ret"], T_equ,
                                    None, None, use_easyCHEM=False,
                                )
                            else:
                                wave_pRT_r, syn_spec_r, *_ = call_pRT(
                                    _mini_b7, p_ret, atm_r,
                                    _mini_b7["species_ret"],
                                    abundances, _mini_b7["MMW_ret"],
                                    _mini_b7["p0_ret"],
                                    _mini_b7["isothermal_ret"],
                                    _mini_b7["T_equ"],
                                    _mini_b7["two_point_T_ret"],
                                    _mini_b7["p_points_ret"],
                                    _mini_b7["t_points_ret"],
                                    _mini_b7["Kappa_IR_ret"],
                                    _mini_b7["Gamma_ret"], _mini_b7["T_equ"],
                                    metallicity=met, C_to_O=c_to_o,
                                    use_easyCHEM=True, P_cloud=None,
                                    easychem_CtoO_ret=True,
                                )

                            if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                v_planet_r = get_V(
                                    K_p, _ph_ll, _berv_ll,
                                    _mini_b7["V_sys"], v_wind)
                            else:
                                v_planet_r = get_V(
                                    _mini_b7["K_p"], _ph_ll, _berv_ll,
                                    _mini_b7["V_sys"], _mini_b7["V_wind"])

                            model_mat[_idx_ll], _ = \
                                spec_to_mat_fraction(
                                    _mini_b7, _sjd_ll, _T0_ll, v_planet_r,
                                    wave_star[
                                        _mini_b7["order_selection"][hh], :],
                                    wave_pRT_r, syn_spec_r,
                                    np.ones((_n_sp_ll, n_pixels)),
                                    _with_ll, _wout_ll, _frac_ll,
                                    include_star=False,
                                )
                            if _mini_b7["prepare_template"]:
                                if not _mini_b7["SYSREM_robust_halt"]:
                                    _sysrem_pass = None
                                if _mini_b7["preparing_pipeline"] != "Gibson22":
                                    model_mat_prepared[_idx_ll] = \
                                        preparing_pipeline(
                                            _mini_b7,
                                            model_mat[_idx_ll],
                                            _std_noise_ret[_night_index][
                                                :_n_sp_ll,
                                                hh*_n_px_b9:(hh+1)*_n_px_b9],
                                            wave_star[
                                                _mini_b7["order_selection"][hh],
                                                :],
                                            np.where(
                                                useful_spectral_points_snr_aux[
                                                    _night_index, hh, :])[0],
                                            np.where(
                                                mask_snr_ret_aux[
                                                    _night_index, hh, :])[0],
                                            _frac_ll, _ph_ll, _wout_ll,
                                            _sysrem_pass, None,
                                            tell_mask_threshold_Blain24=0.8,
                                            max_fit_BL19=False,
                                            sysrem_division=False,
                                            masks=False,
                                            correct_uncertainties=False,
                                            retrieval=True,
                                            mask_inter_retrieval=np.where(
                                                mask_inter_ret_aux[
                                                    _night_index, hh, :])[0],
                                            useful_spectral_points_inter_retrieval=np.where(
                                                useful_spectral_points_inter_aux[
                                                    _night_index, hh, :])[0],
                                        )
                                else:
                                    model_mat_prepared[_idx_ll] = \
                                        filter_model_singleorder(
                                            P[hh, _night_index,
                                              :_n_sp_ll, :_n_sp_ll],
                                            model_mat[_idx_ll],
                                            np.where(
                                                useful_spectral_points_inter_aux[
                                                    _night_index, hh, :])[0],
                                        )
                            else:
                                model_mat_prepared[_idx_ll] = np.copy(
                                    model_mat[_idx_ll])

                    # transpose+reshape to (n_spectra, n_orders*n_pixels)
                    model_mat = np.transpose(model_mat, (1, 0, 2))
                    model_mat = model_mat.reshape(
                        model_mat.shape[0], -1)
                    model_mat_prepared = np.transpose(
                        model_mat_prepared, (1, 0, 2))
                    model_mat_prepared = model_mat_prepared.reshape(
                        model_mat_prepared.shape[0], -1)
                    _ctx["sysrem_pass"] = _sysrem_pass

                    # ---- with_signal_ret selection for time-resolved ----
                    with_signal_ret = with_signal
                    if _interval_tracker is not None:
                        _half_ll = len(with_signal) // 2
                        if _interval_tracker == 0:
                            with_signal_ret = ingress_idx
                        elif _interval_tracker == 1:
                            with_signal_ret = with_signal[:_half_ll]
                        elif _interval_tracker == 2:
                            with_signal_ret = with_signal[_half_ll:]
                        elif _interval_tracker == 3:
                            with_signal_ret = egress_idx

                    # ---- log-likelihood ----
                    if _mini_b7["logL_choice"] == "BL19":
                        if _mini_b7["Different_nights"]:
                            for n in with_signal_ret:
                                _d = _mat_res_ret[n,
                                    useful_spectral_points_ret[_night_index]]
                                _d  = _d - np.mean(_d)
                                _tpl = model_mat[n,
                                    useful_spectral_points_ret[_night_index]]
                                _tpl = _tpl - np.mean(_tpl)
                                sf2 = np.mean(_d**2)
                                sg2 = np.mean(_tpl**2)
                                R   = np.mean(_d * _tpl)
                                log_likelihood += (
                                    -(len(_d) / 2.)
                                    * np.log(sf2 - 2.*R + sg2))
                        else:
                            for n in with_signal_ret:
                                _d = (_mat_res_ret[_night_index, n,
                                        useful_spectral_points_ret[
                                            _night_index, :]]
                                      - np.mean(_mat_res_ret[_night_index, n,
                                        useful_spectral_points_ret[
                                            _night_index, :]]))
                                _tpl = (model_mat[n,
                                        useful_spectral_points_ret[
                                            _night_index, :]]
                                        - np.mean(model_mat[n,
                                        useful_spectral_points_ret[
                                            _night_index, :]]))
                                sf2 = np.mean(_d**2)
                                sg2 = np.mean(_tpl**2)
                                R   = np.mean(_d * _tpl)
                                log_likelihood += (
                                    -(len(_d) / 2.)
                                    * np.log(sf2 - 2.*R + sg2))

                    elif _mini_b7["logL_choice"] == "Blain24":
                        if _mini_b7["Different_nights"]:
                            for n in with_signal_ret:
                                log_likelihood += -0.5 * np.sum(
                                    ((_mat_res_ret[n,
                                       useful_spectral_points_ret[_night_index]]
                                      - model_mat_prepared[n,
                                       useful_spectral_points_ret[_night_index]])
                                     / _propag_ret[n,
                                       useful_spectral_points_ret[_night_index]]
                                     )**2)
                        else:
                            _up9 = useful_spectral_points_ret[_night_index, :]
                            # σ error model (Cheverall26 pipeline): MAD-of-residuals
                            # (paper-faithful: σ_j = 1.4826·MAD over in-transit
                            # exposures per channel, time-independent) vs the
                            # default pipeline-propagated σ.  Cached per night.
                            _em9 = _mini_b7.get("ret_error_model")
                            _chev9 = (_mini_b7["preparing_pipeline"] == "Cheverall26")
                            if _em9 in ("mad_residual", "cov_acf", "cov_lsf") and _chev9:
                                # per-channel MAD sigma (shared cache)
                                _sk9 = f"_sigma_mad_{_night_index}"
                                if _sk9 not in _ctx:
                                    _rr9 = _mat_res_ret[_night_index][
                                        np.asarray(with_signal_ret), :]
                                    _md9 = np.median(_rr9, axis=0)
                                    _sg9 = 1.4826 * np.median(
                                        np.abs(_rr9 - _md9), axis=0)
                                    _sg9[~np.isfinite(_sg9) | (_sg9 <= 0)] = np.inf
                                    _ctx[_sk9] = _sg9
                                if _em9 == "mad_residual":
                                    _sig9 = _ctx[_sk9][_up9]
                                    for n in with_signal_ret:
                                        log_likelihood += -0.5 * np.sum(
                                            ((_mat_res_ret[_night_index, n, _up9]
                                              - model_mat_prepared[n, _up9])
                                             / _sig9)**2)
                                else:
                                    # banded block-diagonal noise covariance
                                    # C = D R D (D=diag(sigma_MAD), R=short-range
                                    # per-order correlation).  The determinant
                                    # cancels in the model-vs-null Bayes factor,
                                    # so only (r-m)^T C^-1 (r-m) is evaluated.
                                    from exoplore.analysis import covnoise as _covn
                                    _ckey = f"_covfac_{_night_index}_{_em9}"
                                    if _ckey not in _ctx:
                                        _resin = _mat_res_ret[_night_index][
                                            np.asarray(with_signal_ret), :]
                                        _ctx[_ckey] = _covn.build_factors(
                                            _resin, _ctx[_sk9],
                                            useful_spectral_points_ret[
                                                _night_index, :],
                                            _n_px_b9, _n_orders_b9,
                                            kernel=("acf" if _em9 == "cov_acf"
                                                    else "lsf"),
                                            kmax=8, fwhm=3.3)
                                    _fac9 = _ctx[_ckey]
                                    for n in with_signal_ret:
                                        _diff9 = (_mat_res_ret[_night_index, n, :]
                                                  - model_mat_prepared[n, :])
                                        log_likelihood += -0.5 * _covn.quad_form(
                                            _fac9, _diff9)
                            else:
                                for n in with_signal_ret:
                                    log_likelihood += -0.5 * np.sum(
                                        ((_mat_res_ret[_night_index, n, _up9]
                                          - model_mat_prepared[n, _up9])
                                         / _propag_ret[_night_index, n, _up9])**2)

                    elif _mini_b7["logL_choice"] == "Gibson22":
                        # Enforce beta prior (reads from prior_bounds config).
                        # beta is the last sampled parameter: index 4 for 1D_Gibson22,
                        # index 3 for 1D_alpha_Gibson22.
                        _beta_lo, _beta_hi = _PRIOR_BOUNDS_B9[
                            _mini_b7["Ret_dim"]][-1]
                        if not (_beta_lo <= beta <= _beta_hi):
                            return -np.inf
                        if _mini_b7["Different_nights"]:
                            _Npix_g22  = np.count_nonzero(
                                useful_spectral_points_ret[_night_index])
                            _Nfr_g22   = len(with_signal_ret)
                            _chi2_g22  = 0.
                            for n in with_signal_ret:
                                _res_g = (_mat_res_ret[n,
                                    useful_spectral_points_ret[_night_index]]
                                    - model_mat_prepared[n,
                                    useful_spectral_points_ret[_night_index]])
                                _sig_g = _propag_ret[n,
                                    useful_spectral_points_ret[_night_index]]
                                _chi2_g22 += np.sum(
                                    (_res_g / (beta * _sig_g))**2)
                            log_likelihood += -0.5 * _chi2_g22
                            log_likelihood += (
                                -_Nfr_g22 * _Npix_g22 * np.log(beta))
                        else:
                            _Npix_g22 = np.count_nonzero(
                                useful_spectral_points_ret[_night_index, :])
                            _Nfr_g22  = len(with_signal_ret)
                            _chi2_g22 = 0.
                            for n in with_signal_ret:
                                _res_g = (
                                    _mat_res_ret[_night_index, n,
                                        useful_spectral_points_ret[
                                            _night_index, :]]
                                    - model_mat_prepared[n,
                                        useful_spectral_points_ret[
                                            _night_index, :]])
                                _sig_g = _propag_ret[_night_index, n,
                                    useful_spectral_points_ret[
                                        _night_index, :]]
                                _chi2_g22 += np.sum(
                                    (_res_g / (beta * _sig_g))**2)
                            log_likelihood += -0.5 * _chi2_g22
                            log_likelihood += (
                                -_Nfr_g22 * _Npix_g22 * np.log(beta))

                    return log_likelihood

                # ==============================================================
                # Combined-nights loglike  (Retrieval_choice == 4)
                # Master lines 7441-7984
                # ==============================================================
                def _loglike_1d_combined(cube, ndim, nparams):
                    _sysrem_pass      = _ctx["sysrem_pass"]
                    _night_index      = _ctx["night_index"]
                    _idx_interval     = _ctx["idx_interval"]
                    _interval_tracker = _ctx["interval_tracker"]
                    _atm_list         = _ctx["atmosphere_ret_list"]

                    # param extraction, same as single-night
                    if _mini_b7["Ret_dim"] == "1D":
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]
                    elif _mini_b7["Ret_dim"] == "1D_CtoO_met":
                        c_to_o = cube[0]; met = cube[1]
                    elif _mini_b7["Ret_dim"] == "1D_Gibson22":
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]; beta = cube[4]
                    elif _mini_b7["Ret_dim"] == "1D_alpha":
                        K_p = cube[0]; v_wind = cube[1]; alpha = cube[2]
                    elif _mini_b7["Ret_dim"] == "1D_alpha_Gibson22":
                        K_p = cube[0]; v_wind = cube[1]
                        alpha = cube[2]; beta = cube[3]

                    log_likelihood = 0.

                    if _mini_b7["Ret_dim"] in ["1D", "1D_Gibson22"]:
                        abundances = np.asarray(
                            [_mini_b7["vmr"][0], _mini_b7["vmr"][1], 10.**log10_X1])
                    else:
                        abundances = np.asarray(
                            [_mini_b7["vmr"][0], _mini_b7["vmr"][1]])

                    for jj in range(retrieved_nights):
                        if not _mini_b7["Different_nights"]:
                            _n_sp_c   = n_spectra
                            _ph_c     = phase
                            _berv_c   = berv
                            _with_c   = with_signal
                            _wout_c   = without_signal
                            _frac_c   = fraction
                            _sjd_c    = syn_jd
                            _T0_c     = T_0
                            model_mat = np.zeros(
                                (_n_ord_sel_b9, _n_sp_c, n_pixels))
                            model_mat_prepared = np.zeros_like(model_mat)

                            for hh in range(_n_orders_b9):
                                atm_r = _atm_list[hh]
                                if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                    wave_pRT_r, syn_spec_r, *_ = \
                                        call_pRT(
                                            _mini_b7, p_ret, atm_r,
                                            _mini_b7["species_ret"],
                                            abundances, _mini_b7["MMW_ret"],
                                            _mini_b7["p0_ret"],
                                            _mini_b7["isothermal_ret"], T_equ,
                                            _mini_b7["two_point_T_ret"],
                                            _mini_b7["p_points_ret"],
                                            _mini_b7["t_points_ret"],
                                            _mini_b7["Kappa_IR_ret"],
                                            _mini_b7["Gamma_ret"], T_equ,
                                            None, None, use_easyCHEM=False,
                                        )
                                else:
                                    wave_pRT_r, syn_spec_r, *_ = \
                                        call_pRT(
                                            _mini_b7, p_ret, atm_r,
                                            _mini_b7["species_ret"],
                                            abundances, _mini_b7["MMW_ret"],
                                            _mini_b7["p0_ret"],
                                            _mini_b7["isothermal_ret"],
                                            _mini_b7["T_equ"],
                                            _mini_b7["two_point_T_ret"],
                                            _mini_b7["p_points_ret"],
                                            _mini_b7["t_points_ret"],
                                            _mini_b7["Kappa_IR_ret"],
                                            _mini_b7["Gamma_ret"],
                                            _mini_b7["T_equ"],
                                            metallicity=log10_Z, C_to_O=0.55,
                                            use_easyCHEM=True,
                                            P_cloud=10.**log10_P,
                                        )

                                if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                    v_planet_r = get_V(
                                        K_p, _ph_c, _berv_c,
                                        _mini_b7["V_sys"], v_wind)
                                else:
                                    v_planet_r = get_V(
                                        _mini_b7["K_p"], _ph_c, _berv_c,
                                        _mini_b7["V_sys"], _mini_b7["V_wind"])

                                model_mat[hh], _ = spec_to_mat_fraction(
                                    _mini_b7, _sjd_c, _T0_c, v_planet_r,
                                    wave_star[
                                        _mini_b7["order_selection"][hh], :],
                                    wave_pRT_r, syn_spec_r,
                                    _mat_star_ret[jj][:, :n_pixels],
                                    _with_c, _wout_c, _frac_c,
                                    include_star=False,
                                )
                                if _mini_b7["prepare_template"]:
                                    if not _mini_b7["SYSREM_robust_halt"]:
                                        _sysrem_pass = None
                                    if _mini_b7["preparing_pipeline"] != "Gibson22":
                                        model_mat_prepared[hh] = \
                                            preparing_pipeline(
                                                _mini_b7, model_mat[hh],
                                                _std_noise_ret[jj,
                                                               :_n_sp_c, :],
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh], :],
                                                np.where(
                                                    useful_spectral_points_snr_aux[
                                                        jj, hh, :])[0],
                                                np.where(
                                                    mask_snr_ret_aux[
                                                        jj, hh, :])[0],
                                                _frac_c, _ph_c, _wout_c,
                                                _sysrem_pass, None,
                                                tell_mask_threshold_Blain24=0.8,
                                                max_fit_BL19=False,
                                                sysrem_division=False,
                                                masks=False,
                                                correct_uncertainties=False,
                                                retrieval=True,
                                                mask_inter_retrieval=np.where(
                                                    mask_inter_ret_aux[
                                                        jj, hh, :])[0],
                                                useful_spectral_points_inter_retrieval=np.where(
                                                    useful_spectral_points_inter_aux[
                                                        jj, hh, :])[0],
                                            )
                                    else:
                                        model_mat_prepared[hh] = \
                                            filter_model_singleorder(
                                                P[hh, jj, :, :],
                                                model_mat[hh],
                                                np.where(
                                                    useful_spectral_points_inter_aux[
                                                        jj, hh, :])[0],
                                            )
                                else:
                                    model_mat_prepared[hh] = np.copy(
                                        model_mat[hh])

                        else:
                            # Different_nights inside combined loglike
                            _indices_c = [
                                np.where(
                                    _mini_b7["order_selection"] == v)[0][0]
                                for v in _mini_b7[
                                    "order_selection_diffnights"][_night_index]
                            ]
                            model_mat = np.zeros(
                                (len(_indices_c), n_spectra, n_pixels))
                            model_mat_prepared = np.zeros_like(model_mat)

                            for _ic, hh in enumerate(_indices_c):
                                atm_r = _atm_list[hh]
                                if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                    wave_pRT_r, syn_spec_r, *_ = \
                                        call_pRT(
                                            _mini_b7, p_ret, atm_r,
                                            _mini_b7["species_ret"],
                                            abundances, _mini_b7["MMW_ret"],
                                            _mini_b7["p0_ret"],
                                            _mini_b7["isothermal_ret"], T_equ,
                                            _mini_b7["two_point_T_ret"],
                                            _mini_b7["p_points_ret"],
                                            _mini_b7["t_points_ret"],
                                            _mini_b7["Kappa_IR_ret"],
                                            _mini_b7["Gamma_ret"], T_equ,
                                            None, None, use_easyCHEM=False,
                                        )
                                else:
                                    wave_pRT_r, syn_spec_r, *_ = \
                                        call_pRT(
                                            _mini_b7, p_ret, atm_r,
                                            _mini_b7["species_ret"],
                                            abundances, _mini_b7["MMW_ret"],
                                            _mini_b7["p0_ret"],
                                            _mini_b7["isothermal_ret"],
                                            _mini_b7["T_equ"],
                                            _mini_b7["two_point_T_ret"],
                                            _mini_b7["p_points_ret"],
                                            _mini_b7["t_points_ret"],
                                            _mini_b7["Kappa_IR_ret"],
                                            _mini_b7["Gamma_ret"],
                                            _mini_b7["T_equ"],
                                            metallicity=log10_Z, C_to_O=0.55,
                                            use_easyCHEM=True,
                                            P_cloud=10.**log10_P,
                                        )
                                if _mini_b7["Ret_dim"] != "1D_CtoO_met":
                                    v_planet_r = get_V(
                                        K_p, phase, berv,
                                        _mini_b7["V_sys"], v_wind)
                                else:
                                    v_planet_r = get_V(
                                        _mini_b7["K_p"], phase, berv,
                                        _mini_b7["V_sys"], _mini_b7["V_wind"])
                                model_mat[_ic], _ = spec_to_mat_fraction(
                                    _mini_b7, syn_jd, T_0, v_planet_r,
                                    wave_star[
                                        _mini_b7["order_selection"][hh], :],
                                    wave_pRT_r, syn_spec_r,
                                    np.ones((n_spectra, n_pixels)),
                                    with_signal, without_signal, fraction,
                                    include_star=False,
                                )
                                if _mini_b7["prepare_template"]:
                                    if not _mini_b7["SYSREM_robust_halt"]:
                                        _sysrem_pass = None
                                    if _mini_b7["preparing_pipeline"] != "Gibson22":
                                        model_mat_prepared[_ic] = \
                                            preparing_pipeline(
                                                _mini_b7, model_mat[_ic],
                                                _std_noise_ret[jj][
                                                    :n_spectra, :],
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh], :],
                                                np.where(
                                                    useful_spectral_points_snr_aux[
                                                        jj, hh, :])[0],
                                                np.where(
                                                    mask_snr_ret_aux[
                                                        jj, hh, :])[0],
                                                fraction, phase, without_signal,
                                                _sysrem_pass, None,
                                                tell_mask_threshold_Blain24=0.8,
                                                max_fit_BL19=False,
                                                sysrem_division=False,
                                                masks=False,
                                                correct_uncertainties=False,
                                                retrieval=True,
                                                mask_inter_retrieval=np.where(
                                                    mask_inter_ret_aux[
                                                        jj, hh, :])[0],
                                                useful_spectral_points_inter_retrieval=np.where(
                                                    useful_spectral_points_inter_aux[
                                                        jj, hh, :])[0],
                                            )
                                    else:
                                        model_mat_prepared[_ic] = \
                                            filter_model_singleorder(
                                                P[hh, jj,
                                                  :n_spectra, :n_spectra],
                                                model_mat[_ic],
                                                np.where(
                                                    useful_spectral_points_inter_aux[
                                                        jj, hh, :])[0],
                                            )
                                else:
                                    model_mat_prepared[_ic] = np.copy(
                                        model_mat[_ic])

                        # transpose + reshape
                        model_mat = np.transpose(model_mat, (1, 0, 2))
                        model_mat = model_mat.reshape(
                            model_mat.shape[0], -1)
                        model_mat_prepared = np.transpose(
                            model_mat_prepared, (1, 0, 2))
                        model_mat_prepared = model_mat_prepared.reshape(
                            model_mat_prepared.shape[0], -1)

                        # with_signal_ret selection
                        with_signal_ret = with_signal
                        if _interval_tracker is not None:
                            _half_c = len(with_signal) // 2
                            if _interval_tracker == 0:
                                with_signal_ret = ingress_idx
                            elif _interval_tracker == 1:
                                with_signal_ret = with_signal[:_half_c]
                            elif _interval_tracker == 2:
                                with_signal_ret = with_signal[_half_c:]
                            elif _interval_tracker == 3:
                                with_signal_ret = egress_idx

                        # log-likelihood accumulation
                        if _mini_b7["logL_choice"] == "BL19":
                            if _mini_b7["Different_nights"]:
                                for n in with_signal_ret:
                                    _d = (_mat_res_ret[n,
                                          useful_spectral_points_ret[_night_index]]
                                          - np.mean(_mat_res_ret[n,
                                          useful_spectral_points_ret[_night_index]]))
                                    _tpl = (model_mat[n,
                                            useful_spectral_points_ret[_night_index]]
                                            - np.mean(model_mat[n,
                                            useful_spectral_points_ret[_night_index]]))
                                    sf2 = np.mean(_d**2)
                                    sg2 = np.mean(_tpl**2)
                                    R   = np.mean(_d * _tpl)
                                    log_likelihood += (
                                        -(len(_d)/2.) * np.log(sf2-2.*R+sg2))
                            else:
                                for n in with_signal_ret:
                                    _d = (_mat_res_ret[_night_index, n,
                                          useful_spectral_points_ret[_night_index,:]]
                                          - np.mean(_mat_res_ret[_night_index, n,
                                          useful_spectral_points_ret[_night_index,:]]))
                                    _tpl = (model_mat[n,
                                            useful_spectral_points_ret[_night_index,:]]
                                            - np.mean(model_mat[n,
                                            useful_spectral_points_ret[_night_index,:]]))
                                    sf2 = np.mean(_d**2)
                                    sg2 = np.mean(_tpl**2)
                                    R   = np.mean(_d * _tpl)
                                    log_likelihood += (
                                        -(len(_d)/2.) * np.log(sf2-2.*R+sg2))

                        elif _mini_b7["logL_choice"] == "Blain24":
                            if _mini_b7["Different_nights"]:
                                for n in with_signal_ret:
                                    log_likelihood += -0.5 * np.sum(
                                        ((_mat_res_ret[n,
                                           useful_spectral_points_ret[_night_index]]
                                          - model_mat_prepared[n,
                                           useful_spectral_points_ret[_night_index]])
                                         / _propag_ret[n,
                                           useful_spectral_points_ret[_night_index]]
                                         )**2)
                            else:
                                for n in with_signal_ret:
                                    log_likelihood += -0.5 * np.sum(
                                        ((_mat_res_ret[_night_index, n,
                                           useful_spectral_points_ret[_night_index,:]]
                                          - model_mat_prepared[n,
                                           useful_spectral_points_ret[_night_index,:]])
                                         / _propag_ret[_night_index, n,
                                           useful_spectral_points_ret[_night_index,:]]
                                         )**2)

                        elif _mini_b7["logL_choice"] == "Gibson22":
                            _beta_lo2, _beta_hi2 = _PRIOR_BOUNDS_B9[
                                _mini_b7["Ret_dim"]][-1]
                            if not (_beta_lo2 <= beta <= _beta_hi2):
                                return -np.inf
                            if _mini_b7["Different_nights"]:
                                _Npix_g = np.count_nonzero(
                                    useful_spectral_points_ret[_night_index])
                                _Nfr_g  = len(with_signal_ret)
                                _chi2_g = 0.
                                for n in with_signal_ret:
                                    _r = (_mat_res_ret[n,
                                          useful_spectral_points_ret[_night_index]]
                                          - model_mat_prepared[n,
                                          useful_spectral_points_ret[_night_index]])
                                    _s = _propag_ret[n,
                                         useful_spectral_points_ret[_night_index]]
                                    _chi2_g += np.sum((_r/(beta*_s))**2)
                                log_likelihood += (-0.5 * _chi2_g
                                                   - _Nfr_g * _Npix_g
                                                   * np.log(beta))
                            else:
                                _Npix_g = np.count_nonzero(
                                    useful_spectral_points_ret[_night_index,:])
                                _Nfr_g  = len(with_signal_ret)
                                _chi2_g = 0.
                                for n in with_signal_ret:
                                    _r = (_mat_res_ret[_night_index, n,
                                          useful_spectral_points_ret[_night_index,:]]
                                          - model_mat_prepared[n,
                                          useful_spectral_points_ret[_night_index,:]])
                                    _s = _propag_ret[_night_index, n,
                                         useful_spectral_points_ret[_night_index,:]]
                                    _chi2_g += np.sum((_r/(beta*_s))**2)
                                log_likelihood += (-0.5 * _chi2_g
                                                   - _Nfr_g * _Npix_g
                                                   * np.log(beta))

                    _ctx["sysrem_pass"] = _sysrem_pass
                    return log_likelihood

                # choose which loglike to pass to the sampler
                if _mini_b7["Retrieval_choice"] != 4:
                    loglike = _loglike_1d_single
                else:
                    loglike = _loglike_1d_combined

                # MCMC log_prior / log_prob (need loglike defined first)
                if _mini_b7["Sampler_choice"] == "MCMC":
                    _cfg_bounds_mc = _mini_b7.get("PRIOR_BOUNDS", {})
                    _bds_mc = (_cfg_bounds_mc.get(_mini_b7["Ret_dim"])
                               or _PRIOR_BOUNDS_B9[_mini_b7["Ret_dim"]])
                    def log_prior_b9(theta):
                        for val, (_lo9, _hi9) in zip(theta, _bds_mc):
                            if not (_lo9 <= val <= _hi9):
                                return -np.inf
                        return 0.
                    def log_prob_b9(theta):
                        lp = log_prior_b9(theta)
                        if not np.isfinite(lp):
                            return -np.inf
                        return lp + loglike(theta, len(theta), len(theta))
                    ndim_b9     = len(_bds_mc)
                    nwalkers_b9 = _mini_b7["n_walkers"]
                    p0_b9       = np.array([
                        [np.random.uniform(_lo9, _hi9)
                         for (_lo9, _hi9) in _bds_mc]
                        for _ in range(nwalkers_b9)
                    ])

                # ==============================================================
                # Helper: run sampler + extract + corner + save
                # (shared between Retrieval_choice != 6 and ==6 loops)
                # ==============================================================
                def _run_and_save_b9(night_idx, base_dir_plot, label_suffix):
                    """Run the sampler for one night/interval and save outputs."""
                    if _pymn is None:
                        return None, None
                    _out_pfx = (
                        f"{_base_dir_b9}/{retrieval_name}{label_suffix}_"
                    )

                    # ── Likelihood emulator: train on night 0 after setup ──────
                    # atmosphere_ret_list and data matrices are fully initialised
                    # by the time this function is called, so loglike is valid.
                    _ll_active = loglike   # default: use true pRT loglike
                    if cfg.retrieval.use_likelihood_emulator and night_idx == 0:
                        import os as _os9
                        from exoplore.retrieval.likelihood_emulator import (
                            LoglikeEmulator as _LLE)
                        _ll_bounds = _mini_b7.get("PRIOR_BOUNDS", {})
                        _ll_bds    = (_ll_bounds.get(_mini_b7["Ret_dim"])
                                      or _PRIOR_BOUNDS_B9[_mini_b7["Ret_dim"]])
                        _ll_n      = len(_ll_bds)
                        _ll_prior  = np.array(_ll_bds, dtype=np.float32).T
                        _ll_save   = _os9.path.join(
                            _mini_b7["plots_dir"], "likelihood_emulator")

                        def _ll_wrap(_theta):
                            _c = np.array(_theta, dtype=float)
                            return loglike(_c, _ll_n, _ll_n)

                        # Sequential mode (recommended): concentrates samples
                        # near the posterior.  Per-round data saved to disk
                        # so completed rounds are skipped on re-run (reuse).
                        # n_per_round: R1=1000 uniform, R2=2000, R3=2000
                        # focused, total 5000 calls (~42 min for 1D).
                        print(f"\n  [loglike-emul] Sequential training "
                              f"(3 rounds, save_dir={_ll_save})...")
                        _ll_emu = _LLE.train_sequential(
                            loglike_fn=_ll_wrap,
                            prior_bounds=_ll_prior,
                            save_dir=_ll_save,
                            n_per_round=(1000, 2000, 2000),
                            verbose=True,
                        )
                        _ll_emu.save(_ll_save)
                        print(f"  [loglike-emul] Saved to {_ll_save}\n")

                        _ll_n2 = _ll_n
                        def _ll_active(_cube, _ndim, _npar,
                                       _em=_ll_emu, _n=_ll_n2):
                            return _em.predict(
                                np.array(_cube[:_n], dtype=np.float32))

                    if _mini_b7["Sampler_choice"] == "Nested_sampling":
                        _pymn.run(
                            _ll_active, prior,
                            n_dims=n_params,
                            outputfiles_basename=_out_pfx,
                            resume=False, verbose=True,
                            evidence_tolerance=0.5,
                            sampling_efficiency=0.8,
                            n_iter_before_update=100,
                            const_efficiency_mode=_mini_b7[
                                "Multinest_Constant_Eff_Mode"],
                            n_live_points=_mini_b7["Multinest_live_points"],
                            max_iter=0,
                        )
                        # ── Analyzer: try the pymultinest Analyzer first;
                        # fall back to reading the plain-text MultiNest
                        # output files directly when the Analyzer's loadtxt
                        # fails on Fortran fixed-point exponents (e.g.
                        # "1.23-313" instead of "1.23E-313").
                        _sa = None
                        _sa_pts = None
                        try:
                            _sa = _pymn.Analyzer(
                                n_params=n_params,
                                outputfiles_basename=_out_pfx)
                            _sa_pts = _sa.get_stats()
                            _json_b9.dump(
                                _sa_pts,
                                open(f"{_out_pfx}stats.json", "w"),
                                indent=4)
                            print("  marginal likelihood:")
                            print("    ln Z = %.4g +- %.4g" % (
                                _sa_pts["global evidence"],
                                _sa_pts["global evidence error"]))
                            print("  parameters:")
                            for _p9, _m9 in zip(parameters,
                                                 _sa_pts["marginals"]):
                                _lo9, _hi9 = _m9["1sigma"]
                                _med9      = _m9["median"]
                                _sig9      = (_hi9 - _lo9) / 2
                                _ii9 = 3 if _sig9 == 0 else max(
                                    0, int(-np.floor(np.log10(
                                        max(_sig9, 1e-99)))) + 1)
                                _fmt9 = "%%.%df" % _ii9
                                print(("\t".join(
                                    ["    %-15s" + _fmt9 + " +- " + _fmt9])
                                    % (_p9, _med9, _sig9)))
                            _dat9 = _sa.get_data()[:, 2:]
                            _wts9 = _sa.get_data()[:, 0]
                        except Exception as _e9_sa:
                            print(f"  WARNING: Analyzer failed ({_e9_sa}).")
                            print("  Falling back to post_equal_weights.dat")
                            # post_equal_weights has n_params cols + logL col;
                            # read only the first n_params cols (standard E fmt)
                            _pew = f"{_out_pfx}post_equal_weights.dat"
                            try:
                                _raw9 = np.genfromtxt(
                                    _pew,
                                    usecols=list(range(n_params)),
                                    invalid_raise=False)
                                _dat9 = _raw9
                                _wts9 = np.ones(len(_dat9))
                                # Parse _stats.dat for summary
                                _stats_f = f"{_out_pfx}stats.dat"
                                print("  Posterior means from stats.dat:")
                                with open(_stats_f) as _sf9:
                                    for _l9 in _sf9:
                                        print("   ", _l9.rstrip())
                            except Exception as _e9_b:
                                print(f"  Could not read fallback: {_e9_b}")
                                return None, None

                        _mask9    = np.ones(len(_dat9), dtype=bool)
                        if _corner is not None:
                            import matplotlib.pyplot as _plt9
                            _plt9.close()
                            _tick9 = {"labelsize": 12}
                            _no_truths9 = (
                                _mini_b7.get("use_real_data", False)
                                or (cfg.atmosphere.limb_asymmetries
                                    and _mini_b7["Ret_dim"] not in ("2D",))
                            )
                            _truths_1d9  = (None if _no_truths9 else
                                            [np.log10(_mini_b7["vmr"][2]),
                                             _mini_b7["K_p"], _mini_b7["T_equ"],
                                             _mini_b7["V_wind"]])
                            _truths_g229 = (None if _no_truths9 else
                                            [np.log10(_mini_b7["vmr"][2]),
                                             _mini_b7["K_p"], _mini_b7["T_equ"],
                                             _mini_b7["V_wind"], 1.0])
                            if _mini_b7["Ret_dim"] == "1D":
                                _fig9 = _corner.corner(
                                    _dat9[_mask9, :],
                                    weights=_wts9[_mask9],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False,
                                    title_fmt=".2E",
                                    truths=_truths_1d9,
                                    quantiles=[0.16, 0.5, 0.84],
                                    color="k", truth_color="firebrick",
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18})
                            elif _mini_b7["Ret_dim"] in ["1D_Gibson22", "Blain24_beta"]:
                                _fig9 = _corner.corner(
                                    _dat9[_mask9, :],
                                    weights=_wts9[_mask9],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False,
                                    title_fmt=".2E",
                                    truths=_truths_g229,
                                    quantiles=[0.16, 0.5, 0.84],
                                    color="k", truth_color="firebrick",
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18})
                            else:
                                _fig9 = _corner.corner(
                                    _dat9[_mask9, :],
                                    weights=_wts9[_mask9],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False,
                                    title_fmt=".2E",
                                    quantiles=[0.16, 0.5, 0.84],
                                    color="k",
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18})
                            for _ax9 in _fig9.get_axes():
                                _ax9.tick_params(axis="both", **_tick9)
                            _plt9.savefig(
                                f"{base_dir_plot}/{retrieval_name}"
                                f"{label_suffix}_corner.pdf")
                            _plt9.show()
                            _plt9.close()

                        # Save dat/weights/maskpoints
                        np.savez_compressed(
                            f"{_base_dir_b9}/{retrieval_name}"
                            f"{label_suffix}_dat_{_sfx9}", a=_dat9)
                        np.savez_compressed(
                            f"{_base_dir_b9}/{retrieval_name}"
                            f"{label_suffix}_weights_{_sfx9}", a=_wts9)
                        np.savez_compressed(
                            f"{_base_dir_b9}/{retrieval_name}"
                            f"{label_suffix}_maskpoints_{_sfx9}", a=_mask9)
                        return _sa, _sa_pts

                    elif _mini_b7["Sampler_choice"] == "MCMC":
                        if _emcee is None:
                            return None, None
                        # Serial evaluation (pool=None).
                        # ThreadPool is counterproductive here: pRT uses
                        # OpenMP internally, which fights Python threads and
                        # serialises everything with extra contention.
                        # atmosphere_ret_list is pre-loaded so each loglike
                        # call is only spectrum computation (~0.4 s); with
                        # minimum walkers (2-3× n_params) serial emcee is
                        # fast enough and gives correct autocorrelation stats.
                        _samp9 = _emcee.EnsembleSampler(
                            nwalkers_b9, ndim_b9, log_prob_b9,
                            pool=None)
                        _samp9.run_mcmc(
                            p0_b9, _mini_b7["n_steps_MCMC"], progress=True)
                        _samples9 = _samp9.get_chain(
                            discard=_mini_b7["MCMC_burnin"], thin=10, flat=True)
                        if _corner is not None:
                            import matplotlib.pyplot as _plt9
                            _PNAMES9 = {
                                "1D": [r"$\log_{10}X_1$", r"$K_p$",
                                       r"$T_{\rm eq}$", r"$v_{\rm wind}$"],
                                "1D_Gibson22": [r"$\log_{10}X_1$", r"$K_p$",
                                           r"$T_{\rm eq}$", r"$v_{\rm wind}$",
                                           r"$\beta$"],
                            }
                            _lbls9 = _PNAMES9.get(_mini_b7["Ret_dim"],
                                                   parameters)
                            _no_truths9 = (
                                _mini_b7.get("use_real_data", False)
                                or (cfg.atmosphere.limb_asymmetries
                                    and _mini_b7["Ret_dim"]
                                    not in ("2D",))
                            )
                            _truths9 = None if _no_truths9 else {
                                "1D": [np.log10(_mini_b7["vmr"][2]),
                                       _mini_b7["K_p"], _mini_b7["T_equ"],
                                       _mini_b7["V_wind"]],
                                "1D_Gibson22": [np.log10(_mini_b7["vmr"][2]),
                                           _mini_b7["K_p"], _mini_b7["T_equ"],
                                           _mini_b7["V_wind"], 1.0],
                            }.get(_mini_b7["Ret_dim"])
                            _fig9 = _corner.corner(
                                _samples9, labels=_lbls9,
                                show_titles=True, title_fmt=".2f",
                                truths=_truths9,
                                truth_color="firebrick",
                                quantiles=[0.16, 0.5, 0.84],
                                color="k",
                                label_kwargs={"fontsize": 18},
                                title_kwargs={"fontsize": 18})
                            _plt9.tight_layout()
                            _corner_path9 = os.path.join(
                                _mini_b7["plots_dir"],
                                "retrieval_night_0_corner.pdf")
                            _fig9.savefig(_corner_path9,
                                          bbox_inches="tight")
                            _plt9.close(_fig9)
                        return None, None

                # ==============================================================
                # Outer retrieval loop, Retrieval_choice != 6
                # (Master lines 6671-7139 / 7988-8125)
                # ==============================================================
                _base_dir_plot_b9 = (
                    _mini_b7["home_dir"].rstrip("/") + "/plots_SNR"
                    if _mini_b7["All_significance_metrics"]
                    else _mini_b7["plots_dir"].rstrip("/")
                )
                os.makedirs(_base_dir_plot_b9, exist_ok=True)

                if _mini_b7["Retrieval_choice"] != 6:
                    if not _mini_b7["Different_nights"]:
                        _ctx["idx_interval"] = np.arange(n_spectra)
                    _ctx["interval_tracker"] = None

                    for night_index in range(retrieved_nights):
                        _ctx["night_index"] = night_index

                        if _mini_b7["Different_nights"]:
                            n_spectra   = int(_n_spectra_store_b5[night_index])
                            phase       = np.asarray(
                                phase_store[night_index], dtype=float)
                            v_planet    = np.asarray(
                                v_planet_store[night_index], dtype=float)
                            with_signal = np.asarray(
                                with_signal_store[night_index], dtype=int)
                            without_signal = np.asarray(
                                without_signal_store[night_index], dtype=int)
                            fraction    = np.asarray(
                                fraction_store[night_index], dtype=float)
                            airmass     = np.asarray(
                                airmass_store[night_index], dtype=float)
                            berv        = np.asarray(
                                berv_store[night_index], dtype=float)
                            syn_jd      = np.asarray(
                                _syn_jd_store_b9[night_index, :n_spectra],
                                dtype=float)
                            T_0         = float(_T_0_store_b9[night_index])
                            _mat_res_ret   = np.asarray(
                                _mat_res_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _propag_ret    = np.asarray(
                                _propag_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _std_noise_ret = np.asarray(
                                _std_noise_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _mat_star_ret  = np.asarray(
                                _mat_star_ret_store[night_index][:n_spectra,:],
                                dtype=float)

                        if night_index == 0:
                            if cfg.retrieval.use_emulator:
                                from exoplore.retrieval.emulator import PrtEmulator
                                _ctx["prt_emulator"] = PrtEmulator.load(
                                    cfg.retrieval.emulator_path)
                                _ctx["atmosphere_ret_list"] = [None] * _n_orders_b9
                                print(f"  [Emulator] Loaded from {cfg.retrieval.emulator_path}")
                            else:
                                _ctx["prt_emulator"] = None
                                _ctx["atmosphere_ret_list"] = []
                                for hh in range(_n_orders_b9):
                                    if _Radtrans is not None:
                                        _atm9 = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_mini_b7["species_ret"][2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[
                                                    _mini_b7["order_selection"][hh],
                                                    :].min() - 0.01,
                                                wave_star[
                                                    _mini_b7["order_selection"][hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx["atmosphere_ret_list"].append(_atm9)

                        # 1D_alpha: precompute the fixed nominal model ONCE per
                        # order (no pRT inside the sampler; α just scales its
                        # depth).  Nominal abundance/T = the CCF template.
                        if (_mini_b7["Ret_dim"] in ("1D_alpha", "1D_alpha_Gibson22")
                                and _ctx.get("nominal_spec") is None
                                and _Radtrans is not None):
                            _nom_ab = np.asarray([
                                _mini_b7["vmr"][0], _mini_b7["vmr"][1],
                                float(cfg.atmosphere.ccf_template.mass_fractions[-1])])
                            _nom_T = float(
                                cfg.atmosphere.ccf_template.isothermal_temperature_K)
                            _ctx["nominal_wave"] = []
                            _ctx["nominal_spec"] = []
                            for hh in range(_n_orders_b9):
                                _wn, _sn, *_ = call_pRT(
                                    _mini_b7, p_ret,
                                    _ctx["atmosphere_ret_list"][hh],
                                    _mini_b7["species_ret"], _nom_ab,
                                    _mini_b7["MMW_ret"], _mini_b7["p0_ret"],
                                    _mini_b7["isothermal_ret"], _nom_T,
                                    _mini_b7["two_point_T_ret"],
                                    _mini_b7["p_points_ret"],
                                    _mini_b7["t_points_ret"],
                                    _mini_b7["Kappa_IR_ret"],
                                    _mini_b7["Gamma_ret"], _nom_T,
                                    None, None, use_easyCHEM=False)
                                _ctx["nominal_wave"].append(_wn)
                                _ctx["nominal_spec"].append(_sn)
                            print(f"  [1D_alpha] precomputed fixed nominal H2S "
                                  f"model for {_n_orders_b9} orders (α scales it "
                                  f"in-sampler; no pRT in the loop).")

                        _lbl_sfx = f"_night_{night_index}"
                        _run_and_save_b9(night_index, _base_dir_plot_b9,
                                         _lbl_sfx)

                # ==============================================================
                # Time-resolved loop, Retrieval_choice == 6
                # (Master lines 7141-7435)
                # ==============================================================
                else:
                    if not _mini_b7["Different_nights"]:
                        _ctx["idx_interval"] = np.arange(n_spectra)

                    for night_index in range(retrieved_nights):
                        _ctx["night_index"] = night_index

                        if _mini_b7["Different_nights"]:
                            n_spectra   = int(_n_spectra_store_b5[night_index])
                            phase       = np.asarray(
                                phase_store[night_index], dtype=float)
                            with_signal = np.asarray(
                                with_signal_store[night_index], dtype=int)
                            without_signal = np.asarray(
                                without_signal_store[night_index], dtype=int)
                            fraction    = np.asarray(
                                fraction_store[night_index], dtype=float)
                            airmass     = np.asarray(
                                airmass_store[night_index], dtype=float)
                            berv        = np.asarray(
                                berv_store[night_index], dtype=float)
                            syn_jd      = np.asarray(
                                _syn_jd_store_b9[night_index, :n_spectra],
                                dtype=float)
                            T_0         = float(_T_0_store_b9[night_index])
                            _mat_res_ret   = np.asarray(
                                _mat_res_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _propag_ret    = np.asarray(
                                _propag_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _std_noise_ret = np.asarray(
                                _std_noise_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _mat_star_ret  = np.asarray(
                                _mat_star_ret_store[night_index][:n_spectra,:],
                                dtype=float)

                        if night_index == 0:
                            if cfg.retrieval.use_emulator:
                                from exoplore.retrieval.emulator import PrtEmulator
                                _ctx["prt_emulator"] = PrtEmulator.load(
                                    cfg.retrieval.emulator_path)
                                _ctx["atmosphere_ret_list"] = [None] * _n_orders_b9
                                print(f"  [Emulator] Loaded from {cfg.retrieval.emulator_path}")
                            else:
                                _ctx["prt_emulator"] = None
                                _ctx["atmosphere_ret_list"] = []
                                for hh in range(_n_orders_b9):
                                    if _Radtrans is not None:
                                        _atm9 = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_mini_b7["species_ret"][2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[
                                                    _mini_b7["order_selection"][hh],
                                                    :].min() - 0.01,
                                                wave_star[
                                                    _mini_b7["order_selection"][hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx["atmosphere_ret_list"].append(_atm9)

                        # n_intervals
                        if _mini_b7.get("Retrieval_time_resolution_step"):
                            _n_int9 = int(len(with_signal)
                                          / _mini_b7[
                                              "Retrieval_time_resolution_step"])
                        else:
                            _n_int9 = 4  # default: ingress/first/second/egress

                        _tmg_means = {
                            k: np.full(_n_int9, np.nan) for k in parameters}
                        _tmg_erl   = {
                            k: np.full(_n_int9, np.nan) for k in parameters}
                        _tmg_erh   = {
                            k: np.full(_n_int9, np.nan) for k in parameters}

                        # ingress/egress indices
                        ingress_idx = np.asarray(
                            per_order_results[0].get("ingress_idx", []),
                            dtype=int)
                        egress_idx  = np.asarray(
                            per_order_results[0].get("egress_idx", []),
                            dtype=int)

                        for interval_tracker in range(_n_int9):
                            _ctx["interval_tracker"] = interval_tracker
                            if _mini_b7.get(
                                    "Retrieval_time_resolution_step"):
                                _step9 = _mini_b7[
                                    "Retrieval_time_resolution_step"]
                                _start9 = interval_tracker * _step9
                                _end9   = min(_start9 + _step9,
                                              len(with_signal))
                                _ctx["idx_interval"] = with_signal[
                                    _start9:_end9]
                            # else use default 4-interval split in loglike

                            _lbl_sfx2 = (f"_night_{night_index}"
                                         f"_interval_{interval_tracker}")
                            _sa9, _sa9_pts = _run_and_save_b9(
                                night_index, _base_dir_plot_b9, _lbl_sfx2)

                            if _sa9_pts is not None:
                                for _ki9, _pk9 in enumerate(parameters):
                                    _m9  = _sa9_pts["marginals"][_ki9]["median"]
                                    _l9, _h9 = (
                                        _sa9_pts["marginals"][_ki9]["1sigma"])
                                    _tmg_means[_pk9][interval_tracker] = _m9
                                    _tmg_erl[_pk9][interval_tracker]   = (
                                        _m9 - _l9)
                                    _tmg_erh[_pk9][interval_tracker]   = (
                                        _h9 - _m9)

                        # Save time-gradient results
                        _tg_save = {}
                        for _pk9 in parameters:
                            _tg_save[f"{_pk9}_mean"]     = _tmg_means[_pk9]
                            _tg_save[f"{_pk9}_err_low"]  = _tmg_erl[_pk9]
                            _tg_save[f"{_pk9}_err_high"] = _tmg_erh[_pk9]
                        np.savez_compressed(
                            f"{_base_dir_b9}/retrieval_timegradients"
                            f"_night_{night_index}_{_sfx9}",
                            **_tg_save)

                        # 4-panel errorbar time-gradient plot
                        if len(parameters) >= 4:
                            import matplotlib.pyplot as _plt9
                            _fig9_tg, _axes9_tg = _plt9.subplots(
                                4, 1, figsize=(8, 10), sharex=True)
                            for _ki9, _pk9 in enumerate(parameters[:4]):
                                _axes9_tg[_ki9].errorbar(
                                    np.arange(_n_int9),
                                    _tmg_means[_pk9],
                                    yerr=[_tmg_erl[_pk9], _tmg_erh[_pk9]],
                                    fmt="o-", capsize=4)
                                _axes9_tg[_ki9].set_ylabel(_pk9, fontsize=12)
                            _axes9_tg[-1].set_xlabel("Interval", fontsize=12)
                            _plt9.tight_layout()
                            _plt9.savefig(
                                f"{_base_dir_plot_b9}/{retrieval_name}"
                                f"_timegradients_night_{night_index}.pdf")
                            _plt9.show()
                            _plt9.close()

            # ------------------------------------------------------------------
            # 9j, 2D retrieval branch (lines 8454-8760)
            # ------------------------------------------------------------------
            elif _mini_b7["Ret_dim"] == "2D":
                if _mini_b7["Retrieval_choice"] != 4:
                    print("Retrieving one by one.")

                    def _loglike_2d(cube, ndim, nparams):
                        _night_index = _ctx["night_index"]
                        _sysrem_pass = _ctx["sysrem_pass"]
                        _atm_list_m  = _ctx["atmosphere_ret_list_morning"]
                        _atm_list_e  = _ctx["atmosphere_ret_list_evening"]
                        _atm_list    = _ctx["atmosphere_ret_list"]

                        log10_X_LL = cube[0]; log10_X_TL = cube[1]
                        K_p = cube[2]
                        T_equ_LL = cube[3]; T_equ_TL = cube[4]
                        v_wind = cube[5]

                        log_likelihood = 0.

                        inp_dat_ret = _mini_b7
                        abundances_morning = np.asarray(
                            [_mini_b7["vmr_morning_day"][0],
                             _mini_b7["vmr_morning_day"][1],
                             10.**log10_X_LL])
                        inp_dat_ret["vmr_morning_day"] = abundances_morning
                        abundances_evening = np.asarray(
                            [_mini_b7["vmr_evening_day"][0],
                             _mini_b7["vmr_evening_day"][1],
                             10.**log10_X_TL])
                        inp_dat_ret["vmr_evening_day"] = abundances_evening

                        model_mat = np.zeros(
                            (_n_ord_sel_b9, n_spectra, n_pixels))
                        model_mat_prepared = np.zeros_like(model_mat)

                        _same_sp = (
                            _mini_b7["species_ret_morning"][2:]
                            == _mini_b7["species_ret_evening"][2:])
                        if _same_sp:
                            _m_list = _atm_list
                            _e_list = _atm_list
                        else:
                            _m_list = _atm_list_m
                            _e_list = _atm_list_e

                        inp_dat_ret["T_eq_morning_day"] = T_equ_LL
                        inp_dat_ret["T_eq_evening_day"] = T_equ_TL

                        for hh in range(_n_orders_b9):
                            atm_m = _m_list[hh]
                            atm_e = _e_list[hh]
                            atm_m.setup_opa_structure(p_ret)
                            atm_e.setup_opa_structure(p_ret)

                            (wave_pRT_r, syn_spec_LL, _, syn_spec_TL,
                             *_) = call_pRT_limbs(
                                _mini_b7, p_ret, None, p_ret, None,
                                atm_m, None, atm_e, None,
                                mode="full", retrieval=True,
                                inp_dat_ret=inp_dat_ret)

                            syn_LL_w = convolve_spectrum_with_kernel(
                                wave_pRT_r, syn_spec_LL,
                                _kernel_wind_morning_store[hh],
                                _delta_v_windkernel[hh])
                            syn_TL_w = convolve_spectrum_with_kernel(
                                wave_pRT_r, syn_spec_TL,
                                _kernel_wind_evening_store[hh],
                                _delta_v_windkernel[hh])
                            syn_LL_r = convolve_spectrum_with_kernel(
                                wave_pRT_r, syn_LL_w,
                                _kernel_rot_morning_store[hh],
                                _delta_v_rotkernel[hh])
                            syn_TL_r = convolve_spectrum_with_kernel(
                                wave_pRT_r, syn_TL_w,
                                _kernel_rot_evening_store[hh],
                                _delta_v_rotkernel[hh])

                            v_planet_r = get_V(
                                K_p, phase, berv, _mini_b7["V_sys"], v_wind)

                            model_mat[hh], _ = spec_to_mat_fraction(
                                _mini_b7, syn_jd, T_0, v_planet_r,
                                wave_star[_mini_b7["order_selection"][hh], :],
                                wave_pRT_r,
                                per_order_results[hh]["syn_spec"],
                                np.ones((n_spectra, n_pixels)),
                                with_signal, without_signal, fraction,
                                spec_morning_day=syn_LL_r,
                                spec_evening_day=syn_TL_r,
                                sf_morning_day=per_order_results[hh].get(
                                    "sf_morning_day"),
                                sf_evening_day=per_order_results[hh].get(
                                    "sf_evening_day"),
                                include_star=False,
                            )
                            if _mini_b7["prepare_template"]:
                                if not _mini_b7["SYSREM_robust_halt"]:
                                    _sysrem_pass = None
                                model_mat_prepared[hh] = \
                                    preparing_pipeline(
                                        _mini_b7, model_mat[hh],
                                        _std_noise_ret[_night_index,
                                                       :n_spectra,
                                                       hh*_n_px_b9:(hh+1)*_n_px_b9],
                                        wave_star[
                                            _mini_b7["order_selection"][hh], :],
                                        np.where(
                                            useful_spectral_points_snr_aux[
                                                _night_index, hh, :])[0],
                                        np.where(
                                            mask_snr_ret_aux[
                                                _night_index, hh, :])[0],
                                        airmass, phase, without_signal,
                                        _sysrem_pass, None,
                                        tell_mask_threshold_Blain24=0.8,
                                        max_fit_BL19=False,
                                        sysrem_division=False, masks=False,
                                        correct_uncertainties=False,
                                        retrieval=True,
                                        mask_inter_retrieval=np.where(
                                            mask_inter_ret_aux[
                                                _night_index, hh, :])[0],
                                        useful_spectral_points_inter_retrieval=np.where(
                                            useful_spectral_points_inter_aux[
                                                _night_index, hh, :])[0],
                                    )
                            else:
                                model_mat_prepared[hh] = np.copy(model_mat[hh])

                        model_mat = np.transpose(model_mat, (1, 0, 2))
                        model_mat = model_mat.reshape(model_mat.shape[0], -1)
                        model_mat_prepared = np.transpose(
                            model_mat_prepared, (1, 0, 2))
                        model_mat_prepared = model_mat_prepared.reshape(
                            model_mat_prepared.shape[0], -1)

                        for n in with_signal:
                            log_likelihood += -0.5 * np.sum(
                                ((_mat_res_ret[_night_index, n,
                                   useful_spectral_points_ret[_night_index, :]]
                                  - model_mat_prepared[n,
                                   useful_spectral_points_ret[_night_index, :]])
                                 / _propag_ret[_night_index, n,
                                   useful_spectral_points_ret[_night_index, :]]
                                 )**2)
                        _ctx["sysrem_pass"] = _sysrem_pass
                        return log_likelihood

                    loglike = _loglike_2d

                    _base_dir_plot_b9 = (
                        _mini_b7["home_dir"].rstrip("/") + "/plots_SNR"
                        if _mini_b7["All_significance_metrics"]
                        else _mini_b7["plots_dir"].rstrip("/")
                    )
                    os.makedirs(_base_dir_plot_b9, exist_ok=True)

                    if not _mini_b7["Different_nights"]:
                        _ctx["idx_interval"] = np.arange(n_spectra)
                    _ctx["interval_tracker"] = None

                    for night_index in range(retrieved_nights):
                        _ctx["night_index"] = night_index

                        if _mini_b7["Different_nights"]:
                            n_spectra   = int(_n_spectra_store_b5[night_index])
                            phase       = np.asarray(
                                phase_store[night_index], dtype=float)
                            with_signal = np.asarray(
                                with_signal_store[night_index], dtype=int)
                            without_signal = np.asarray(
                                without_signal_store[night_index], dtype=int)
                            fraction    = np.asarray(
                                fraction_store[night_index], dtype=float)
                            airmass     = np.asarray(
                                airmass_store[night_index], dtype=float)
                            berv        = np.asarray(
                                berv_store[night_index], dtype=float)
                            syn_jd      = np.asarray(
                                _syn_jd_store_b9[night_index, :n_spectra],
                                dtype=float)
                            T_0         = float(_T_0_store_b9[night_index])
                            _mat_res_ret   = np.asarray(
                                _mat_res_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _propag_ret    = np.asarray(
                                _propag_ret_store[night_index][:n_spectra,:],
                                dtype=float)
                            _mat_star_ret  = np.asarray(
                                _mat_star_ret_store[night_index][:n_spectra,:],
                                dtype=float)

                        if night_index == 0:
                            _ctx["atmosphere_ret_list"]        = []
                            _ctx["atmosphere_ret_list_morning"] = []
                            _ctx["atmosphere_ret_list_evening"] = []
                            for hh in range(_n_orders_b9):
                                if _Radtrans is not None:
                                    _same_sp_b9 = (
                                        _mini_b7["species_ret_morning"][2:]
                                        != _mini_b7["species_ret_evening"][2:])
                                    if _same_sp_b9:
                                        if hh == 0:
                                            _ctx["atmosphere_ret_list"] = []
                                        _atm9m = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_mini_b7[
                                                "species_ret_morning"][2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh],
                                                    :].min() - 0.01,
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx[
                                            "atmosphere_ret_list_morning"
                                        ].append(_atm9m)
                                        _atm9e = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_mini_b7[
                                                "species_ret_evening"][2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh],
                                                    :].min() - 0.01,
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx[
                                            "atmosphere_ret_list_evening"
                                        ].append(_atm9e)
                                    else:
                                        if hh == 0:
                                            _ctx[
                                                "atmosphere_ret_list_morning"
                                            ] = []
                                            _ctx[
                                                "atmosphere_ret_list_evening"
                                            ] = []
                                        _atm9 = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_mini_b7[
                                                "species_ret_evening"][2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh],
                                                    :].min() - 0.01,
                                                wave_star[_mini_b7[
                                                    "order_selection"][hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx["atmosphere_ret_list"].append(
                                            _atm9)

                        if _pymn is not None:
                            _pymn.run(
                                loglike, prior,
                                n_dims=n_params,
                                outputfiles_basename=(
                                    f"{_base_dir_b9}/{retrieval_name}"
                                    f"_night_{night_index}_"),
                                resume=False, verbose=True,
                                evidence_tolerance=0.5,
                                sampling_efficiency=0.8,
                                n_iter_before_update=100,
                                const_efficiency_mode=_mini_b7[
                                    "Multinest_Constant_Eff_Mode"],
                                n_live_points=_mini_b7["Multinest_live_points"],
                                max_iter=0,
                            )
                            _sa2d = _pymn.Analyzer(
                                n_params=n_params,
                                outputfiles_basename=(
                                    f"{_base_dir_b9}/{retrieval_name}"
                                    f"_night_{night_index}_"))
                            _sa2d_pts = _sa2d.get_stats()
                            _json_b9.dump(
                                _sa2d_pts,
                                open(
                                    f"{_base_dir_b9}/{retrieval_name}"
                                    f"_night_{night_index}_stats.json", "w"),
                                indent=4)
                            # corner plot
                            if _corner is not None:
                                import matplotlib.pyplot as _plt9
                                _dat2d  = _sa2d.get_data()[:, 2:]
                                _wts2d  = _sa2d.get_data()[:, 0]
                                _msk2d  = _wts2d > 1e-4
                                _truths2d9 = (None
                                    if _mini_b7.get("use_real_data", False)
                                    else [
                                        np.log10(_mini_b7[
                                            "vmr_morning_day"][2]),
                                        np.log10(_mini_b7[
                                            "vmr_evening_day"][2]),
                                        _mini_b7["K_p"],
                                        _mini_b7["T_equ_morning_day"],
                                        _mini_b7["T_equ_evening_day"],
                                    ])
                                _fig2d  = _corner.corner(
                                    _dat2d[_msk2d, :],
                                    weights=_wts2d[_msk2d],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False, title_fmt=".2E",
                                    truths=_truths2d9,
                                    quantiles=[0.16, 0.5, 0.84],
                                    color="k", truth_color="firebrick",
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18})
                                for _ax2d in _fig2d.get_axes():
                                    _ax2d.tick_params(
                                        axis="both", labelsize=12)
                                _plt9.savefig(
                                    f"{_base_dir_plot_b9}/{retrieval_name}"
                                    f"_night_{night_index}_corner.pdf")
                                _plt9.show()
                                _plt9.close()
                            # save dat/weights/maskpoints
                            _dat2d  = _sa2d.get_data()[:, 2:]
                            _wts2d  = _sa2d.get_data()[:, 0]
                            _msk2d  = _wts2d > 1e-4
                            np.savez_compressed(
                                f"{_base_dir_b9}/{retrieval_name}"
                                f"_dat_{night_index}_{_sfx9}", a=_dat2d)
                            np.savez_compressed(
                                f"{_base_dir_b9}/{retrieval_name}"
                                f"_weights_{night_index}_{_sfx9}", a=_wts2d)
                            np.savez_compressed(
                                f"{_base_dir_b9}/{retrieval_name}"
                                f"_maskpoints_{night_index}_{_sfx9}",
                                a=_msk2d)

            _t_b9_end = _time.time()
            _t_blocks["Block 9, Bayesian retrieval"] = _t_b9_end - _t_b9_start
            print("\n  Block 9, Retrieval complete.")
            if cfg.timing:
                print(f"  [timing] Block 9 (retrieval only): "
                      f"{_t_blocks['Block 9, Bayesian retrieval']:.1f} s  "
                      f"({_t_blocks['Block 9, Bayesian retrieval']/60:.2f} min)")
            self._state["retrieval_base_dir"] = _base_dir_b9
        else:
            if cfg.cross_correlation.all_significance_metrics:
                print(
                    "  Block 9, Retrieval skipped "
                    "(retrieval.enabled=False).\n"
                    "  NOTE: all_significance_metrics=true produced S/N and "
                    "Welch t-test maps only.\n"
                    "  Set retrieval.enabled=true to also run Bayesian "
                    "retrieval (slow)."
                )
            else:
                print("  Block 9, Retrieval skipped "
                      "(Perform_retrieval=False, All_significance_metrics=False).")

        # ----------------------------------------------------------------
        # Final summary
        # ----------------------------------------------------------------
        if cfg.timing:
            _t_total = _time.time() - _t_run_start
            _t_blocks["TOTAL"] = _t_total
            print(f"\n  {'─'*50}")
            print(f"  [timing] Wall-clock breakdown:")
            for _blk, _dt in _t_blocks.items():
                if _blk != "TOTAL":
                    _pct = 100 * _dt / _t_total
                    print(f"    {_blk:<50s}  {_dt:7.1f} s  ({_pct:4.1f}%)")
            print(f"    {'TOTAL':<50s}  {_t_total:7.1f} s  (100.0%)")
            print(f"  {'─'*50}\n")
            # Write timing document to outputs
            import os as _os, datetime as _dt, locale as _lc
            try:
                _now = _dt.datetime.now()
                # Short stamp for filename: DD-MM-AAAA_HHMM
                _stamp_file = _now.strftime("%d-%m-%Y_%Hh%M")
                # Full European Spanish date for document header
                _MESES = ["enero","febrero","marzo","abril","mayo","junio",
                          "julio","agosto","septiembre","octubre","noviembre","diciembre"]
                _stamp_full = (f"{_now.day} de {_MESES[_now.month-1]} "
                               f"de {_now.year}, {_now.strftime('%H:%M:%S')}")
                _dirs = self._state.get("dirs", {})
                _root = str(_dirs.get("root", "."))
                _doc_path = _os.path.join(_root,
                                          f"timing_report_{_stamp_file}.txt")
                _os.makedirs(_root, exist_ok=True)
                with open(_doc_path, "w") as _tf:
                    _tf.write("EXoPLORE, informe de tiempos de simulación\n")
                    _tf.write(f"Ejecutado el {_stamp_full}\n")
                    _tf.write("=" * 60 + "\n\n")
                    _tf.write(f"Config:     {cfg.instrument.name}  "
                              f"{cfg.planet.name}  "
                              f"{cfg.observation.event_type}\n")
                    _tf.write(f"Orders:     {cfg.instrument.order_indices}\n")
                    _tf.write(f"Nights:     {cfg.observation.n_nights}\n")
                    _tf.write(f"Noiseless:  {cfg.observation.noiseless}\n")
                    _tf.write(f"Pipeline:   {cfg.pipeline.name}\n")
                    _tf.write(f"Retrieval:  {cfg.retrieval.enabled}  "
                              f"dim={cfg.retrieval.dimensionality}  "
                              f"sampler={cfg.retrieval.sampler}\n")
                    _tf.write(f"Emulator:   {cfg.retrieval.use_emulator}\n\n")
                    _tf.write("Block descriptions and timings\n")
                    _tf.write("-" * 60 + "\n")
                    _BLOCK_DESC = {
                        "Block 1, setup & instrument loading":
                            "Load instrument wavelength grid, SNR, blaze function, "
                            "stellar PHOENIX spectrum. Build output directories.",
                        "Block 2, event timing, phase, airmass, SNR":
                            "Compute Julian dates, orbital phase, barycentric "
                            "velocity, airmass, and per-exposure SNR for all nights.",
                        "Blocks 3-6, atm forward model, matrices, pipeline, CCF":
                            "Block 3: petitRADTRANS forward model (atmosphere + wind "
                            "+ rotation kernels). Block 4: stellar matrix + Doppler "
                            "shift + BATMAN transit fraction. Block 5: telluric "
                            "transmittances + noise injection. Block 6: preparing "
                            "pipeline (SYSREM / BL19 / Blain24) + CCF computation "
                            "per order per night.",
                        "Block 7, CCF statistics, Kp-Vsys maps":
                            "Assemble CCF across orders and nights. Compute "
                            "significance metrics (S/N, Welch t-test) and "
                            "Kp-Vsys detection maps.",
                        "Block 8, save matrices & CCF products":
                            "Write residual matrices, propagated noise, CCF "
                            "products, and pipeline masks to disk.",
                        "Block 9, Bayesian retrieval":
                            f"Nested sampling (MultiNest) or MCMC (emcee) retrieval. "
                            f"Dimensionality: {cfg.retrieval.dimensionality}. "
                            f"Sampler: {cfg.retrieval.sampler}. "
                            f"Emulator: {cfg.retrieval.use_emulator}.",
                    }
                    for _blk, _dt in _t_blocks.items():
                        if _blk == "TOTAL":
                            continue
                        _pct = 100 * _dt / _t_total
                        _tf.write(f"\n{_blk}\n")
                        _tf.write(f"  Time : {_dt:.1f} s  "
                                  f"({_dt/60:.2f} min)  "
                                  f"{_pct:.1f}% of total\n")
                        _desc = _BLOCK_DESC.get(_blk, "")
                        if _desc:
                            import textwrap as _tw
                            for _ln in _tw.wrap(_desc, 56):
                                _tf.write(f"  {_ln}\n")
                    _tf.write(f"\n{'─'*60}\n")
                    _tf.write(f"TOTAL  {_t_total:.1f} s  ({_t_total/60:.2f} min)\n")
                print(f"  [timing] Report saved: {_doc_path}")
            except Exception as _te:
                print(f"  [timing] Could not save report: {_te}")
        print("\n  === run() complete ===\n")
