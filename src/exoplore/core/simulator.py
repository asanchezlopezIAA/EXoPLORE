"""
exoplore.core.simulator
=======================

Simulation runner.

:class:`ExoploreSimulator` takes a :class:`~exoplore.config.SimulationConfig`
and runs the full simulation workflow.

Workflow map
------------
The numbered blocks below describe the standard molecular-spectroscopy route.
Direct sodium, helium simulation, observed-data recipes and helium grids dispatch to their
own runners at the beginning of :meth:`ExoploreSimulator.run`.

1. Resolve inputs, planet parameters, instrument grids and stellar spectrum.
2. Build exposure times, orbital phases, airmass and time-dependent inputs.
3–6. For each selected order: atmosphere -> observation -> preparation -> CCF.
7. Combine CCF information and construct orbital-velocity detection maps.
8. Save the products selected by the output configuration.
9. Run atmospheric retrieval when enabled.

Blocks 3–6 share an order loop. They are not four complete passes over the
dataset. The developer architecture guide in ``docs/architecture.md`` maps
these stages to the modules that perform each calculation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from exoplore.config.models import SimulationConfig
from exoplore.io.paths import build_output_tree
from exoplore.results import CCFResult, DetectionMapResult

# ---------------------------------------------------------------------------
# Scientific dependencies, grouped by the stage that owns the calculation.
# This module coordinates them. numerical implementations belong in their
# domain packages. Keep import order because some packages initialize state
# at import time.
# ---------------------------------------------------------------------------
from exoplore.atmosphere.prt import call_pRT, call_pRT_limbs, convolve
from exoplore.atmosphere.models import RadiativeTransferInputs
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
from exoplore.ccf.statistics import Welch_ttest_map, statistical_study
from exoplore.ccf.models import NightStudyInputs
from exoplore.instruments.wavegrid import (
    From1OrderTo1Detector,
    get_WaveGrid,
    pixel_snr_one_order,
)
from exoplore.io.stellar import LoadPhoenix, get_stellar_matrix
from exoplore.core.observation import spec_to_mat_fraction
from exoplore.io.utils import find_nearest, format_number, save_compressed
from exoplore.observation.airmass import PWV_handling, get_airmass
from exoplore.observation.noise import add_throughput
from exoplore.observation.randomness import apply_configured_throughput, gaussian_noise_seed
from exoplore.observation.timing import (
    block_parameter,
    dayside_fraction,
    find_nights_with_extrema,
    get_event,
)
from exoplore.observation.velocity import get_V, get_V_eccentric, stellar_radial_velocity
from exoplore.pipelines.masking import Correct_NaN, Remove_Outliers, merge_masks, _merge_masks
from exoplore.pipelines.prepare import injection
from exoplore.pipelines.runner import prepare_observation
from exoplore.pipelines.models import PreparationInputs
from exoplore.pipelines.sysrem import (
    SYSREM_filtering_projector,
    SYSREM_filtering_projector_singleorder,
    filter_model_singleorder,
    get_SYSREM_its_ordbyord,
)
from exoplore.pipelines.tellurics import Load_Telluric_Transmittances, validate_telluric_file
from exoplore.plotting.kpvsys import plot_1D_CCF, plot_Kp_Vrest
from exoplore.plotting.models import DetectionPlotSettings
from exoplore.retrieval.models import LimbRetrievalState
from exoplore.plotting.matrices import CCF_matrix_ERF
from exoplore.analysis.stats import plot_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_throughput(flux, observation, order_index, night_index):
    """Apply configured throughput using this module's add_throughput function."""
    # Pass this module's add_throughput binding to the order/night seed helper.
    return apply_configured_throughput(
        flux, observation, order_index, night_index,
        throughput_function=add_throughput,
    )


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
    if pipe.name not in {"ASL19", "Gibson22", "Cheverall26", "Nortmann26"}:
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
        # distinct output folders.
        pca_flag = f"N{pipe.sysrem_iterations}_"
        kp_flag = opt_flag = ""

    sf_str = _format_number(obs.noise_scaling_factor)
    # Keep the fitted-beta retrieval in its own directory so its posterior
    # files can be compared with the fixed-beta Blain24 run.
    pipeline_label = pipe.name
    if cfg.retrieval.enabled and cfg.retrieval.log_likelihood == "Blain24_beta":
        pipeline_label = ("Blain24_beta" if pipe.name == "Blain24"
                          else f"{pipe.name}_Blain24_beta")
    # Stellar-effect experiments get their own output directory. Default
    # configurations retain the established simulation names.
    stellar_flag = (f"_stellar_{cfg.rm_clv.effects}"
                    f"_sim{int(cfg.rm_clv.simulate)}_cor{int(cfg.rm_clv.correct)}"
                    if cfg.rm_clv.active else "")
    if cfg.rm_clv.active and cfg.rm_clv.coupling == "spatial":
        stellar_flag += "_spatial"
    if cfg.rm_clv.correct and cfg.rm_clv.correction_method == "factorized":
        stellar_flag += "_correction_factorized"

    return (
        f"{pipeline_label}_{signal_flag}_"
        f"{pca_flag}{kp_flag}{opt_flag}"
        f"{obs.n_nights}nights_"
        f"{signif_flag}_{stack_flag}_{real_flag}_"
        f"{noise_flag}_"
        f"stdnoisex{sf_str}{stellar_flag}"
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
    helium_backend: Optional[str] = None
    helium_grid_models: Optional[int] = None
    helium_observed: bool = False
    atomic_transmission: bool = False
    stellar_effects: Optional[str] = None
    stellar_simulate: bool = False
    stellar_correct: bool = False
    stellar_coupling: str = "factorized"

    def __post_init__(self):
        # Expose the configured planet name in the summary.
        object.__setattr__(self, "planet", self.planet_name)

    # Pipeline citation strings
    _PIPELINE_REFS = {
        "czesla2024":   "czesla2024    (Czesla et al. 2024, A&A, 692, A230)",
        "BL19":         "BL19          (Brogi & Line 2019, AJ, 157, 114)",
        "Blain24":      "Blain24       (Blain, Sanchez-Lopez & Molliere 2024, AJ, 167, 179)",
        "ASL19":        "ASL19         (Sanchez-Lopez et al. 2019, A&A, 630, A53)",
        "Gibson22":     "Gibson22      (Gibson et al. 2022, MNRAS, 512, 4618)",
        "Cheverall26":  "Cheverall26   (Cheverall et al. 2026, MNRAS, IGRINS pipeline)",
        "Nortmann26":   "Nortmann26    (Nortmann et al. 2026, A&A, CRIRES+ division-SYSREM pipeline)",
    }
    _SYSREM_PIPELINES = {"ASL19", "Gibson22", "Cheverall26", "Nortmann26"}

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
        if self.atomic_transmission:
            lines.append("  Spectroscopy        : Direct sodium transmission")
        if self.stellar_effects is not None:
            lines += [f"  Stellar effects     : {self.stellar_effects}",
                      f"  Stellar coupling    : {self.stellar_coupling}",
                      f"  Simulate RM/CLV     : {self.stellar_simulate}",
                      f"  Correct RM/CLV      : {self.stellar_correct}"]
        if self.helium_grid_models is not None:
            lines += ['  Analysis            : Sunbather He I grid fit',
                      f'  Grid models         : {self.helium_grid_models}',
                      f'  Output root         : {self.output_root}', '']
            return '\n'.join(lines)
        if self.helium_observed:
            lines += ['  Analysis            : observed He I transmission',
                      f'  Output root         : {self.output_root}', '']
            return '\n'.join(lines)
        if self.helium_backend is not None:
            lines += [f"  Atmosphere          : {self.helium_backend}",
                      "  Observations        : synthetic optical transit",
                      "  Analysis            : direct He I transmission",
                      f"  Output root         : {self.output_root}", ""]
            return "\n".join(lines)
        if self.pipeline == "czesla2024":
            lines += ["  Analysis            : direct He I transmission", "  Calibration         : validated per-exposure CRIRES+ corrections", f"  Output root         : {self.output_root}", ""]
            return "\n".join(lines)
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
        # Publish access only after a successful standard run. Keep its context
        # separate from a config the caller might edit for the next experiment.
        self._completed_run = None
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        cfg = self.config
        from exoplore.observation.carmenes_noise import configured_carmenes_etc
        configured_carmenes_etc(cfg)
        if cfg.pipeline.name == 'Seidel22':
            from exoplore.pipelines.seidel22 import validate_run
            validate_run(cfg)
        from exoplore.core.spectroscopy_mode import spectroscopy_route
        route = spectroscopy_route(cfg)
        # Check the stellar-effect route before any specialized runner returns.
        # Stop early when the selected runner cannot use the requested stellar effect.
        from exoplore.stellar.observation import validate_route
        validate_route(cfg, route)
        # The direct sodium runner checks its own instrument, atmosphere, ETC and
        # Molecfit inputs before any spectra are generated.
        if route == 'atomic_simulation':
            from exoplore.core.atomic_simulation import validate_atomic_run
            validate_atomic_run(cfg)
            return
        if route == 'helium_grid':
            from exoplore.retrieval.helium_grid import validate_grid_config
            validate_grid_config(cfg)
            return
        if route == "helium_simulation":
            from exoplore.core.helium_simulation import validate_helium_run
            validate_helium_run(cfg)
            return
        if route == "helium_observed":
            if cfg.pipeline.name == 'Palle2020':
                from exoplore.pipelines.palle2020 import validate_run_config
                validate_run_config(cfg)
                return
            from exoplore.pipelines.czesla2024 import validate_run_config
            validate_run_config(cfg)

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
        from exoplore.core.spectroscopy_mode import spectroscopy_route
        route = spectroscopy_route(cfg)
        grid_models=None
        if route=='helium_grid':
            from exoplore.config.helium_grid import axis_values
            grid_models=(len(axis_values(cfg.helium_grid.temperature_grid_K,'temperature_grid_K',True))
                         *len(axis_values(cfg.helium_grid.log10_mass_loss_grid_g_s,'log10_mass_loss_grid_g_s')))
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
            helium_backend=((cfg.atmosphere.helium.backend if cfg.observation.simulate_planet else "no injection")
                            if route == 'helium_simulation' else None),
            helium_grid_models=grid_models,
            helium_observed=(route == 'helium_observed'),
            atomic_transmission=(route == 'atomic_simulation'),
            stellar_effects=cfg.rm_clv.effects if cfg.rm_clv.active else None,
            stellar_simulate=cfg.rm_clv.simulate,
            stellar_correct=cfg.rm_clv.correct,
            stellar_coupling=cfg.rm_clv.coupling,
        )

    # ------------------------------------------------------------------
    # Read completed CCF and detection-map results
    # ------------------------------------------------------------------

    def get_combined_ccf(self, *, night_index: int | None = None, copy: bool = True) -> CCFResult:
        """Return a named CCFResult after a successful standard run.

        Arrays retain axes (Earth-frame velocity, exposure). By default they
        are detached copies. Copy=False returns read-only views. Different
        observing sequences require a night_index. Shared-geometry nights have
        already been coadded. Injection/iteration cubes require separate
        selection and are not silently collapsed by this accessor.
        """
        from exoplore.io.results import _combined_ccf_from_state
        state, orders, metadata = self._completed_result_context()
        return _combined_ccf_from_state(
            state, orders, metadata, night_index=night_index, copy=copy,
        )

    def get_detection_map(self, *, metric: str = "snr", night_index: int | None = None) -> DetectionMapResult:
        """Load the explicitly saved S/N or Welch map as DetectionMapResult.

        metric selects the saved S/N or Welch archive. The ccf_tot_sig working
        array holds the last calculated statistic in an all-metrics run.
        Missing archives raise FileNotFoundError.
        """
        from exoplore.io.results import load_detection_map
        if metric not in ("snr", "welch"):
            raise ValueError("Detection-map metric must be 'snr' or 'welch'.")
        state, _, metadata = self._completed_result_context()
        # Require a CCF from this run before reading its saved detection map.
        if state.get("ccf_complete") is None:
            raise RuntimeError("This run has no CCF products; an older saved map cannot be returned as its result.")
        suffix = ""
        if night_index is not None:
            if (isinstance(night_index, bool)
                    or not isinstance(night_index, (int, np.integer))
                    or not 0 <= night_index < metadata["n_nights"]):
                raise ValueError("night_index is outside the available nights.")
            suffix = f"_night{night_index}"
        stem = "ccf_tot_sn_map" if metric == "snr" else "ccf_tot_welch_map"
        path = Path(state["dirs"]["matrices"]) / f"{stem}_{state['sim_name']}{suffix}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"No saved {metric} map at {path}. This route/metric/night may "
                "not produce that archive; no replacement calculation was run."
            )
        return load_detection_map(path, metric=metric)

    def _completed_result_context(self):
        """Return the state, order selection and metadata of the completed run."""
        if self._completed_run is None:
            raise RuntimeError(
                "No successful standard run is available. Run the simulator first; "
                "helium workflows do not yet expose these result accessors."
            )
        return self._completed_run

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
    # run(): route dispatch and complete standard molecular workflow
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full simulation.

        Dispatch to the configured spectroscopy route. The standard route
        performs setup, per-order simulation/preparation, cross-correlation,
        detection analysis and optional retrieval, as configured.

        Returns ``None``. persistent products are written under the configured
        output tree. ``_state`` holds intermediate arrays. Output flags select
        which arrays are written to disk.
        """

        # ----------------------------------------------------------------
        # Block 1a, setup, simulation name, output directories
        # ----------------------------------------------------------------
        import time as _time
        # Results become available when this attempt completes successfully.
        self._completed_run = None
        cfg = self.config
        from exoplore.observation.carmenes_noise import configured_carmenes_etc
        _carmenes_etc = configured_carmenes_etc(cfg)
        if cfg.pipeline.name == 'Seidel22':
            from exoplore.pipelines.seidel22 import validate_run
            validate_run(cfg)
        from exoplore.core.spectroscopy_mode import spectroscopy_route
        # Select the runner for the configured molecular, sodium or helium experiment.
        route = spectroscopy_route(cfg)
        if cfg.rm_clv.active:
            # Users can edit a config after previewing it. Recheck stellar
            # compatibility at execution so unsupported changes stop early.
            from exoplore.stellar.observation import validate_route
            validate_route(cfg, route)
        # Run the integrated sodium observation and transmission analysis. Its own
        # runner writes the two line diagnostics and returns after reduction.
        if route == 'atomic_simulation':
            from exoplore.core.atomic_simulation import run_atomic_simulation
            run_atomic_simulation(cfg)
            return
        if route=='helium_grid':
            from exoplore.retrieval.helium_grid import run_helium_grid
            run_helium_grid(cfg)
            return
        if route == "helium_simulation":
            from exoplore.core.helium_simulation import run_helium_simulation
            run_helium_simulation(cfg)
            return
        if route == "helium_observed":
            if cfg.pipeline.name == 'Palle2020':
                from exoplore.pipelines.palle2020 import run_palle2020
                run_palle2020(cfg)
                return
            from exoplore.pipelines.czesla2024 import run_czesla2024
            run_czesla2024(cfg)
            return
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
        from exoplore.instruments import load_instrument, get_WaveGrid
        inst = load_instrument(cfg)
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
        _etc_reader_options = {'use_etc': True} if _carmenes_etc is not None else {}
        _wg = get_WaveGrid(cfg.instrument, cfg.observation, inst, n_orders_total,
                           **_etc_reader_options)
        _wg = _wg + (None,) * (7 - len(_wg))   # normalise to 7-tuple
        wave_star, n_pixels, sig_og, snr_og, JD_og, airmass_og, wave_mid_og = _wg

        # Validate the reference against SELECTED orders before loading a star
        # or calculating atmospheric spectra. Observed data do not consume
        # synthetic tellurics unless a telluric CCF template is explicitly used.
        if ((cfg.tellurics.include_tellurics
             and not cfg.observation.use_real_data
             and not cfg.tellurics.use_full_skycalc)
                or cfg.cross_correlation.ccf_template_source == "telluric"):
            _reference_path = (
                cfg.tellurics.reference_telluric_file
                or (f"{cfg.paths.inputs_dir}tellurics/"
                    f"tell_ref_airmass_{cfg.tellurics.reference_airmass:.1f}.fits")
            )
            # CARMENES readers return (pixel, order) until Block 2 transposes
            # them. Other standard readers return (order, pixel) at this stage.
            _selected_wave = (wave_star[:, _order_sel_b1]
                              if cfg.instrument.name in ("CARMENES_NIR", "CARMENES_VIS")
                              else wave_star[_order_sel_b1, :])
            validate_telluric_file(_reference_path, _selected_wave)

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
        # An enabled surface calculation supplies its own disk-integrated
        # stellar reference. Existing runs keep their PHOENIX loading path.
        _stellar_surface = None
        _correction_surface = None
        # Choose the local ray integration or the separate photospheric-factor calculation.
        _spatial_rm_clv = cfg.rm_clv.active and cfg.rm_clv.coupling == "spatial"
        if cfg.rm_clv.active:
            from exoplore.stellar.observation import TransitStellarModel
            if _spatial_rm_clv:
                from exoplore.stellar.coupled import SpatialTransitModel
                # Spatial mode combines local stellar light with atmospheric rays from pRT.
                _stellar_surface = SpatialTransitModel(cfg.rm_clv, planet, inst.res, surface_config=cfg.stellar_surface)
            else:
                # Factorized mode calculates the stellar line distortion as a separate multiplier.
                _stellar_surface = TransitStellarModel(cfg.rm_clv, planet, inst.res, surface_config=cfg.stellar_surface)
            # A separate factorized correction can be applied to a spatial injection
            # through Seidel22, optionally using another stellar synthesis.
            if cfg.rm_clv.correct and cfg.rm_clv.correction_method == "factorized":
                if cfg.pipeline.name != 'Seidel22':
                    raise ValueError("The separate factorized correction requires Seidel22 preparation")
                # Construct that correction model with its own selected stellar input.
                correction_choices = replace(cfg.rm_clv, coupling="factorized",
                    model_file=cfg.rm_clv.correction_model_file or cfg.rm_clv.model_file)
                _correction_surface = TransitStellarModel(correction_choices, planet, inst.res)
            if _stellar_surface.field is not None:
                # Record each input hash and the selected geometry beside the spectra.
                import json as _surface_json
                (Path(dirs['root'])/'stellar_surface_provenance.json').write_text(
                    _surface_json.dumps(_stellar_surface.field.provenance(), indent=2))
            # Check the selected orders before loading atmospheric opacities.
            # Exposure-specific Doppler and convolution margins are checked
            # again when the observing-frame model is evaluated in Block 4.
            _model_wave = (wave_star[:, _order_sel_b1]
                           if cfg.instrument.name in ("CARMENES_NIR", "CARMENES_VIS")
                           else wave_star[_order_sel_b1, :])
            _local_wave = _stellar_surface.surface.local.wavelength_um
            if (_model_wave.min() < _local_wave[0]
                    or _model_wave.max() > _local_wave[-1]):
                raise ValueError(
                    "RM/CLV intensity model must cover the selected instrument orders "
                    f"({_model_wave.min():.6f} to {_model_wave.max():.6f} microns)")
            print(f"  Stellar effects : {cfg.rm_clv.effects}, "
                  f"simulate={cfg.rm_clv.simulate}, correct={cfg.rm_clv.correct}")
        if _phoenix is not None and not cfg.rm_clv.active:
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
            stellar_surface=_stellar_surface,
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
        _accurate_nights = False   # True when synthetic nights sit at real
                                   # observable epochs (Block 2a below)
        if (cfg.observation.different_nights
                and not cfg.observation.specific_event
                and JD_og is None):
            _jd_step = (cfg.observation.exposure_time_seconds
                        + cfg.observation.overhead_time_seconds
                        + cfg.observation.readout_time_seconds) / 86400.0
            _td_d = planet.transit_duration_hours / 24.0
            _pre_d = _pre_h / 24.0
            _post_d = _post_h / 24.0
            # Night placement: with use_accurate_airmass, each night is a
            # real observable transit from the site (deterministic search
            # from the reference epoch), so airmass and BERV genuinely
            # differ between nights.  Otherwise consecutive orbits at
            # T0 + n * P (nights differ only through the config inputs).
            _tmid_list = None
            if (cfg.tellurics.use_accurate_airmass
                    and planet.ra_deg is not None
                    and planet.dec_deg is not None):
                from exoplore.observation.ephemeris import (
                    find_observable_transit_epochs)
                try:
                    _tmid_list = find_observable_transit_epochs(
                        planet.transit_epoch_bjd,
                        planet.orbital_period_days,
                        planet.ra_deg, planet.dec_deg,
                        cfg.instrument.observatory,
                        cfg.observation.n_nights,
                    )
                except Exception as _eph_err:
                    print(f"  WARNING: observable-transit search failed "
                          f"({_eph_err}); falling back to T0 + n * P.")
            elif cfg.tellurics.use_accurate_airmass:
                print("  WARNING: use_accurate_airmass=True but the planet "
                      "file lacks ra_deg/dec_deg; falling back to T0 + n * P "
                      "and the parabolic airmass model.")
            JD_og = []
            for _n in range(cfg.observation.n_nights):
                _tmid_n = (float(_tmid_list[_n])
                           if _tmid_list is not None
                           else planet.transit_epoch_bjd
                           + _n * planet.orbital_period_days)
                _jd_n = np.arange(
                    _tmid_n - _td_d / 2.0 - _pre_d,
                    _tmid_n + _td_d / 2.0 + _post_d + _jd_step,
                    _jd_step,
                )
                JD_og.append(_jd_n)
            _accurate_nights = _tmid_list is not None
            if _accurate_nights:
                from astropy.time import Time as _Time
                _iso = [_Time(t, format="jd").iso[:10] for t in _tmid_list]
                print(f"  Different nights at observable transit epochs: "
                      f"{', '.join(_iso)}")

        from exoplore.observation import get_event
        syn_jd, with_signal, without_signal, transit_mid_JD = \
            get_event(cfg.observation, planet, JD_og)

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
                _amf_n = (f"{cfg.paths.inputs_dir}"
                          f"Skycalc_{cfg.observation.flag_event}/"
                          f"night_{_n}/Fixed_PWV/airmass.fits")
                if not cfg.tellurics.include_tellurics:
                    _new_airmass.append(None)
                elif (cfg.tellurics.use_full_skycalc
                        and os.path.exists(_amf_n)):
                    from astropy.io import fits as _fits_amn
                    _arr_n = _fits_amn.open(_amf_n)[0].data.astype(float)
                    if len(_arr_n) != len(_jd_n):
                        raise ValueError(
                            f"night {_n}: airmass.fits has {len(_arr_n)} "
                            f"entries but this night's cadence gives "
                            f"{len(_jd_n)} exposures. Regenerate the SkyCalc "
                            f"inputs for the per-night exposure times.")
                    _new_airmass.append(_arr_n)
                elif _accurate_nights:
                    from exoplore.observation.ephemeris import (
                        accurate_airmass as _acc_am_pn)
                    _new_airmass.append(
                        _acc_am_pn(_jd_n, planet.ra_deg, planet.dec_deg,
                                   cfg.instrument.observatory))
                else:
                    _new_airmass.append(
                        get_airmass(cfg.tellurics.airmass_evolution, _jd_n,
                                    _am_limits_n))
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
            # embedded in the per-night SkyCalc files and is not needed here.
            # use a list of None as a safe placeholder.
            if airmass_og is None:
                # Synthetic different_nights: no reference airmass files.
                # The per-exposure airmass is a physical quantity the
                # preparation requires (Blain24 fits log-flux against airmass),
                # so it is always computed on each night's own JD grid,
                # regardless of the telluric mode. It must never be left as
                # None (which np.asarray later collapses to a 0-d scalar).
                # Priority: (1) a per-night airmass.fits stored alongside the
                # SkyCalc files (written by generate_skycalc_inputs.py), which
                # is the airmass the telluric spectra were actually generated
                # with, so pipelines that regress against airmass (Blain24)
                # see a regressor consistent with the applied tellurics.
                # (2) with accurate nights, the actual sec z from the site.
                # (3) the parabolic model with the configured limits.
                _am_files = []
                if cfg.tellurics.use_full_skycalc:
                    from astropy.io import fits as _fits_am
                    for n in range(cfg.observation.n_nights):
                        _amf = (f"{cfg.paths.inputs_dir}"
                                f"Skycalc_{cfg.observation.flag_event}/"
                                f"night_{n}/Fixed_PWV/airmass.fits")
                        if os.path.exists(_amf):
                            _arr = _fits_am.open(_amf)[0].data.astype(float)
                            if len(_arr) != len(syn_jd[n]):
                                raise ValueError(
                                    f"night {n}: airmass.fits has "
                                    f"{len(_arr)} entries but this night "
                                    f"has {len(syn_jd[n])} exposures. "
                                    f"Regenerate the SkyCalc inputs to "
                                    f"match the observation cadence.")
                            _am_files.append(_arr)
                if len(_am_files) == cfg.observation.n_nights:
                    print("  Per-night airmass read from the SkyCalc "
                          "input directories (matches the telluric files).")
                    airmass = _am_files
                elif _accurate_nights:
                    from exoplore.observation.ephemeris import (
                        accurate_airmass as _acc_am)
                    airmass = [
                        _acc_am(syn_jd[n], planet.ra_deg, planet.dec_deg,
                                cfg.instrument.observatory)
                        for n in range(cfg.observation.n_nights)
                    ]
                else:
                    _amlpn = cfg.tellurics.airmass_limits_per_night
                    _aml_default = (cfg.tellurics.airmass_limits
                                    if cfg.tellurics.airmass_limits is not None
                                    else [1.2, 1.6])
                    airmass = [
                        get_airmass(
                            cfg.tellurics.airmass_evolution,
                            syn_jd[n],
                            (_amlpn[n] if _amlpn is not None
                             and n < len(_amlpn)
                             else _aml_default),
                        )
                        for n in range(cfg.observation.n_nights)
                    ]
            else:
                airmass = airmass_og

        else:  # specific_event, single night
            phase = (syn_jd - cfg.observation.specific_T0_bjd) / planet.orbital_period_days
            airmass = airmass_og

        # Per-night BERV for fully synthetic different_nights.  Each night
        # sits at its own epoch, so the barycentric correction differs from
        # night to night. stellar and planetary lines then shift against the
        # telluric rest frame between nights, as they do between real epochs.
        # Consumed by Blocks 4a (stellar matrix) and 4b (planet velocities).
        _berv_pn = None
        if (cfg.observation.different_nights
                and not cfg.observation.use_real_data
                and not cfg.observation.specific_event):
            if (cfg.observation.use_accurate_berv
                    and planet.ra_deg is not None
                    and planet.dec_deg is not None):
                from exoplore.observation.ephemeris import (
                    accurate_berv as _acc_berv)
                _berv_pn = [
                    _acc_berv(syn_jd[n], planet.ra_deg, planet.dec_deg,
                              cfg.instrument.observatory)
                    for n in range(cfg.observation.n_nights)
                ]
                print("  Per-night BERV (mean): "
                      + ", ".join(f"{np.mean(b):+.2f} km/s"
                                  for b in _berv_pn))
                if not _accurate_nights:
                    print("  NOTE: nights sit at consecutive orbits "
                          "(T0 + n * P), so per-night BERVs differ by "
                          "less than they would between real epochs.")
            elif cfg.observation.use_accurate_berv:
                print("  WARNING: use_accurate_berv=True but the planet "
                      "file lacks ra_deg/dec_deg; using constant berv_kms "
                      "for all nights.")

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
        if _carmenes_etc is not None:
            # The ETC S/N is evaluated for the actual exposures in Block 5d.
            snr_all = np.ones_like(wave_star, dtype=float)
        elif cfg.noise.fixed_snr is None:
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
                    # For different_nights n_spectra is an array. Use its max
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

        if _carmenes_etc is not None:
            # Check the full cadence before paying for any pRT or stellar models.
            _etc_airmass = (np.concatenate([np.asarray(x) for x in airmass])
                            if cfg.observation.different_nights else airmass)
            _carmenes_etc.check_airmass(_etc_airmass)

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

        # The named limb regions supply the atmospheric spectra and their
        # chemistry. Their system quantities are shared with trial models.
        _md = cfg.atmosphere.morning_day
        _ed = cfg.atmosphere.evening_day
        # These quantities are shared by the injected spectrum and trial models.
        rt_inputs = RadiativeTransferInputs(
            event_type=cfg.observation.event_type,
            planet_mass_g=planet.M_pl,
            planet_radius_cm=planet.R_pl,
            stellar_radius_cm=planet.R_star,
            stellar_teff_K=planet.stellar_teff_K,
            resolving_power=inst.res,
            internal_temperature_K=planet.t_int_K,
            gravity_cgs=None,
            convolve_to_resolution=cfg.instrument.convolve_to_resolution,
        )

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
        # Allocated at h==0, b==0 inside the per-order loop. declared
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

        # Collect fully masked orders for the run's combined warning report.
        _fully_masked_orders: list = []

        # Per-frame wavelength solution (solution-only secondary wavecal):
        # the flux is native (never resampled). The MODEL is interpolated onto
        # each exposure's grid, following Line+2021 / Smith+2024.  Loaded once.
        # absent => the common grid is used (identical to <0.2 px).
        _wave_pf_all = None
        if cfg.observation.use_real_data:
            from astropy.io import fits as _fits_pf
            _pf_file = (f"{cfg.paths.inputs_dir}reference_night/"
                        f"wave_perframe_0.fits")
            if os.path.exists(_pf_file):
                _wave_pf_all = _fits_pf.open(_pf_file)[0].data.astype(np.float64)
                print("  [wavecal] per-frame wavelength solution loaded "
                      "(model interpolated per exposure; flux native).")

        # Theoretical telluric transmittance per order (molecfit), for the
        # Nortmann26 pipeline's 96% normalisation mask and 20% deep mask
        # (Nortmann 2024/2026 Sec 3.1).  On the reference grid, one row per
        # order. absent => the branch falls back to a data-driven deep mask.
        _telluric_model = None
        if cfg.observation.use_real_data and cfg.pipeline.name == "Nortmann26":
            from astropy.io import fits as _fits_tt
            _tt_file = (f"{cfg.paths.inputs_dir}reference_night/"
                        f"telluric_model_0.fits")
            if os.path.exists(_tt_file):
                _telluric_model = _fits_tt.open(_tt_file)[0].data.astype(np.float64)
                print("  [Nortmann26] molecfit telluric model loaded "
                      "(96% normalisation mask + 20% deep mask).")

        # Follow one selected wavelength order through atmosphere -> observed
        # time series -> preparation -> CCF before advancing to the next order.
        # h indexes our selected list. order_selection[h] is the instrument ID.
        # b in the inner observation loop indexes nights, not spectral orders.
        for h in range(n_orders):
            # ----------------------------------------------------------
            # 3c, per-order wavelength and SNR setup
            # ----------------------------------------------------------
            wave_ins = wave_star[order_selection[h], :].astype(np.float64)
            # per-exposure wavelength grid for this order (or None => common)
            _wave_pf_ord = (_wave_pf_all[:, order_selection[h], :]
                            if _wave_pf_all is not None else None)
            if spec_star_ins is not None:
                spec_star_phoenix = spec_star_ins[order_selection[h], :]
            else:
                spec_star_phoenix = None

            # Wavelength bounds wider than array to avoid edge effects
            wvl_min = wave_ins[0]  - 1.0e-3
            wvl_max = wave_ins[-1] + 1.0e-3

            # Per-order SNR and signal arrays
            if _carmenes_etc is not None:
                # Fill these arrays in Block 5d, when the current night's
                # telluric spectrum and airmass are available.
                snr = ([np.ones((int(n_spectra[n]), n_pixels))
                        for n in range(cfg.observation.n_nights)]
                       if cfg.observation.different_nights
                       else np.ones((n_spectra, n_pixels)))
                sig = ([np.zeros_like(row) for row in snr]
                       if cfg.observation.different_nights else None)
            elif not _is_andes_any:
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
                # For different_nights: init empty lists. Block 5a appends
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
            # This is the injected planet, described by atmosphere.planet_model
            # or its limb regions. The search template is built separately.
            # its composition need not match this simulated atmospheric truth.
            _atmospheric_chords = None
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
                        rt_inputs, p, atmosphere,
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
                    if _spatial_rm_clv:
                        from exoplore.atmosphere.transmission import calculate_chord_transmission
                        from petitRADTRANS import physical_constants as _cst_chord
                        # Use the exact temperature, chemistry and pressure
                        # structure just calculated by call_pRT. Additional
                        # opacity outputs preserve the native ray information.
                        # Export transmission at each atmospheric radius for local stellar illumination.
                        # The adapter checks its integrated area against the pRT transit radius.
                        _atmospheric_chords = calculate_chord_transmission(
                            atmosphere, temperatures=temperature,
                            mass_fractions=mass_fractions, mean_molar_masses=MMW,
                            reference_gravity=_cst_chord.G*planet.M_pl/planet.R_pl**2,
                            reference_pressure=pm.reference_pressure_bar,
                            planet_radius=planet.R_pl,
                            opaque_cloud_top_pressure=pm.cloud_pressure_bar)

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
                            rt_inputs, cfg.atmosphere,
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
                                planet_rot_vel(planet.orbital_period_days, planet.R_pl), r1,
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
                                planet_rot_vel(planet.orbital_period_days, planet.R_pl), r1,
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
                                planet, with_signal, without_signal,
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

            # Store a reference to this order's dictionary. Blocks 4–6 add its
            # observation, preparation and CCF products. Block 7 combines them.
            per_order_results.append(_order_result)


            # ----------------------------------------------------------
            # Block 4a, stellar matrix (every order)
            # ----------------------------------------------------------
            from astropy.io import fits as _fits
            _berv_ref = (cfg.observation.berv_kms
                         if cfg.observation.berv_kms is not None else 0.0)
            # Real data: the analysis/injection velocity track (v_planet, the
            # stellar model, and any injected signal) must use the SAME
            # per-exposure BERV that the Kp-Vsys map co-add uses (loaded later
            # from observations_berv).  Otherwise an injected signal is placed
            # with berv=0 but recovered along a berv-shifted trail, so it lands
            # displaced by the full BERV (a real ~+18 km/s offset on a
            # large-BERV night).  Load the real BERV here for consistency.
            if (cfg.observation.use_real_data
                    and not cfg.observation.different_nights):
                _berv_file = (f"{cfg.paths.inputs_dir}reference_night/"
                              f"observations_berv_0.fits")
                if os.path.exists(_berv_file):
                    _berv_ref = _fits.open(_berv_file)[0].data
            if not cfg.observation.different_nights:
                if _stellar_surface is not None:
                    # Both RM/CLV modes use stellar reflex motion opposite to
                    # the planet's orbital RV, in the same observer frame.
                    v_star = stellar_radial_velocity(
                        phase, planet.stellar_rv_semiamplitude_kms,
                        planet.systemic_velocity_kms, _berv_ref)
                else:
                    v_star = get_V(
                        planet.stellar_rv_semiamplitude_kms, phase,
                        _berv_ref, planet.systemic_velocity_kms, 0,
                    )
                # The surface model already contains stellar rotation. Its
                # reference replaces the integrated template when simulating
                # RM/CLV, so rotational broadening is applied once.
                _stellar_model = (None if _stellar_surface is None or _spatial_rm_clv else
                                  _stellar_surface.evaluate(
                                      wave_ins, phase, v_star,
                                      cfg.observation.exposure_time_seconds, times_bjd=syn_jd))
                # The planet-only template builder needs an exposure-shaped
                # background. Spatial mode assembles the physical stellar
                # reference and full observed flux after the planet velocities.
                mat_star = (np.ones((len(phase), len(wave_ins))) if _spatial_rm_clv else
                            _stellar_model.reference if _stellar_model is not None else
                            get_stellar_matrix(spec_star_phoenix, v_star, wave_ins))
            else:
                v_star = []
                mat_star = []
                _stellar_model = []
                for nn in range(cfg.observation.n_nights):
                    _stellar_berv = (_berv_pn[nn] if _berv_pn is not None else _berv_ref)
                    # The enabled surface calculation must follow the same
                    # measured/reference BERV as the planetary track in Block
                    # 4b. Resolve it here before constructing stellar spectra.
                    if (_stellar_surface is not None and
                            (cfg.observation.use_real_data or
                             (cfg.observation.specific_event and
                              cfg.observation.exposure_time_seconds_per_night is None))):
                        _stellar_berv = _fits.getdata(
                            Path(cfg.paths.inputs_dir) / 'reference_night' /
                            f'observations_berv_{nn}.fits')
                    if _stellar_surface is not None:
                        v_star.append(stellar_radial_velocity(
                            phase[nn], planet.stellar_rv_semiamplitude_kms,
                            planet.systemic_velocity_kms, _stellar_berv))
                    else:
                        v_star.append(get_V(
                            planet.stellar_rv_semiamplitude_kms, phase[nn],
                            _stellar_berv,
                            planet.systemic_velocity_kms, 0,
                        ))
                    exposure_seconds = (cfg.observation.exposure_time_seconds
                        if cfg.observation.exposure_time_seconds_per_night is None else
                        cfg.observation.exposure_time_seconds_per_night[nn])
                    _model_night = (None if _stellar_surface is None or _spatial_rm_clv else
                                    _stellar_surface.evaluate(
                                        wave_ins, phase[nn], v_star[nn], exposure_seconds, times_bjd=syn_jd[nn]))
                    _stellar_model.append(_model_night)
                    mat_star.append(np.ones((len(phase[nn]), len(wave_ins))) if _spatial_rm_clv else
                                    _model_night.reference if _model_night is not None else
                                    get_stellar_matrix(spec_star_phoenix, v_star[nn], wave_ins))

            # ----------------------------------------------------------
            # Block 4b, planet velocity (h==0 only, reused per order)
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
                        # Synthetic different_nights (no reference BERV
                        # files): per-night BERV from the epoch geometry
                        # when available, else constant berv_kms.
                        berv.append(
                            np.asarray(_berv_pn[n], dtype=float)
                            if _berv_pn is not None
                            else np.full(int(n_spectra[n]),
                                         cfg.observation.berv_kms,
                                         dtype=float)
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
            # Injection placement: when a simulated planet is injected into
            # REAL data (injection-recovery tests), honour the
            # (dKp, dV_rest) offsets of pipeline.kp_vrest_injection so the
            # signal can be placed at an offset velocity.  [0, 0] keeps the
            # nominal track (the default). The
            # offsets apply ONLY to the injected matrix (the analysis
            # reference track, v_planet, is untouched).
            _inj_off = cfg.pipeline.kp_vrest_injection
            if (cfg.observation.use_real_data
                    and cfg.observation.simulate_planet
                    and not cfg.observation.different_nights
                    and not cfg.atmosphere.limb_asymmetries
                    and (_inj_off[0] != 0.0 or _inj_off[1] != 0.0)):
                if not cfg.observation.significant_eccentricity:
                    _v_inject = get_V(
                        planet.kp_kms + _inj_off[0], phase, berv,
                        planet.systemic_velocity_kms + _inj_off[1],
                        v_wind,
                    )
                else:
                    _v_inject = get_V_eccentric(
                        planet.kp_kms + _inj_off[0], phase,
                        planet.eccentricity,
                        planet.argument_of_periastron_deg,
                        berv,
                        planet.systemic_velocity_kms + _inj_off[1],
                        v_wind,
                    )
            else:
                _v_inject = v_planet

            if not cfg.atmosphere.limb_asymmetries:
                if not cfg.observation.different_nights:
                    spec_mat, spec_mat_shift = \
                        spec_to_mat_fraction(
                            _v_inject,
                            wave_ins,
                            wave_pRT,
                            syn_spec,
                            mat_star,
                            with_signal,
                            without_signal,
                            fraction,
                            event_type=cfg.observation.event_type,
                            signal_scale=cfg.observation.scale_injection,
                            limb_asymmetries=cfg.atmosphere.limb_asymmetries,
                            limb_divisions=cfg.atmosphere.limb_divisions,
                            resolving_power=inst.res,
                            include_star=False,
                        )
                else:
                    spec_mat       = []
                    spec_mat_shift = []
                    for n in range(cfg.observation.n_nights):
                        _a, _b = spec_to_mat_fraction(
                            v_planet[n],
                            wave_ins,
                            wave_pRT,
                            syn_spec,
                            mat_star[n],
                            with_signal[n],
                            without_signal[n],
                            fraction[n],
                            event_type=cfg.observation.event_type,
                            signal_scale=cfg.observation.scale_injection,
                            limb_asymmetries=cfg.atmosphere.limb_asymmetries,
                            limb_divisions=cfg.atmosphere.limb_divisions,
                            resolving_power=inst.res,
                            include_star=False,
                        )
                        spec_mat.append(_a)
                        spec_mat_shift.append(_b)

            else:   # limb asymmetries
                spec_mat, spec_mat_shift = \
                    spec_to_mat_fraction(
                        v_planet,
                        wave_ins,
                        wave_pRT,
                        syn_spec,
                        mat_star,
                        with_signal,
                        without_signal,
                        fraction,
                        syn_spec_morning_rot,
                        None,
                        syn_spec_evening_rot,
                        None,
                        _sf_evening,
                        None,
                        _sf_morning,
                        None,
                        event_type=cfg.observation.event_type,
                        signal_scale=cfg.observation.scale_injection,
                        limb_asymmetries=cfg.atmosphere.limb_asymmetries,
                        limb_divisions=cfg.atmosphere.limb_divisions,
                        resolving_power=inst.res,
                    )

            # ----------------------------------------------------------
            # Build the stellar contribution before telluric transmission and
            # noise. Spatial mode includes the continuum transit in its flux.
            # Factorized mode keeps BATMAN's white-light factor and adds the
            # continuum-normalised photospheric line distortion.
            if _spatial_rm_clv:
                # The coupled flux already contains opaque occultation and
                # atmospheric absorption. Replace the factorized observation
                # once, keeping the shifted planet template for CCF diagnostics.
                if cfg.observation.different_nights:
                    _stellar_model = []
                    for night in range(cfg.observation.n_nights):
                        exposure_seconds = (cfg.observation.exposure_time_seconds
                            if cfg.observation.exposure_time_seconds_per_night is None else
                            cfg.observation.exposure_time_seconds_per_night[night])
                        # Integrate the complete stellar and planetary observation for this night
                        # using its phases, stellar velocities and exposure duration.
                        _stellar_model.append(_stellar_surface.evaluate(
                            wave_ins, phase[night], v_star[night], exposure_seconds,
                            chords=_atmospheric_chords if cfg.observation.simulate_planet else None,
                            planet_velocity_kms=v_planet[night], times_bjd=syn_jd[night]))
                    # Use the resulting coupled flux as the observation and keep its stellar reference.
                    spec_mat = [model.flux.copy() for model in _stellar_model]
                    mat_star = [model.reference for model in _stellar_model]
                else:
                    # For one shared observing sequence, evaluate the coupled transit directly
                    # at the injection velocities already selected by this runner.
                    _stellar_model = _stellar_surface.evaluate(
                        wave_ins, phase, v_star, cfg.observation.exposure_time_seconds,
                        chords=_atmospheric_chords if cfg.observation.simulate_planet else None,
                        planet_velocity_kms=_v_inject, times_bjd=syn_jd)
                    # This flux includes opaque occultation, limb brightness and atmospheric absorption.
                    # It supplies the complete transit observation before tellurics and noise.
                    spec_mat = _stellar_model.flux.copy()
                    mat_star = _stellar_model.reference
            # For separate stellar and planetary factors, apply the photospheric line
            # distortion to the existing transit observation.
            elif cfg.rm_clv.simulate:
                from exoplore.stellar.observation import apply_factorized_surface
                if cfg.observation.different_nights:
                    for night, model in enumerate(_stellar_model):
                        apply_factorized_surface(spec_mat[night], model,
                            radius_ratio=planet.R_pl/planet.R_star,
                            transit_fraction=fraction[night],
                            planet_present=cfg.observation.simulate_planet,
                            include_reference=not cfg.atmosphere.limb_asymmetries)
                else:
                    apply_factorized_surface(spec_mat, _stellar_model,
                        radius_ratio=planet.R_pl/planet.R_star, transit_fraction=fraction,
                        planet_present=cfg.observation.simulate_planet,
                        include_reference=not cfg.atmosphere.limb_asymmetries)

            # Update per-order results with Block 4 outputs
            # ----------------------------------------------------------
            _order_result.update({
                "mat_star":       mat_star,
                "v_star":         v_star,
                "spec_mat":       spec_mat,
                "spec_mat_shift": spec_mat_shift,
            })
            if cfg.rm_clv.active:
                _order_result["rm_clv_model"] = _stellar_model

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

            if cfg.tellurics.include_tellurics and not cfg.observation.use_real_data:
                # Synthetic observations use the configured transmission files.
                # Observed-data runs read their measured spectra. include_tellurics
                # selects the preparation's telluric_variation setting.
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
                                    _fits_dn.open(inst.wave_file)[2].data[
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
                        # For ANDES the real SNR is computed after tellurics.
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
            # Build the spectrum we will search for, using ccf_template settings
            # (or the configured alternative source). It is separate from the
            # injected atmosphere above unless cc_with_true_model requests reuse.
            # ----------------------------------------------------------
            if cfg.cross_correlation.ccf_template_source == "telluric":
                # Telluric-template CCF: use the Earth-transmittance
                # spectrum as the cross-correlation template, handled
                # downstream exactly like a planetary template (same
                # normalisation, Doppler track, weighting and S/N map).
                # Stored as a depth (1 - T**airmass) so telluric
                # absorption carries the same sign as planetary
                # absorption in 5d.12 (mat = 1 - depth).
                from astropy.io import fits as _fits_tt
                with _fits_tt.open(_tell_ref_file) as _th_tt:
                    _lam_tt = _th_tt[1].data['lam'] * 1e-3
                    _trn_tt = _th_tt[1].data['trans']
                _sel_tt = ((_lam_tt >= wave_ins.min() - 0.01)
                           & (_lam_tt <= wave_ins.max() + 0.01))
                if not np.any(_sel_tt):
                    raise ValueError(
                        "ccf_template_source='telluric': the reference "
                        f"telluric file {_tell_ref_file} does not cover "
                        f"this order ({wave_ins.min():.3f}-"
                        f"{wave_ins.max():.3f} um).")
                wave_pRT_cc = np.asarray(_lam_tt[_sel_tt], dtype=np.float64)
                try:
                    _am_tt = float(np.mean(
                        np.concatenate([np.atleast_1d(a) for a in airmass])
                        if isinstance(airmass, list) else airmass))
                except Exception:
                    # No usable airmass in scope (e.g. fully synthetic
                    # multi-night setups): fall back to zenith.
                    _am_tt = 1.0
                _trn_tt = np.clip(
                    np.asarray(_trn_tt[_sel_tt], dtype=np.float64),
                    0.0, 1.0) ** _am_tt
                if cfg.instrument.convolve_to_resolution:
                    _trn_tt = convolve(wave_pRT_cc, _trn_tt, inst.res)
                spec_cc = 1.0 - _trn_tt
            elif cfg.cross_correlation.ccf_template_source == "stellar":
                # Stellar-template CCF (contamination diagnostic): use the
                # PHOENIX stellar spectrum as the cross-correlation
                # template, handled downstream exactly like a planetary
                # template.  Stored as a depth (1 - F/median) so stellar
                # absorption carries the same sign as planetary absorption
                # in 5d.12. The chev26 template normalisation removes the
                # continuum as it does for the data.
                from astropy.io import fits as _fits_st
                _wv_st = 1.0e-4 * _fits_st.getdata(
                    cfg.paths.phoenix_wave_file)
                _fl_st = _fits_st.getdata(cfg.paths.phoenix_flux_file)
                _sel_st = ((_wv_st >= wave_ins.min() - 0.01)
                           & (_wv_st <= wave_ins.max() + 0.01))
                if not np.any(_sel_st):
                    raise ValueError(
                        "ccf_template_source='stellar': the PHOENIX file "
                        "does not cover this order "
                        f"({wave_ins.min():.3f}-{wave_ins.max():.3f} um).")
                wave_pRT_cc = np.ascontiguousarray(
                    _wv_st[_sel_st], dtype=np.float64)
                _fl_sel = np.ascontiguousarray(
                    _fl_st[_sel_st], dtype=np.float64)
                if cfg.instrument.convolve_to_resolution:
                    _fl_sel = convolve(wave_pRT_cc, _fl_sel, inst.res)
                _fl_sel /= np.median(_fl_sel)
                spec_cc = 1.0 - _fl_sel
            elif cfg.atmosphere.cc_with_true_model:
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
                    rt_inputs, p_cc, _atmosphere_cc,
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
            _cor_b5          = None
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
                _spec_tp = _apply_throughput(
                    spec_mat, cfg.observation, order_selection[h], b)

                if _carmenes_etc is not None:
                    # Apply the same ETC-anchor prescription to every pipeline.
                    # A 2-D S/N matrix already includes relative telluric scaling.
                    _tell_etc = (tell_trans[b] if _dn and tell_trans is not None
                                 else tell_trans)
                    if _tell_etc is None:
                        _tell_etc = np.ones((n_spectra, n_pixels))
                    _snr_etc = _carmenes_etc.pixel_snr(
                        order_selection[h], wave_ins, _tell_etc, airmass,
                        cfg.observation.exposure_time_seconds, inst.res)
                    if _dn:
                        snr[b] = _snr_etc
                    else:
                        snr = _snr_etc

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
                        # array). Use ETC-based noise so the SNR mask does
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
                # Seed selection belongs to the observation layer. The draw and
                # uncertainty scaling below stay in their original order.
                # Gaussian seeds use the selected-order position h. Throughput
                # seeds use the physical instrument order ID order_selection[h].
                _nseed = gaussian_noise_seed(
                    cfg.noise.noise_seed, n_nights=_nn,
                    pipeline_name=cfg.pipeline.name,
                    selected_order_position=h, night_index=b,
                )
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
                    # A fixed per-pixel SNR can be one-dimensional while
                    # uncertainties have one row per exposure. Let NumPy
                    # broadcast the pixel mask into that time-series shape.
                    _mask_snr_bool = _mask_snr_bool | (_std_noise_b5[b, :] < 1e-9)
                    _mask_snr_indices = np.argwhere(_mask_snr_bool)
                else:
                    _mask_snr_bool = (
                        (snr[b] / _nf) < cfg.pipeline.snr_mask_threshold)
                    _mask_snr_bool = _mask_snr_bool | (
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
                        # The orbital alignment in Block 7 uses these
                        # measured barycentric corrections for each exposure.

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

                # Direct transmission prepares the stellar model alongside
                # the data, then corrects the individual master-out ratios.
                # Other preparation methods retain their native-flux correction.
                # Prepare an optional atmosphere-free stellar spectrum for Seidel22.
                _seidel_stellar_flux = None
                if cfg.rm_clv.active:
                    from exoplore.stellar.observation import correct_stellar_distortion
                    _model_current = (_stellar_model[b] if _dn else _stellar_model)
                    # Seidel22 will normalize this stellar spectrum alongside the data, build
                    # a matching model reference and divide the individual transmission ratios.
                    if cfg.rm_clv.correct and cfg.pipeline.name == 'Seidel22':
                        if _correction_surface is not None:
                            exposure_seconds = (cfg.observation.exposure_time_seconds
                                if cfg.observation.exposure_time_seconds_per_night is None else
                                cfg.observation.exposure_time_seconds_per_night[b])
                            # Calculate the requested separate stellar-correction model for this night.
                            correction_model = _correction_surface.evaluate(
                                wave_ins, phase,
                                v_star[b] if _dn else v_star, exposure_seconds)
                            _seidel_stellar_flux = (correction_model.reference
                                                    * correction_model.distortion)
                        else:
                            # The matched choice reconstructs the opaque-body spectrum from the injected model.
                            _seidel_stellar_flux = (_model_current.reference
                                                    * _model_current.distortion)
                            if _model_current.opaque_transmission is not None:
                                # Restore the continuum transit when reconstructing a spatial opaque-body spectrum.
                                _seidel_stellar_flux *= _model_current.opaque_transmission
                        if _model_current.correction_flux is not None:
                            _seidel_stellar_flux = _model_current.correction_flux
                    # For other preparations, apply the fixed distortion to flux and error
                    # on the observer grid before their preparation begins.
                    elif cfg.rm_clv.correct:
                        _flux = _mat_noise_b5[b, :n_spectra, :]
                        _error = _std_noise_b5[b, :n_spectra, :]
                        (_mat_noise_b5[b, :n_spectra, :],
                         _std_noise_b5[b, :n_spectra, :]) = correct_stellar_distortion(
                            _flux, _error, (_model_current.correction_distortion
                                            if _model_current.correction_distortion is not None else _model_current.distortion))
                        # Keep the saved Gaussian draw in these same flux
                        # units. The noiseless diagnostic subtracts this draw
                        # from the corrected noisy matrix later in the run.
                        if not _nl and not _urd:
                            _gauss_noise_b5[b, :n_spectra, :] /= (_model_current.correction_distortion
                                if _model_current.correction_distortion is not None else _model_current.distortion)
                    # Save the stellar factors, model source and correction stage for later
                    # comparison plots and checks of the applied normalization.
                    if cfg.output.save_rm_clv_model:
                        _spatial_products = {}
                        if _seidel_stellar_flux is not None:
                            _spatial_products['correction_flux'] = _seidel_stellar_flux
                        if _spatial_rm_clv:
                            _spatial_products.update(dict(
                                coupled_flux=_model_current.flux,
                                opaque_transmission=_model_current.opaque_transmission,
                                atmosphere_base_radius_cm=_model_current.base_radius_cm,
                                atmosphere_top_radius_cm=_model_current.top_radius_cm,
                                planet_velocity_kms=_v_inject[b] if _dn else _v_inject,
                                prt_version=(_atmospheric_chords.prt_version
                                             if _atmospheric_chords is not None else ""),
                                atmospheric_radial_points=(len(_atmospheric_chords.impact_parameter_cm)
                                    if _atmospheric_chords is not None else 0)))
                        np.savez_compressed(
                            Path(dirs['root']) / f"rm_clv_order_{order_selection[h]}_night_{b}.npz",
                            wavelength_um=wave_ins, phase=phase,
                            distortion=_model_current.distortion,
                            reference=_model_current.reference,
                            stellar_velocity_kms=v_star[b] if _dn else v_star,
                            effects=cfg.rm_clv.effects,
                            coupling=cfg.rm_clv.coupling,
                            normalization=cfg.rm_clv.normalization,
                            simulate=cfg.rm_clv.simulate, correct=cfg.rm_clv.correct,
                            correction_stage=("stellar_residuals" if _seidel_stellar_flux is not None
                                              else "native_flux" if cfg.rm_clv.correct else "disabled"),
                            correction_method=cfg.rm_clv.correction_method,
                            correction_source=(_correction_surface.metadata['source']
                                if _correction_surface is not None else _stellar_surface.metadata['source']),
                            model_source=_stellar_surface.metadata['source'], **_spatial_products)

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
                    # Only SYSREM/PCA pipelines have a per-order component count.
                    # BL19 and Blain24 are polynomial fitting pipelines (no SYSREM,
                    # no PCA), so this stays None and is saved as an empty array.
                    _sysrem_passes_per_order = (
                        np.full(n_orders, _si, int)
                        if cfg.pipeline.name in {"ASL19", "Gibson22",
                                                 "Cheverall26", "Nortmann26"}
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
                preparation_inputs = PreparationInputs(
                    pipeline=cfg.pipeline,
                    tellurics=cfg.tellurics,
                    use_real_data=_urd,
                    stellar_rv_kms=(np.asarray(v_star[b] if _dn else v_star)
                                    if cfg.pipeline.name == 'Seidel22' else None),
                    stellar_model_flux=_seidel_stellar_flux,
                    sysrem_iterations=_si,
                    telluric_transmission=(
                        _telluric_model[order_selection[h]] if _telluric_model is not None else None
                    ),
                )

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
                            wave_ins,
                            _mat_noise_inj_b5[b, :],
                            wave_pRT,
                            syn_spec,
                            with_signal,
                            without_signal,
                            fraction,
                            phase,
                            mat_star,
                            event_type=cfg.observation.event_type,
                            signal_scale=cfg.pipeline.inject_scale_factor,
                            limb_asymmetries=cfg.atmosphere.limb_asymmetries,
                            limb_divisions=cfg.atmosphere.limb_divisions,
                            resolving_power=inst.res,
                            kp_kms=cfg.pipeline.kp_vrest_injection[0],
                            vrest_kms=cfg.pipeline.kp_vrest_injection[1],
                            berv_kms=berv,
                            systemic_velocity_kms=planet.systemic_velocity_kms,
                        )
                    else:
                        _mat_noise_inj_b5[b, :] = injection(
                            wave_ins,
                            _mat_noise_inj_b5[b, :n_spectra, :],
                            wave_pRT,
                            syn_spec,
                            with_signal,
                            without_signal,
                            fraction,
                            phase,
                            mat_star,
                            event_type=cfg.observation.event_type,
                            signal_scale=cfg.pipeline.inject_scale_factor,
                            limb_asymmetries=cfg.atmosphere.limb_asymmetries,
                            limb_divisions=cfg.atmosphere.limb_divisions,
                            resolving_power=inst.res,
                            kp_kms=cfg.pipeline.kp_vrest_injection[0],
                            vrest_kms=cfg.pipeline.kp_vrest_injection[1],
                            berv_kms=berv,
                            systemic_velocity_kms=planet.systemic_velocity_kms,
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
                # (with injection × SYSREM iteration axes). DeltaSigma uses
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

                # Prepare data and propagate their uncertainties together, using
                # this order/night's already selected columns. _sl excludes only
                # exposure-storage padding when nights have different lengths.
                # The dispatcher returns residuals, uncertainties, masks and
                # detrending information needed later by CCFs and retrieval.
                _sl = slice(None, n_spectra) if _dn else slice(None)
                _pp_result = prepare_observation(
                    preparation_inputs,
                    _mat_noise_b5[b, _sl],
                    _std_noise_b5[b, _sl],
                    wave_ins,
                    _useful_snr,
                    _mask_snr_1d,
                    airmass,
                    phase,
                    without_signal,
                    None,
                    _inj_slice_b5,
                    tell_mask_threshold_Blain24=0.8,
                    max_fit_BL19=False,
                    sysrem_division=False,
                    masks=True,
                    correct_uncertainties=True,
                )
                if not cfg.pipeline.sysrem_robust_halt:
                    (
                        _mat_res_b5[b, _sl],
                        _propag_noise_b5[b, _sl],
                        _useful_spectral_points_b5,
                        _mask_b5, _n_passes_b5,
                        _inter_mask_b5, _inter_useful_b5,
                        _cor_b5,
                        _U_ret_b5,
                    ) = _pp_result
                    # With DeltaSigma halting the pipeline may stop before
                    # sysrem_iterations passes. pad the U store accordingly.
                    # Non-SYSREM pipelines return a broadcastable placeholder.
                    _U_ret_b5 = np.asarray(_U_ret_b5)
                    if _U_ret_b5.ndim == 2:
                        _U_sysrem[b, _sl, :_U_ret_b5.shape[1]] = _U_ret_b5
                    else:
                        _U_sysrem[b, _sl] = _U_ret_b5
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
                                _v_cc_b5,
                                wave_ins,
                                wave_pRT_cc,
                                spec_cc,
                                mat_star,
                                _ws_ccf0,
                                without_signal,
                                np.ones_like(fraction),
                                event_type=cfg.observation.event_type,
                                signal_scale=cfg.observation.scale_injection,
                                limb_asymmetries=False,
                                limb_divisions=cfg.atmosphere.limb_divisions,
                                resolving_power=inst.res,
                                include_star=False,
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
                                            n_spectra,
                                            _Nf,
                                            _Uf,
                                            n_nights=_nn,
                                            sysrem_iterations=_n_proj,
                                        ))
                                    _syn_mat_res_b5 = (
                                        filter_model_singleorder(
                                            _Pf[0, 0, :, :], _mat_cc_b5,
                                            _useful_spectral_points_b5,
                                        ))
                                    del _Pf, _Uf, _Nf
                                else:
                                    _syn_mat_res_b5 = prepare_observation(
                                        preparation_inputs,
                                        _mat_cc_b5,
                                        _std_noise_b5[b, :],
                                        wave_ins,
                                        _useful_spectral_points_b5,
                                        _mask_b5,
                                        airmass,
                                        phase,
                                        without_signal,
                                        _sysrem_pass_b5,
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
                                # for the Cheverall26 CCF template. The template preparation
                                # consists of rescaling and continuum normalization.
                                # Other pipelines use the reprocessing branch above.
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
                                # Use the fixed, Cheverall26-normalised template for every
                                # exposure, including out-of-transit exposures, so the
                                # duration test samples their noise CCFs.
                            else:
                                _syn_mat_res_b5 = _mat_cc_b5
                                _syn_mat_res_b5[without_signal, :] = 1.01

                            for _i in range(n_spectra):
                                # Interpolate the (velocity-shifted) template
                                # onto exposure i's own wavelength grid so it
                                # aligns with the native-flux data columns
                                # (per-frame solution, falls back to common).
                                _out_grid = (_wave_pf_ord[_i]
                                             if _wave_pf_ord is not None
                                             else wave_ins)
                                _mat_back_b5[_i, :] = np.interp(
                                    _out_grid,
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
                            _v_cc_b5,
                            wave_ins,
                            wave_pRT_cc,
                            spec_cc,
                            mat_star,
                            with_signal,
                            without_signal,
                            fraction,
                            event_type=cfg.observation.event_type,
                            signal_scale=cfg.observation.scale_injection,
                            limb_asymmetries=False,
                            limb_divisions=cfg.atmosphere.limb_divisions,
                            resolving_power=inst.res,
                            include_star=False,
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
                                        n_spectra,
                                        _Nf,
                                        _Uf,
                                        no_night_loop=True,
                                        n_nights=_nn,
                                        sysrem_iterations=_si,
                                    ))
                                _syn_mat_res_b5 = (
                                    filter_model_singleorder(
                                        _Pf[0, 0, :, :], _mat_cc_b5,
                                        _useful_spectral_points_b5,
                                    ))
                                del _Pf, _Uf, _Nf
                            else:
                                _syn_mat_res_b5 = prepare_observation(
                                    preparation_inputs,
                                    _mat_cc_b5,
                                    _std_noise_b5[b, :n_spectra, :],
                                    wave_ins,
                                    _useful_spectral_points_b5,
                                    _mask_b5,
                                    airmass,
                                    phase,
                                    without_signal,
                                    _sysrem_pass_b5,
                                    None,
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

                    # 6b, ccf_store allocation. Allocate on the first night that
                    # reaches here: night 0 may have hit the fully-masked
                    # `continue` above, so gating on b==0 could leave this None
                    # and crash a later night at the NaN-fill below.
                    if _ccf_store_b6 is None:
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
                    # the propagated per-pixel noise. "mad_residual" uses the
                    # per-wavelength-channel MAD of the time-series residuals
                    # (Gibson+20, Nortmann+24, Cheverall+26).  Applied on the
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
                # Structure removed by the detrending (empirical telluric/
                # stellar template). single-night shape (n_spectra,
                # n_pixels).  Used by the "embedded" model-reprocessing
                # mode in Block 9.
                "cor": (np.copy(_cor_b5) if _cor_b5 is not None else None),
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
                else:
                    # Polynomial pipelines (BL19, Blain24): no iterations, so a
                    # single "Corrected (residual)" panel (raw -> corrected).
                    try:
                        _resid_full = np.full_like(_my_steps, np.nan)
                        _resid_full[:, _usp_steps] = _mr_steps[:, _usp_steps]
                        _sysrem_stages = {'Corrected (residual)': _resid_full}
                        _sysrem_iters = ['Corrected (residual)']
                    except Exception as _e_poly:
                        print(f"  NOTE: corrected-stage panel skipped ({_e_poly})")
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
                    noise_illustration_factor = cfg.plotting.pipeline_steps_noise_illustration_factor,
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
        #         specialised optional analyses requiring saved per-order
        #         noise arrays. The current runner does not execute them.
        # ================================================================

        # Shared plot settings keep the same labels and reference motion across
        # the CCF trail, detection map and velocity slices.
        night_study = NightStudyInputs(
            cross_correlation=cfg.cross_correlation,
            pipeline=cfg.pipeline,
            n_orders=n_orders,
            n_nights=_nn,
            stack_group_size=cfg.statistics.stack_group_size,
            berv_kms=berv,
            systemic_velocity_kms=planet.systemic_velocity_kms,
            reference_kp_kms=planet.kp_kms,
            reference_vrest_kms=pm.wind_velocity_kms,
            sysrem_iterations=_si,
            exclude_around="point" if cfg.pipeline.name == "Cheverall26" else "peak",
        )

        detection_plot = DetectionPlotSettings(
            simulation_name=sim_name,
            plots_dir=str(dirs["plots"]) + "/",
            reference_kp_kms=planet.kp_kms,
            kp_max_kms=cfg.cross_correlation.kp_max_kms,
            velocity_half_range_kms=cfg.cross_correlation.noise_velocity_max_kms,
            velocity_tick_step_kms=cfg.cross_correlation.plot_ccf_xstep,
            sysrem_iterations=_si,
            optimize_sysrem=_opt,
            event_type=cfg.observation.event_type,
        )

        # Read atmospheric and analysis choices from their configuration objects.
        # Local labels and selections below can change as products are saved or
        # fully masked orders are removed from the retrieval.
        retrieval_atmosphere_config = (
            cfg.retrieval.atmosphere
            if cfg.retrieval.atmosphere is not None else pm
        )
        sampling_method = (
            "Nested_sampling"
            if cfg.retrieval.sampler.lower().replace("-", "_")
            in ("nested_sampling", "multinest", "nested") else "MCMC"
        )
        use_snr_statistic = cfg.cross_correlation.ccf_snr
        use_welch_statistic = cfg.cross_correlation.welch_ttest
        plot_name = sim_name
        prepare_retrieval_template = cfg.pipeline.prepare_template
        analysis_order_count = n_orders
        retrieval_orders = order_selection
        retrieval_limb_asymmetries = cfg.atmosphere.limb_asymmetries
        retrieval_species = retrieval_atmosphere_config.species
        limb_retrieval = LimbRetrievalState(
            morning_abundances=cfg.atmosphere.morning_day.vmr,
            evening_abundances=cfg.atmosphere.evening_day.vmr,
        )

        # Pre-Block-7 outputs (initialised to None, set only when paths run)
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
                for _b7 in range(_nn):
                    _oi7 = list(range(n_orders))
                    ccf_nights[_b7] = np.nansum(
                        _ccf_all_orders[_oi7, _b7, :], 0)

            # 7c, CO-ADD NIGHTS → ccf_complete + ERF plot
            _plots_subdir_b7 = str(dirs["plots"])
            os.makedirs(_plots_subdir_b7, exist_ok=True)

            _zoom = cfg.cross_correlation.plot_velocity_zoom_kms
            _erf_xlims = [-_zoom, _zoom] if _zoom else None
            if not _dn:
                ccf_complete = np.nansum(ccf_nights, 0)
                CCF_matrix_ERF(
                    replace(detection_plot, simulation_name=plot_name),
                    _v_ccf,
                    phase,
                    ccf_complete,
                    with_signal,
                    without_signal,
                    v_planet,
                    show_plot=False,
                    save_plot=True,
                    CCF_Noise=False,
                    xlims=_erf_xlims,
                )
            else:
                ccf_complete = ccf_nights
                for _bb7 in range(_nn):
                    CCF_matrix_ERF(
                        replace(detection_plot, simulation_name=plot_name),
                        _v_ccf,
                        phase[_bb7],
                        ccf_complete[_bb7, :, : int(n_spectra[_bb7])],
                        with_signal[_bb7],
                        without_signal[_bb7],
                        v_planet[_bb7],
                        show_plot=False,
                        save_plot=False,
                        CCF_Noise=False,
                        xlims=_erf_xlims,
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
                    _kp_t  = float(planet.kp_kms)
                    _vs_t  = float(planet.systemic_velocity_kms)
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
                                f"{plot_name}.npz",
                                snmap=_snm, vsys_axis=_vsa,
                                vsys_expected=_vs_t, kp=_kp_t,
                                n_in_cross=_ncross)
                            # Label the figure with the template it came from,
                            # so a run never claims the wrong species.
                            try:
                                import re as _re_t
                                _src_t = cfg.cross_correlation.ccf_template_source
                                if _src_t in ("stellar", "telluric"):
                                    _lbl_t = _src_t.capitalize()
                                else:
                                    from exoplore.atmosphere.prt import (
                                        easychem_molecule as _echem_t)
                                    # "CO_high_Sam" and friends are opacity
                                    # variants of one molecule. The label wants
                                    # only the molecule.
                                    _mol_t = [_echem_t(_s).split("_")[0]
                                              for _s in cc.species]
                                    _lbl_t = ", ".join(
                                        _re_t.sub(r"(\d+)", r"$_\1$", _m)
                                        for _m in _mol_t
                                        if _m not in ("H2", "He")
                                    ) or None
                            except Exception:
                                _lbl_t = None
                            chev26_plot_duration_map(
                                _snm, _vsa, _vs_t,
                                f"{_plots_subdir_b7}/duration_test_fig4_"
                                f"{plot_name}.png",
                                kp=_kp_t, n_in_expected=_ncross,
                                species=_lbl_t)
                    save_compressed(
                        _base_dir_b7, sim_name,
                        {'stats_timing_pos': _chev_tim_pos,
                         'stats_timing_max': _chev_tim_max})
                    print(f"  [Cheverall26] in-run duration/timing map + "
                          f"metrics produced (N_in_cross={_ncross})")
                except Exception as _e:
                    print(f"  [Cheverall26] timing map skipped: {_e}")

            # 7e, get_shifted_ccf_matrix (Kp-grid shift)
            # CCFs are still exposure-by-exposure in the Earth velocity frame.
            # Sample each exposure along trial orbital tracks before summing
            # exposures. Summing first would discard the planet's changing RV.
            if not _dn:
                ccf_values_shift = get_shifted_ccf_matrix(
                    with_signal,
                    v_rest,
                    _v_ccf,
                    _kp_range,
                    phase,
                    planet.systemic_velocity_kms,
                    berv,
                    _pix_lr_b7,
                    _ccf_v_step,
                    ccf_complete,
                    sysrem_opt=_opt_needs_injection,
                    significant_eccentricity=cfg.observation.significant_eccentricity,
                    eccentricity=planet.eccentricity,
                    argument_of_periastron_deg=planet.argument_of_periastron_deg,
                    sysrem_iterations=_si,
                )
            else:
                ccf_values_shift = []
                for _b7 in range(_nn):
                    ccf_values_shift.append(
                        get_shifted_ccf_matrix(
                            with_signal[_b7],
                            v_rest,
                            _v_ccf,
                            _kp_range,
                            phase[_b7],
                            planet.systemic_velocity_kms,
                            berv[_b7],
                            _pix_lr_b7,
                            _ccf_v_step,
                            ccf_complete[_b7, :, : int(n_spectra[_b7])],
                            sysrem_opt=_opt_needs_injection,
                            significant_eccentricity=cfg.observation.significant_eccentricity,
                            eccentricity=planet.eccentricity,
                            argument_of_periastron_deg=planet.argument_of_periastron_deg,
                            sysrem_iterations=_si,
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
                         ccf_tot,
                         v_rest,
                         _kp_range,
                         b=None,
                         stats=None,
                         sysrem_opt=_opt_needs_injection,
                         CCF_Noise=False,
                         snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                         sysrem_iterations=_si,
                         exclude_around=("point" if cfg.pipeline.name == "Cheverall26" else "peak"),
                         noise_source=cfg.cross_correlation.snr_noise_source,
                         noise_kp_exclude_kms=cfg.cross_correlation.snr_noise_kp_exclude_kms,
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
                            f"{plot_name}.npz",
                            ccf_tot_sn=ccf_tot_sig, v_rest=v_rest,
                            kp_range=_kp_range,
                            sysrem_opt=bool(_opt_needs_injection),
                            order_indices=np.asarray(order_selection),
                            n_components_per_order=(
                                np.asarray(_sysrem_passes_per_order)
                                if _sysrem_passes_per_order is not None
                                else np.array([])),
                            # v_rest is an OFFSET from the expected planet
                            # trail: a signal at the nominal (Kp, V_sys)
                            # peaks at v_rest = 0 (not at V_sys).
                            v_rest_convention="offset_from_expected_trail",
                        )
                    except Exception as _e:
                        print(f"  NOTE: S/N map save skipped ({_e})")
                    plot_Kp_Vrest(
                        replace(detection_plot, simulation_name=plot_name),
                        _kp_range,
                        ccf_tot_sig,
                        v_rest,
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    plot_1D_CCF(
                        replace(detection_plot, simulation_name=plot_name),
                        v_rest,
                        ccf_tot_sig,
                        max_kp_idx,
                        max_sig,
                        _n_kp,
                        max_v_wind,
                        [-100, 100],
                        show_plot=False,
                        save_plot=True,
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

                    # Per-night Kp-Vsys maps + 1D CCFs + overlay for the
                    # simple co-add (different_nights=False, n_nights>1). The
                    # nights share one geometry (identical phase/berv), so each
                    # night's map comes from ccf_nights[b] with the same shift
                    # arguments. ccf_complete above is their co-add. Mirrors
                    # the different_nights per-night save/plot calls so the
                    # nightwise CCFs and the overlay are available alongside
                    # the combined map.
                    if _nn > 1:
                        _night_1d_slices = []
                        for _bco in range(_nn):
                            plot_name = (
                                f"{sim_name}_night{_bco}")
                            _cvs_bco = get_shifted_ccf_matrix(
                                with_signal,
                                v_rest,
                                _v_ccf,
                                _kp_range,
                                phase,
                                planet.systemic_velocity_kms,
                                berv,
                                _pix_lr_b7,
                                _ccf_v_step,
                                ccf_nights[_bco],
                                sysrem_opt=_opt_needs_injection,
                                significant_eccentricity=cfg.observation.significant_eccentricity,
                                eccentricity=planet.eccentricity,
                                argument_of_periastron_deg=planet.argument_of_periastron_deg,
                                sysrem_iterations=_si,
                            )
                            _ccf_tot_bco = np.nansum(_cvs_bco, 1)
                            (_cts_bco, _ms_bco, _mki_bco,
                             _mvw_bco, _) = get_max_CCF_peak(
                                 _ccf_tot_bco,
                                 v_rest,
                                 _kp_range,
                                 b=None,
                                 stats=None,
                                 sysrem_opt=_opt_needs_injection,
                                 CCF_Noise=False,
                                 snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                                 sysrem_iterations=_si,
                                 exclude_around=("point" if cfg.pipeline.name == "Cheverall26" else "peak"),
                                 noise_source=cfg.cross_correlation.snr_noise_source,
                                 noise_kp_exclude_kms=cfg.cross_correlation.snr_noise_kp_exclude_kms,
                             )
                            try:
                                _snr_mdco = str(dirs["matrices"])
                                os.makedirs(_snr_mdco, exist_ok=True)
                                np.savez(
                                    f"{_snr_mdco}/ccf_tot_sn_map_"
                                    f"{plot_name}.npz",
                                    ccf_tot_sn=_cts_bco, v_rest=v_rest,
                                    kp_range=_kp_range,
                                    sysrem_opt=bool(_opt_needs_injection),
                                    order_indices=np.asarray(order_selection),
                                )
                            except Exception as _eco:
                                print("  NOTE: per-night S/N map save "
                                      f"skipped ({_eco})")
                            plot_Kp_Vrest(
                                replace(detection_plot, simulation_name=plot_name),
                                _kp_range,
                                _cts_bco,
                                v_rest,
                                show_plot=False,
                                save_plot=True,
                                sysrem_opt=_opt_needs_injection,
                            )
                            plot_1D_CCF(
                                replace(detection_plot, simulation_name=plot_name),
                                v_rest,
                                _cts_bco,
                                _mki_bco,
                                _ms_bco,
                                _n_kp,
                                _mvw_bco,
                                [-100, 100],
                                show_plot=False,
                                save_plot=True,
                                sysrem_opt=_opt_needs_injection,
                            )
                            _night_1d_slices.append(
                                (_cts_bco[:, int(_mki_bco)], float(_ms_bco))
                            )
                        plot_name = sim_name  # restore
                        from exoplore.plotting.kpvsys import (
                            plot_multi_night_1D_CCF)
                        plot_multi_night_1D_CCF(
                            replace(detection_plot, simulation_name=plot_name),
                            v_rest,
                            _night_1d_slices,
                            (ccf_tot_sig[:, int(max_kp_idx)], float(max_sig)),
                            xlims=[-100, 100],
                            show_plot=False,
                            save_plot=True,
                        )
                else:  # Different_nights
                    # Per-night Kp-Vsys maps (each saved with _nightN suffix)
                    _night_1d_slices = []
                    for _nn7 in range(_nn):
                        plot_name = (
                            f"{sim_name}_night{_nn7}")
                        ccf_tot = np.nansum(ccf_values_shift[_nn7], 1)
                        (ccf_tot_sig, max_sig, max_kp_idx,
                         max_v_wind, _) = get_max_CCF_peak(
                             ccf_tot,
                             v_rest,
                             _kp_range,
                             b=None,
                             stats=None,
                             sysrem_opt=_opt_needs_injection,
                             CCF_Noise=False,
                             snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                             sysrem_iterations=_si,
                             exclude_around=("point" if cfg.pipeline.name == "Cheverall26" else "peak"),
                             noise_source=cfg.cross_correlation.snr_noise_source,
                             noise_kp_exclude_kms=cfg.cross_correlation.snr_noise_kp_exclude_kms,
                         )
                        try:
                            _snr_md7 = str(dirs["matrices"])
                            os.makedirs(_snr_md7, exist_ok=True)
                            np.savez(
                                f"{_snr_md7}/ccf_tot_sn_map_"
                                f"{plot_name}.npz",
                                ccf_tot_sn=ccf_tot_sig, v_rest=v_rest,
                                kp_range=_kp_range,
                                sysrem_opt=bool(_opt_needs_injection),
                                order_indices=np.asarray(order_selection),
                                n_components_per_order=(
                                    np.asarray(_sysrem_passes_per_order)
                                    if _sysrem_passes_per_order is not None
                                    else np.array([])),
                            )
                        except Exception as _e7:
                            print(f"  NOTE: per-night S/N map save skipped ({_e7})")
                        plot_Kp_Vrest(
                            replace(detection_plot, simulation_name=plot_name),
                            _kp_range,
                            ccf_tot_sig,
                            v_rest,
                            show_plot=False,
                            save_plot=True,
                            sysrem_opt=_opt_needs_injection,
                        )
                        plot_1D_CCF(
                            replace(detection_plot, simulation_name=plot_name),
                            v_rest,
                            ccf_tot_sig,
                            max_kp_idx,
                            max_sig,
                            _n_kp,
                            max_v_wind,
                            [-100, 100],
                            show_plot=False,
                            save_plot=True,
                            sysrem_opt=_opt_needs_injection,
                        )
                        _night_1d_slices.append(
                            (ccf_tot_sig[:, int(max_kp_idx)], float(max_sig))
                        )
                    plot_name = sim_name  # restore

                    # Combined Kp-Vsys map: co-add all nights
                    # nansum over spectra per night then sum across nights, 
                    # valid even when n_spectra differs per night.
                    _ccf_combined = sum(
                        np.nansum(ccf_values_shift[_n], 1)
                        for _n in range(_nn)
                    )
                    (ccf_tot_sig, max_sig, max_kp_idx,
                     max_v_wind, _) = get_max_CCF_peak(
                         _ccf_combined,
                         v_rest,
                         _kp_range,
                         b=None,
                         stats=None,
                         sysrem_opt=_opt_needs_injection,
                         CCF_Noise=False,
                         snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                         sysrem_iterations=_si,
                         exclude_around=("point" if cfg.pipeline.name == "Cheverall26" else "peak"),
                         noise_source=cfg.cross_correlation.snr_noise_source,
                         noise_kp_exclude_kms=cfg.cross_correlation.snr_noise_kp_exclude_kms,
                     )
                    try:
                        _snr_mdc = str(dirs["matrices"])
                        os.makedirs(_snr_mdc, exist_ok=True)
                        np.savez(
                            f"{_snr_mdc}/ccf_tot_sn_map_"
                            f"{plot_name}.npz",
                            ccf_tot_sn=ccf_tot_sig, v_rest=v_rest,
                            kp_range=_kp_range,
                            sysrem_opt=bool(_opt_needs_injection),
                            order_indices=np.asarray(order_selection),
                            n_components_per_order=(
                                np.asarray(_sysrem_passes_per_order)
                                if _sysrem_passes_per_order is not None
                                else np.array([])),
                            # v_rest is an OFFSET from the expected planet
                            # trail: a signal at the nominal (Kp, V_sys)
                            # peaks at v_rest = 0 (not at V_sys).
                            v_rest_convention="offset_from_expected_trail",
                        )
                    except Exception as _ec:
                        print(f"  NOTE: combined S/N map save skipped ({_ec})")
                    plot_Kp_Vrest(
                        replace(detection_plot, simulation_name=plot_name),
                        _kp_range,
                        ccf_tot_sig,
                        v_rest,
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    plot_1D_CCF(
                        replace(detection_plot, simulation_name=plot_name),
                        v_rest,
                        ccf_tot_sig,
                        max_kp_idx,
                        max_sig,
                        _n_kp,
                        max_v_wind,
                        [-100, 100],
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    # Overlay: all individual nights + combined on one figure
                    from exoplore.plotting.kpvsys import plot_multi_night_1D_CCF
                    plot_multi_night_1D_CCF(
                        replace(detection_plot, simulation_name=plot_name),
                        v_rest,
                        _night_1d_slices,
                        (ccf_tot_sig[:, int(max_kp_idx)], float(max_sig)),
                        xlims=[-100, 100],
                        show_plot=False,
                        save_plot=True,
                    )

            elif (not cfg.cross_correlation.ccf_snr
                  and cfg.cross_correlation.welch_ttest
                  and not cfg.cross_correlation.all_significance_metrics):
                # --- Welch only ---
                ccf_tot = np.nansum(ccf_values_shift, 1)
                (ccf_tot_sig, _, _, v_rest_ttest, max_sig,
                 max_kp_idx, max_v_wind,
                 _, _, _, _, _, _) = Welch_ttest_map(
                     ccf_values_shift,
                     v_rest,
                     _kp_range,
                     CCF_Noise=False,
                     in_trail_left_right=cfg.cross_correlation.in_trail_left_right,
                     snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                     velocity_step_kms=cfg.cross_correlation.velocity_step_kms,
                 )
                # Persist the Welch Kp-Vrest map for completeness (distinct
                # name so it never collides with the CCF S/N map).
                try:
                    _snr_mat_dir = str(dirs["matrices"])
                    os.makedirs(_snr_mat_dir, exist_ok=True)
                    np.savez(
                        f"{_snr_mat_dir}/ccf_tot_welch_map_"
                        f"{plot_name}.npz",
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
                    replace(detection_plot, simulation_name=plot_name),
                    _kp_range,
                    ccf_tot_sig,
                    v_rest_ttest,
                    show_plot=False,
                    save_plot=False,
                    sysrem_opt=_opt_needs_injection,
                )
                plot_1D_CCF(
                    replace(detection_plot, simulation_name=plot_name),
                    v_rest_ttest,
                    ccf_tot_sig,
                    max_kp_idx,
                    max_sig,
                    _n_kp,
                    max_v_wind,
                    [-100, 100],
                    show_plot=False,
                    save_plot=False,
                    sysrem_opt=_opt_needs_injection,
                )

            elif cfg.cross_correlation.all_significance_metrics:
                # --- Both S/N and Welch ---
                if not _dn:
                    ccf_tot = np.nansum(ccf_values_shift, 1)
                    # S/N branch
                    use_snr_statistic    = True
                    use_welch_statistic = False
                    (ccf_tot_sn, max_sn, max_kp_idx_sn,
                     max_v_wind_sn, _) = get_max_CCF_peak(
                         ccf_tot,
                         v_rest,
                         _kp_range,
                         b=None,
                         stats=None,
                         sysrem_opt=_opt_needs_injection,
                         CCF_Noise=False,
                         snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                         sysrem_iterations=_si,
                         exclude_around=("point" if cfg.pipeline.name == "Cheverall26" else "peak"),
                         noise_source=cfg.cross_correlation.snr_noise_source,
                         noise_kp_exclude_kms=cfg.cross_correlation.snr_noise_kp_exclude_kms,
                     )
                    # Persist the combined Kp-Vrest S/N map so a zoomed figure
                    # and the S/N at a chosen velocity can be produced post-hoc.
                    try:
                        _snr_mat_dir = str(dirs["matrices"])
                        os.makedirs(_snr_mat_dir, exist_ok=True)
                        np.savez(
                            f"{_snr_mat_dir}/ccf_tot_sn_map_"
                            f"{plot_name}.npz",
                            ccf_tot_sn=ccf_tot_sn, v_rest=v_rest,
                            kp_range=_kp_range,
                            sysrem_opt=bool(_opt_needs_injection),
                            order_indices=np.asarray(order_selection),
                            n_components_per_order=(
                                np.asarray(_sysrem_passes_per_order)
                                if _sysrem_passes_per_order is not None
                                else np.array([])),
                            # v_rest is an OFFSET from the expected planet
                            # trail: a signal at the nominal (Kp, V_sys)
                            # peaks at v_rest = 0 (not at V_sys).
                            v_rest_convention="offset_from_expected_trail",
                        )
                    except Exception as _e:
                        print(f"  NOTE: S/N map save skipped ({_e})")
                    plot_Kp_Vrest(
                        replace(detection_plot, simulation_name=plot_name),
                        _kp_range,
                        ccf_tot_sn,
                        v_rest,
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    plot_1D_CCF(
                        replace(detection_plot, simulation_name=plot_name),
                        v_rest,
                        ccf_tot_sn,
                        max_kp_idx_sn,
                        max_sn,
                        _n_kp,
                        max_v_wind_sn,
                        [-100, 100],
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=_opt_needs_injection,
                    )
                    # Welch branch, use a distinct Simulation_name so the
                    # Welch map doesn't overwrite the S/N map on disk.
                    use_snr_statistic    = False
                    use_welch_statistic = True
                    (ccf_tot_sig, _, _, v_rest_ttest, max_sig,
                     max_kp_idx, max_v_wind,
                     _, _, _, _, _, _) = Welch_ttest_map(
                         ccf_values_shift,
                         v_rest,
                         _kp_range,
                         CCF_Noise=False,
                         in_trail_left_right=cfg.cross_correlation.in_trail_left_right,
                         snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                         velocity_step_kms=cfg.cross_correlation.velocity_step_kms,
                     )
                    _sim_name_b7 = plot_name
                    plot_name = f"{_sim_name_b7}_Welch"
                    # The Welch t-test map is collapsed to (n_v, n_kp) with a
                    # scalar max_kp and has no per-SYSREM-iteration axis, so it
                    # always uses the single-panel plot path (sysrem_opt=False),
                    # unlike the optimisation-aware CCF S/N map above.
                    plot_Kp_Vrest(
                        replace(detection_plot, simulation_name=plot_name),
                        _kp_range,
                        ccf_tot_sig,
                        v_rest_ttest,
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=False,
                    )
                    plot_1D_CCF(
                        replace(detection_plot, simulation_name=plot_name),
                        v_rest_ttest,
                        ccf_tot_sig,
                        max_kp_idx,
                        max_sig,
                        _n_kp,
                        max_v_wind,
                        [-100, 100],
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=False,
                    )
                    plot_name = _sim_name_b7


            # 7g, Injection-based per-order SYSREM iteration selection
            # (Maximum / Max_Diff only, DeltaSigma selects per-order in
            # prepare.py and does not need post-hoc CCF maximisation).
            if _opt_needs_injection:
                sysrem_it_opt = get_SYSREM_its_ordbyord(
                    _ccf_all_orders,
                    v_rest,
                    with_signal,
                    phase,
                    berv,
                    planet.systemic_velocity_kms,
                    _pix_lr_b7,
                    _ccf_v_step,
                    _v_ccf,
                    n_orders=analysis_order_count,
                    n_nights=_nn,
                    sysrem_iterations=_si,
                    injection_velocity_kms=cfg.pipeline.kp_vrest_injection,
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
                        with_signal,
                        v_rest,
                        _v_ccf,
                        _kp_range,
                        phase,
                        planet.systemic_velocity_kms,
                        berv,
                        _pix_lr_b7,
                        _ccf_v_step,
                        _ccf_opt,
                        sysrem_opt=False,
                        significant_eccentricity=cfg.observation.significant_eccentricity,
                        eccentricity=planet.eccentricity,
                        argument_of_periastron_deg=planet.argument_of_periastron_deg,
                        sysrem_iterations=_si,
                    )
                    _ctot_opt = np.sum(_cvs_opt, 1)
                    (ccf_tot_sig, max_sig, max_kp_idx,
                     max_v_wind, _) = get_max_CCF_peak(
                         _ctot_opt,
                         v_rest,
                         _kp_range,
                         b=None,
                         stats=None,
                         sysrem_opt=False,
                         CCF_Noise=False,
                         snr_exclude_kms=cfg.cross_correlation.snr_exclude_kms,
                         sysrem_iterations=_si,
                         exclude_around=("point" if cfg.pipeline.name == "Cheverall26" else "peak"),
                         noise_source=cfg.cross_correlation.snr_noise_source,
                         noise_kp_exclude_kms=cfg.cross_correlation.snr_noise_kp_exclude_kms,
                     )
                    plot_Kp_Vrest(
                        replace(detection_plot, simulation_name=plot_name),
                        _kp_range,
                        ccf_tot_sig,
                        v_rest,
                        show_plot=False,
                        save_plot=True,
                        sysrem_opt=False,
                    )
                    plot_1D_CCF(
                        replace(detection_plot, simulation_name=plot_name),
                        v_rest,
                        ccf_tot_sig,
                        max_kp_idx,
                        max_sig,
                        _n_kp,
                        max_v_wind,
                        [-100, 100],
                        show_plot=False,
                        save_plot=True,
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
                        corner_name = (
                            f"{sim_name}{name_suffix}" if name_suffix else plot_name
                        )
                        plot_stats(
                            stats_arr,
                            kp_lim_inf=-350,
                            kp_lim_sup=350,
                            kp_step=175,
                            vrest_lim_inf=-100,
                            vrest_lim_sup=100,
                            vrest_step=50,
                            sn_lim_inf=1,
                            sn_lim_sup=13,
                            sn_lim_step=2,
                            binwidth_sn=0.5,
                            binwidth_kp=20,
                            binwidth_v_rest=5,
                            significance_metric=sig_map,
                            v_rest=v_rest,
                            auto_lims=True,
                            mark_true_values=True,
                            show_dist_CC_values=False,
                            show_plot=False,
                            save_plot=True,
                            CCF_Noise=False,
                            reference_kp_kms=planet.kp_kms,
                            reference_vrest_kms=pm.wind_velocity_kms,
                            plots_dir=str(dirs["plots"]) + "/",
                            simulation_name=corner_name,
                            kp_max_kms=cfg.cross_correlation.kp_max_kms,
                            velocity_step_kms=cfg.cross_correlation.velocity_step_kms,
                            systemic_velocity_kms=planet.systemic_velocity_kms,
                        )
                        print(f"  [stats] Corner plot saved: "
                              f"Corner_plot_{corner_name}.png")
                    except Exception as _e:
                        print(f"  [stats] Could not save corner plot: {_e}")

                if not cfg.cross_correlation.all_significance_metrics:
                    (_, ccf_tot_sn_stat, _, _, stats,
                     stats_tvalue, stats_pvalue,
                     stats_planet_pos, stats_planet_area,
                     _, _, _, _,
                     _, _, shuffled_nights, v_rest_sigma) = (
                        statistical_study(
                            replace(
                                night_study,
                                cross_correlation=replace(
                                    cfg.cross_correlation,
                                    ccf_snr=use_snr_statistic,
                                    welch_ttest=use_welch_statistic,
                                ),
                            ),
                            _ccf_v_step,
                            _ccf_all_orders,
                            _kp_range,
                            phase,
                            _v_ccf,
                            v_rest,
                            with_signal,
                            _pix_lr_b7,
                            sysrem_it_opt,
                            _ccf_iterations,
                            in_trail_pix=cfg.cross_correlation.in_trail_left_right,
                            auto_lims=True,
                            input_stats=None,
                            verbose=True,
                            show_plot=False,
                            save_plot=True,
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
                            # in-run by the 7d.2 block (single tested source).
                            # only recompute as a fallback if unavailable.
                            if _chev_tim_pos is not None:
                                _tim_pos, _tim_max = _chev_tim_pos, _chev_tim_max
                            else:
                                _excl = float((cfg.cross_correlation.snr_exclude_kms))
                                _tim_pos, _tim_max = chev26_timing_stats(
                                    _ccf_all_orders, _v_ccf, phase, berv,
                                    planet.kp_kms, planet.systemic_velocity_kms, _nn,
                                    n_in_cross=14, exclude=_excl,
                                    vmax_noise=cfg.cross_correlation.noise_velocity_max_kms)
                                save_compressed(
                                    _sd_b7, sim_name,
                                    {'stats_timing_pos': _tim_pos,
                                     'stats_timing_max': _tim_max})
                            # Metric 1, Kp-Vsys S/N at the planet position:
                            # stats_planet_pos is (n_nights, 3) = [S/N, Kp,
                            # Vrest]. take the S/N column.
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
                                        f"{plot_name}.png")
                            chev26_fourmetric_corner(
                                _metrics4, _corner4,
                                title=f"Cheverall26 significance metrics: {sim_name}")
                            print(f"  [stats] Cheverall26 timing stats + "
                                  f"4-metric corner saved: {_corner4}")
                        except Exception as _e:
                            print(f"  [stats] Cheverall26 timing/corner skipped: {_e}")

                else:
                    # All_significance_metrics: run both S/N and Welch
                    use_snr_statistic    = True
                    use_welch_statistic = False
                    use_snr_statistic    = True
                    use_welch_statistic = False
                    (_, ccf_tot_sn_stat, _, _,
                     stats_sn, _, _,
                     stats_planet_pos_sn, stats_planet_area_sn,
                     _, _, _, _,
                     _, _, shuffled_nights_sn, _) = (
                        statistical_study(
                            replace(
                                night_study,
                                cross_correlation=replace(
                                    cfg.cross_correlation,
                                    ccf_snr=use_snr_statistic,
                                    welch_ttest=use_welch_statistic,
                                ),
                            ),
                            _ccf_v_step,
                            _ccf_all_orders,
                            _kp_range,
                            phase,
                            _v_ccf,
                            v_rest,
                            with_signal,
                            _pix_lr_b7,
                            sysrem_it_opt,
                            _ccf_iterations,
                            in_trail_pix=cfg.cross_correlation.in_trail_left_right,
                            auto_lims=True,
                            input_stats=None,
                            verbose=True,
                            show_plot=False,
                            save_plot=True,
                            CCF_Noise=False,
                        )
                    )
                    use_snr_statistic    = False
                    use_welch_statistic = True
                    (_, _, _, _,
                     stats_sig, _, _,
                     stats_planet_pos_sig, stats_planet_area_sig,
                     _, _, _, _,
                     _, _, shuffled_nights_sig, v_rest_sigma) = (
                        statistical_study(
                            replace(
                                night_study,
                                cross_correlation=replace(
                                    cfg.cross_correlation,
                                    ccf_snr=use_snr_statistic,
                                    welch_ttest=use_welch_statistic,
                                ),
                            ),
                            _ccf_v_step,
                            _ccf_all_orders,
                            _kp_range,
                            phase,
                            _v_ccf,
                            v_rest,
                            with_signal,
                            _pix_lr_b7,
                            sysrem_it_opt,
                            _ccf_iterations,
                            in_trail_pix=cfg.cross_correlation.in_trail_left_right,
                            auto_lims=True,
                            input_stats=None,
                            verbose=True,
                            show_plot=False,
                            save_plot=True,
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
                    # Welch statistics share the single matrices/ directory. The
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
            # * Velocity-range study
            #   → requires iterating ccf_v_step grid. skipped pending
            #     dedicated analysis module
            # * Noise-only study
            #   → requires gauss_noise per order which is not yet stored
            #     in per_order_results. skipped pending Block 5 update
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

        # Validate the likelihood against its sampled parameter space.
        # Each beta-enabled parameter space samples one global uncertainty
        # scale. Its likelihood must include the beta-dependent normalization.
        _VALID_LL_FOR_DIM = {
            "1D":          ("BL19", "Blain24"),
            "1D_Gibson22":      ("Gibson22",),
            "1D_Blain24_beta": ("Blain24_beta",),
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
                    "Choose the parameter space paired with the selected likelihood."
                )

        # -- Parameter labels per Ret_dim --
        _RET_PARAM_LABELS_B9 = {
            "1D":   ["log(X$_{H_2O}$)", "$K_P$", "T$_{equ}$", "V$_{rest}$"],
            "1D_Gibson22": ["log(X$_{H_2O}$)", "$K_P$", "T$_{equ}$", "V$_{rest}$",
                       r"$\beta$"],
            "1D_Blain24_beta": ["log(X$_{H_2O}$)", "$K_P$", "T$_{equ}$",
                                "V$_{rest}$", r"$\beta$"],
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
            "1D_Blain24_beta": [(-8., 0.), (85., 200.), (400., 1500.),
                                (-25., 25.), (0.01, 100.)],
            "1D_CtoO_met": [(0., 2.), (-2.5, 2.5)],
            "2D":          [(-8., 0.), (-8., 0.), (85., 200.), (400., 1500.),
                            (400., 1500.), (-20., 20.)],
            # K_p, V_rest, alpha  (alpha prior includes 0 so the posterior can
            # approach the null. config overrides for the target).
            "1D_alpha":    [(40., 100.), (-25., 25.), (0., 5.)],
            # K_p, V_rest, alpha, beta  (beta = Gibson22 noise-rescaling of the
            # propagated σ. config overrides alpha for model/null).
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

        # Detection above searches template similarity in velocity space.
        # Retrieval below evaluates atmospheric/orbital parameter proposals
        # against the prepared data, using the selected likelihood and priors.
        # The enable flag gates expensive sampling. setup labels above it are
        # also constructed when sampling is disabled.
        if cfg.retrieval.enabled:
            if _dn:
                raise ValueError(
                    "Atmospheric retrieval requires a common observing sequence. "
                    "Set observation.different_nights=false."
                )
            import copy as _copy_b9
            import json as _json_b9
            # Set the signal amplitude used inside each retrieval likelihood.
            # Trial spectra use rt_inputs for the system quantities and
            # cfg.atmosphere for the atmospheric regions and chemistry.
            # Select species in the same order as the abundances constructed by each
            # likelihood. The species list and abundance array must align.
            _ret_dim_b9 = cfg.retrieval.dimensionality
            if _ret_dim_b9 in ("1D", "1D_Gibson22", "1D_Blain24_beta"):
                # Single-species abundance retrieval: honour the species
                # named in retrieval.atmosphere when provided (H2/He filler
                # + one line species, matching the abundances array built in
                # the likelihood). The default line species is H2O.
                if (cfg.retrieval.atmosphere is not None
                        and cfg.retrieval.atmosphere.species):
                    retrieval_species = [
                        "H2", "He", cfg.retrieval.atmosphere.species[-1]]
                else:
                    retrieval_species = ["H2", "He", "H2O"]
            elif _ret_dim_b9 in ("1D_alpha", "1D_alpha_Gibson22"):
                # Amplitude scaling: nominal model uses the config's species
                # opacity (the same line list as the CCF template).
                retrieval_species = [
                    "H2", "He", cfg.atmosphere.ccf_template.species[-1]]
            elif _ret_dim_b9 == "1D_CtoO_met":
                retrieval_species = ["H2", "He"]
            elif _ret_dim_b9 in ("1D_extended", "1D_extended_fast"):
                retrieval_species = [
                    "H2", "He", "H2O", "CH4", "NH3", "CO", "CO2", "HCN"]
            elif _ret_dim_b9 == "2D":
                retrieval_species = ["H2", "He", "H2O"]

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

            prepare_retrieval_template = True

            # ------------------------------------------------------------------
            # 9a, Build retrieval matrices from per_order_results (in-memory)
            #      Read the order products held in memory.
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
            #   retrieval_orders[hh]
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
                retrieval_orders = order_selection[
                    np.asarray(_good_h_b9, dtype=int)]
            else:
                _good_h_b9 = list(range(_n_orders_b9))

            # Retrieval arrays and opacity objects use the retained order count.
            analysis_order_count = _n_orders_b9

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
            _mrp_b9 = getattr(cfg.retrieval, "model_reprocessing", "bare")
            if _mrp_b9 != "bare" and cfg.retrieval.retrieval_choice == 6:
                raise NotImplementedError(
                    f"retrieval.model_reprocessing='{_mrp_b9}' is not "
                    "implemented for time-resolved retrievals "
                    "(retrieval_choice=6).")
            _U_b9 = None
            if (cfg.pipeline.name in ["ASL19", "Gibson22"]
                    or _mrp_b9 == "projector"):
                if per_order_results[0].get("U_sysrem") is not None:
                    _U_b9 = np.stack(
                        [per_order_results[h]["U_sysrem"]
                         for h in _good_h_b9])
            # Empirical removed-structure matrices for the "embedded"
            # model-reprocessing mode.
            _cor_b9 = None
            if (_mrp_b9 == "embedded"
                    and per_order_results[0].get("cor") is not None):
                _cor_b9 = np.stack(
                    [per_order_results[h]["cor"] for h in _good_h_b9])

            # ------------------------------------------------------------------
            # 9b, base directory
            # ------------------------------------------------------------------
            # Block 8 saved masks and matrices directly into _base_dir_b8.
            # Block 9 must load from the same directory.
            _base_dir_b9 = _base_dir_b8
            _sfx9 = plot_name

            # ------------------------------------------------------------------
            # 9c, Night selection (lines 5757-5828)
            # ------------------------------------------------------------------
            _night_min_b9 = int(_night_min)
            _night_max_b9 = int(_night_max)

            if cfg.retrieval.retrieval_choice == 1:
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
            elif cfg.retrieval.retrieval_choice == 2:
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
            elif cfg.retrieval.retrieval_choice == 3:
                print("Performing retrieval on night_max and night_min")
                _sel9 = [_night_max_b9, _night_min_b9]
                _mat_res_ret   = mat_res_b9[:,  _sel9, :, :]
                _propag_ret    = propag_b9[:,   _sel9, :, :]
                _mat_star_ret  = mat_star_b9[:, _sel9, :, :]
                _std_noise_ret = std_noise_b9[:,_sel9, :, :]
                retrieved_nights = 2
            elif cfg.retrieval.retrieval_choice in [4, 5, 6]:
                if cfg.retrieval.retrieval_choice == 4:
                    print("Combining all retrievals")
                _mat_res_ret   = np.copy(mat_res_b9)
                _propag_ret    = np.copy(propag_b9)
                _mat_star_ret  = np.copy(mat_star_b9)
                _std_noise_ret = np.copy(std_noise_b9)
                retrieved_nights = _n_nights_b9

            # ------------------------------------------------------------------
            # 9d, Wave and mask loading (lines 5838-5858)
            # ------------------------------------------------------------------
            wave_ret = wave_star[retrieval_orders, :].reshape(
                _n_px_b9 * len(retrieval_orders))

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
            # built from the retained orders in 9a. retrieval_orders uses the
            # same selection, so wave_ret (built in 9d) follows those orders.
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

            # ------------------------------------------------------------------
            # 9f, Different_nights deep copies (lines 6008-6021)
            # ------------------------------------------------------------------

            retrieval_preparation = PreparationInputs(
                pipeline=cfg.pipeline,
                tellurics=cfg.tellurics,
                use_real_data=cfg.observation.use_real_data,
                sysrem_iterations=_si,
            )

            # ------------------------------------------------------------------
            # 9g, SYSREM filtering projector (lines 6024-6035)
            # ------------------------------------------------------------------
            P = None
            if ((cfg.pipeline.name in ["ASL19", "Gibson22"]
                 or (getattr(cfg.retrieval, 'model_reprocessing', 'bare')) == "projector")
                    and _U_b9 is not None):
                P = SYSREM_filtering_projector(
                    n_spectra,
                    propag_b9,
                    _U_b9,
                    different_nights=_dn,
                    n_orders=n_orders,
                    n_nights=_nn,
                    sysrem_iterations=_si,
                )
            if ((getattr(cfg.retrieval, 'model_reprocessing', 'bare')) == "projector"
                    and P is None):
                raise RuntimeError(
                    "model_reprocessing='projector' requires the detrending "
                    "temporal modes (U), which this run did not produce.")
            if ((getattr(cfg.retrieval, 'model_reprocessing', 'bare')) == "embedded"
                    and _cor_b9 is None):
                raise RuntimeError(
                    "model_reprocessing='embedded' requires the removed-"
                    "structure matrices (cor), which this run did not "
                    "produce.")

            # ------------------------------------------------------------------
            # 9h, Sampler definitions (lines 6052-6087)
            # ------------------------------------------------------------------
            if sampling_method == "Nested_sampling":
                def prior(cube, ndim, nparams):
                    # Use user-supplied prior_bounds from config if available,
                    # falling back to the hardcoded defaults otherwise.
                    _cfg_bounds = (cfg.retrieval.prior_bounds)
                    _bds = (_cfg_bounds.get(cfg.retrieval.dimensionality)
                            or _PRIOR_BOUNDS_B9[cfg.retrieval.dimensionality])
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
            if cfg.retrieval.dimensionality in [
                "1D", "1D_CtoO_met", "1D_Gibson22", "1D_Blain24_beta",
                "1D_alpha", "1D_alpha_Gibson22"
            ]:
                retrieval_limb_asymmetries = False

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
                    if cfg.retrieval.dimensionality == "1D":
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]
                    elif cfg.retrieval.dimensionality == "1D_CtoO_met":
                        c_to_o = cube[0]; met = cube[1]
                    elif cfg.retrieval.dimensionality in ("1D_Gibson22", "1D_Blain24_beta"):
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]; beta = cube[4]
                    elif cfg.retrieval.dimensionality == "1D_alpha":
                        # Amplitude scaling: free K_p, V_rest, α on a fixed
                        # nominal model (no pRT / abundance / T sampled).
                        K_p = cube[0]; v_wind = cube[1]; alpha = cube[2]
                    elif cfg.retrieval.dimensionality == "1D_alpha_Gibson22":
                        # Same fixed-model α detection + Gibson22 β noise-rescaling.
                        K_p = cube[0]; v_wind = cube[1]
                        alpha = cube[2]; beta = cube[3]

                    log_likelihood = 0.

                    # ---- abundances ----
                    if cfg.retrieval.dimensionality in ["1D", "1D_Gibson22", "1D_Blain24_beta"]:
                        abundances = np.asarray(
                            [pm.vmr[0], pm.vmr[1], 10.**log10_X1])
                    else:
                        abundances = np.asarray(
                            [pm.vmr[0], pm.vmr[1]])

                    # ---- model matrix ----
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
                        if cfg.retrieval.dimensionality in ("1D_alpha", "1D_alpha_Gibson22"):
                            # FIXED nominal model (computed once, pre-sampler).
                            # α scales its depth via Scale_inj below. No pRT.
                            wave_pRT_r = _ctx["nominal_wave"][hh]
                            syn_spec_r = _ctx["nominal_spec"][hh]
                        elif _emulator_ll is not None:
                            wave_pRT_r, syn_spec_r = _emulator_ll.predict(
                                T_eq=T_equ,
                                log10_x_h2o=log10_X1,
                            )
                        elif cfg.retrieval.dimensionality != "1D_CtoO_met":
                            wave_pRT_r, syn_spec_r, *_ = call_pRT(
                                rt_inputs, p_ret, atm_r,
                                retrieval_species,
                                abundances, retrieval_atmosphere_config.mean_molecular_weight,
                                retrieval_atmosphere_config.reference_pressure_bar,
                                retrieval_atmosphere_config.isothermal, T_equ,
                                retrieval_atmosphere_config.two_point,
                                retrieval_atmosphere_config.two_point_pressures_bar,
                                retrieval_atmosphere_config.two_point_temperatures_K,
                                retrieval_atmosphere_config.kappa_ir,
                                retrieval_atmosphere_config.gamma_guillot, T_equ,
                                None, None, use_easyCHEM=False,
                            )
                        else:  # 1D_CtoO_met
                            wave_pRT_r, syn_spec_r, *_ = call_pRT(
                                rt_inputs, p_ret, atm_r,
                                retrieval_species,
                                abundances, retrieval_atmosphere_config.mean_molecular_weight,
                                retrieval_atmosphere_config.reference_pressure_bar,
                                retrieval_atmosphere_config.isothermal,
                                pm.equilibrium_temperature_K,
                                retrieval_atmosphere_config.two_point,
                                retrieval_atmosphere_config.two_point_pressures_bar,
                                retrieval_atmosphere_config.two_point_temperatures_K,
                                retrieval_atmosphere_config.kappa_ir,
                                retrieval_atmosphere_config.gamma_guillot, pm.equilibrium_temperature_K,
                                metallicity=met, C_to_O=c_to_o,
                                use_easyCHEM=True, P_cloud=None,
                                easychem_CtoO_ret=True,
                            )

                        # v_planet
                        if cfg.retrieval.dimensionality != "1D_CtoO_met":
                            if not cfg.observation.significant_eccentricity:
                                v_planet_r = get_V(
                                    K_p, _ph_ll, _berv_ll,
                                    planet.systemic_velocity_kms, v_wind)
                            else:
                                v_planet_r = get_V_eccentric(
                                    K_p, _ph_ll,
                                    planet.eccentricity,
                                    planet.argument_of_periastron_deg,
                                    _berv_ll, planet.systemic_velocity_kms, v_wind)
                        elif cfg.retrieval.dimensionality == "1D_CtoO_met":
                            if not cfg.observation.significant_eccentricity:
                                v_planet_r = get_V(
                                    planet.kp_kms, _ph_ll, _berv_ll,
                                    planet.systemic_velocity_kms, 0.)
                            else:
                                v_planet_r = get_V_eccentric(
                                    planet.kp_kms, _ph_ll,
                                    planet.eccentricity,
                                    planet.argument_of_periastron_deg,
                                    _berv_ll, planet.systemic_velocity_kms, 0.)
                        else:
                            if not cfg.observation.significant_eccentricity:
                                v_planet_r = get_V(
                                    planet.kp_kms, _ph_ll, _berv_ll,
                                    planet.systemic_velocity_kms, pm.wind_velocity_kms)
                            else:
                                v_planet_r = get_V_eccentric(
                                    planet.kp_kms, _ph_ll,
                                    planet.eccentricity,
                                    planet.argument_of_periastron_deg,
                                    _berv_ll, planet.systemic_velocity_kms,
                                    pm.wind_velocity_kms)

                        # 1D_alpha[_Gibson22]: α scales the model depth via
                        # Scale_inj. All other modes keep the config value.
                        _model_signal_scale = (
                            alpha if cfg.retrieval.dimensionality in
                            ("1D_alpha", "1D_alpha_Gibson22")
                            else cfg.pipeline.inject_scale_factor
                        )
                        model_mat[hh], _ = spec_to_mat_fraction(
                            v_planet_r,
                            wave_star[retrieval_orders[hh], :],
                            wave_pRT_r,
                            syn_spec_r,
                            _mat_star_ret[_night_index][:, :n_pixels],
                            _with_ll,
                            _wout_ll,
                            _frac_ll,
                            event_type=cfg.observation.event_type,
                            signal_scale=_model_signal_scale,
                            limb_asymmetries=retrieval_limb_asymmetries,
                            limb_divisions=cfg.atmosphere.limb_divisions,
                            resolving_power=inst.res,
                            include_star=False,
                        )

                        if prepare_retrieval_template:
                            if not cfg.pipeline.sysrem_robust_halt:
                                _sysrem_pass = None
                            _mrp_ll = (getattr(cfg.retrieval, 'model_reprocessing', 'bare'))
                            if (cfg.pipeline.name == "Gibson22"
                                    or _mrp_ll == "projector"):
                                # Linear filter: project the data-derived
                                # temporal modes out of the model only.
                                # the template carries no background
                                # structure.
                                _usp_i9 = np.where(
                                    useful_spectral_points_inter_aux[
                                        _night_index, hh, :])[0]
                                model_mat_prepared[hh] = \
                                    filter_model_singleorder(
                                        P[hh, _night_index, :, :],
                                        model_mat[hh], _usp_i9,
                                    )
                                if (_mrp_ll == "projector"
                                        and cfg.pipeline.name
                                            != "Gibson22"):
                                    # Subtraction-convention pipelines
                                    # compare zero-centred residuals to
                                    # the template, so filter the model
                                    # DEVIATION: by linearity,
                                    # filt(m-1) = filt(m) - filt(1).
                                    # Without this, the filtered constant
                                    # level survives as a spurious
                                    # time-structured baseline.
                                    model_mat_prepared[hh] -= \
                                        filter_model_singleorder(
                                            P[hh, _night_index, :, :],
                                            np.ones_like(model_mat[hh]),
                                            _usp_i9,
                                        )
                            else:
                                # "bare": re-run the preparing pipeline on
                                # the model matrix alone.  "embedded":
                                # multiply the model into the structure the
                                # detrending removed from the data (the
                                # empirical telluric/stellar template) and
                                # re-run the pipeline on the combination.
                                _mm_in = (
                                    _cor_b9[hh, :_n_sp_ll, :]
                                    * model_mat[hh]
                                    if _mrp_ll == "embedded"
                                    else model_mat[hh])
                                model_mat_prepared[hh] = \
                                    prepare_observation(
                                        retrieval_preparation,
                                        _mm_in,
                                        _std_noise_ret[_night_index, :_n_sp_ll, hh * _n_px_b9 : (hh + 1) * _n_px_b9],
                                        wave_star[retrieval_orders[hh], :],
                                        np.where(useful_spectral_points_snr_aux[_night_index, hh, :])[0],
                                        np.where(mask_snr_ret_aux[_night_index, hh, :])[0],
                                        _frac_ll,
                                        _ph_ll,
                                        _wout_ll,
                                        _sysrem_pass,
                                        None,
                                        tell_mask_threshold_Blain24=0.8,
                                        max_fit_BL19=False,
                                        sysrem_division=False,
                                        masks=False,
                                        correct_uncertainties=False,
                                        retrieval=True,
                                        mask_inter_retrieval=np.where(mask_inter_ret_aux[_night_index, hh, :])[0],
                                        useful_spectral_points_inter_retrieval=np.where(
                                            useful_spectral_points_inter_aux[_night_index, hh, :]
                                        )[0],
                                    )
                        else:
                            model_mat_prepared[hh] = np.copy(model_mat[hh])

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
                    if cfg.retrieval.log_likelihood == "BL19":
                        # The model must carry the same preparation as the data
                        # (as the CCF and the Blain24/Gibson22 loglikes do), so
                        # compare the prepared data with model_mat_prepared.
                        for n in with_signal_ret:
                            _d = (_mat_res_ret[_night_index, n,
                                    useful_spectral_points_ret[
                                        _night_index, :]]
                                  - np.mean(_mat_res_ret[_night_index, n,
                                    useful_spectral_points_ret[
                                        _night_index, :]]))
                            _tpl = (model_mat_prepared[n,
                                    useful_spectral_points_ret[
                                        _night_index, :]]
                                    - np.mean(model_mat_prepared[n,
                                    useful_spectral_points_ret[
                                        _night_index, :]]))
                            sf2 = np.mean(_d**2)
                            sg2 = np.mean(_tpl**2)
                            R   = np.mean(_d * _tpl)
                            log_likelihood += (
                                -(len(_d) / 2.)
                                * np.log(sf2 - 2.*R + sg2))

                    elif cfg.retrieval.log_likelihood in ("Blain24", "Blain24_beta"):
                        _up9 = useful_spectral_points_ret[_night_index, :]
                        # σ error model (Cheverall26 pipeline): MAD-of-residuals
                        # (paper-faithful: σ_j = 1.4826·MAD over in-transit
                        # exposures per channel, time-independent) vs the
                        # default pipeline-propagated σ.  Cached per night.
                        _em9 = (getattr(cfg.retrieval, 'error_model', 'propagated'))
                        _chev9 = (cfg.pipeline.name == "Cheverall26")
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

                        if cfg.retrieval.log_likelihood == "Blain24_beta":
                            # P-19 adds one global uncertainty scale to the same
                            # prepared model, residuals and fixed noise estimates.
                            beta_low, beta_high = _PRIOR_BOUNDS_B9[
                                cfg.retrieval.dimensionality][-1]
                            if not (beta_low <= beta <= beta_high) or beta <= 0:
                                return -np.inf
                            n_data = len(with_signal_ret) * np.count_nonzero(_up9)
                            log_likelihood = (log_likelihood / beta**2
                                              - n_data * np.log(beta))

                    elif cfg.retrieval.log_likelihood == "Gibson22":
                        # Enforce beta prior (reads from prior_bounds config).
                        # beta is the last sampled parameter: index 4 for 1D_Gibson22,
                        # index 3 for 1D_alpha_Gibson22.
                        _beta_lo, _beta_hi = _PRIOR_BOUNDS_B9[
                            cfg.retrieval.dimensionality][-1]
                        if not (_beta_lo <= beta <= _beta_hi):
                            return -np.inf
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
                    if cfg.retrieval.dimensionality == "1D":
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]
                    elif cfg.retrieval.dimensionality == "1D_CtoO_met":
                        c_to_o = cube[0]; met = cube[1]
                    elif cfg.retrieval.dimensionality in ("1D_Gibson22", "1D_Blain24_beta"):
                        log10_X1 = cube[0]; K_p = cube[1]
                        T_equ = cube[2]; v_wind = cube[3]; beta = cube[4]
                    elif cfg.retrieval.dimensionality == "1D_alpha":
                        K_p = cube[0]; v_wind = cube[1]; alpha = cube[2]
                    elif cfg.retrieval.dimensionality == "1D_alpha_Gibson22":
                        K_p = cube[0]; v_wind = cube[1]
                        alpha = cube[2]; beta = cube[3]

                    log_likelihood = 0.

                    if cfg.retrieval.dimensionality in ["1D", "1D_Gibson22", "1D_Blain24_beta"]:
                        abundances = np.asarray(
                            [pm.vmr[0], pm.vmr[1], 10.**log10_X1])
                    else:
                        abundances = np.asarray(
                            [pm.vmr[0], pm.vmr[1]])

                    for jj in range(retrieved_nights):
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
                            if cfg.retrieval.dimensionality != "1D_CtoO_met":
                                wave_pRT_r, syn_spec_r, *_ = \
                                    call_pRT(
                                        rt_inputs, p_ret, atm_r,
                                        retrieval_species,
                                        abundances, retrieval_atmosphere_config.mean_molecular_weight,
                                        retrieval_atmosphere_config.reference_pressure_bar,
                                        retrieval_atmosphere_config.isothermal, T_equ,
                                        retrieval_atmosphere_config.two_point,
                                        retrieval_atmosphere_config.two_point_pressures_bar,
                                        retrieval_atmosphere_config.two_point_temperatures_K,
                                        retrieval_atmosphere_config.kappa_ir,
                                        retrieval_atmosphere_config.gamma_guillot, T_equ,
                                        None, None, use_easyCHEM=False,
                                    )
                            else:
                                wave_pRT_r, syn_spec_r, *_ = \
                                    call_pRT(
                                        rt_inputs, p_ret, atm_r,
                                        retrieval_species,
                                        abundances, retrieval_atmosphere_config.mean_molecular_weight,
                                        retrieval_atmosphere_config.reference_pressure_bar,
                                        retrieval_atmosphere_config.isothermal,
                                        pm.equilibrium_temperature_K,
                                        retrieval_atmosphere_config.two_point,
                                        retrieval_atmosphere_config.two_point_pressures_bar,
                                        retrieval_atmosphere_config.two_point_temperatures_K,
                                        retrieval_atmosphere_config.kappa_ir,
                                        retrieval_atmosphere_config.gamma_guillot,
                                        pm.equilibrium_temperature_K,
                                        metallicity=log10_Z, C_to_O=0.55,
                                        use_easyCHEM=True,
                                        P_cloud=10.**log10_P,
                                    )

                            if cfg.retrieval.dimensionality != "1D_CtoO_met":
                                v_planet_r = get_V(
                                    K_p, _ph_c, _berv_c,
                                    planet.systemic_velocity_kms, v_wind)
                            else:
                                v_planet_r = get_V(
                                    planet.kp_kms, _ph_c, _berv_c,
                                    planet.systemic_velocity_kms, pm.wind_velocity_kms)

                            model_mat[hh], _ = spec_to_mat_fraction(
                                v_planet_r,
                                wave_star[retrieval_orders[hh], :],
                                wave_pRT_r,
                                syn_spec_r,
                                _mat_star_ret[jj][:, :n_pixels],
                                _with_c,
                                _wout_c,
                                _frac_c,
                                event_type=cfg.observation.event_type,
                                signal_scale=cfg.pipeline.inject_scale_factor,
                                limb_asymmetries=retrieval_limb_asymmetries,
                                limb_divisions=cfg.atmosphere.limb_divisions,
                                resolving_power=inst.res,
                                include_star=False,
                            )
                            if prepare_retrieval_template:
                                if not cfg.pipeline.sysrem_robust_halt:
                                    _sysrem_pass = None
                                if cfg.pipeline.name != "Gibson22":
                                    model_mat_prepared[hh] = \
                                        prepare_observation(
                                            retrieval_preparation,
                                            model_mat[hh],
                                            _std_noise_ret[jj, :_n_sp_c, :],
                                            wave_star[retrieval_orders[hh], :],
                                            np.where(useful_spectral_points_snr_aux[jj, hh, :])[0],
                                            np.where(mask_snr_ret_aux[jj, hh, :])[0],
                                            _frac_c,
                                            _ph_c,
                                            _wout_c,
                                            _sysrem_pass,
                                            None,
                                            tell_mask_threshold_Blain24=0.8,
                                            max_fit_BL19=False,
                                            sysrem_division=False,
                                            masks=False,
                                            correct_uncertainties=False,
                                            retrieval=True,
                                            mask_inter_retrieval=np.where(mask_inter_ret_aux[jj, hh, :])[0],
                                            useful_spectral_points_inter_retrieval=np.where(
                                                useful_spectral_points_inter_aux[jj, hh, :]
                                            )[0],
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
                        if cfg.retrieval.log_likelihood == "BL19":
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

                        elif cfg.retrieval.log_likelihood == "Blain24":
                            for n in with_signal_ret:
                                log_likelihood += -0.5 * np.sum(
                                    ((_mat_res_ret[_night_index, n,
                                       useful_spectral_points_ret[_night_index,:]]
                                      - model_mat_prepared[n,
                                       useful_spectral_points_ret[_night_index,:]])
                                     / _propag_ret[_night_index, n,
                                       useful_spectral_points_ret[_night_index,:]]
                                     )**2)

                        elif cfg.retrieval.log_likelihood in ("Gibson22", "Blain24_beta"):
                            _beta_lo2, _beta_hi2 = _PRIOR_BOUNDS_B9[
                                cfg.retrieval.dimensionality][-1]
                            if not (_beta_lo2 <= beta <= _beta_hi2):
                                return -np.inf
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
                if cfg.retrieval.retrieval_choice != 4:
                    loglike = _loglike_1d_single
                else:
                    loglike = _loglike_1d_combined

                # MCMC log_prior / log_prob (need loglike defined first)
                if sampling_method == "MCMC":
                    _cfg_bounds_mc = (cfg.retrieval.prior_bounds)
                    _bds_mc = (_cfg_bounds_mc.get(cfg.retrieval.dimensionality)
                               or _PRIOR_BOUNDS_B9[cfg.retrieval.dimensionality])
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
                    nwalkers_b9 = cfg.retrieval.n_walkers
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
                        _ll_bounds = (cfg.retrieval.prior_bounds)
                        _ll_bds    = (_ll_bounds.get(cfg.retrieval.dimensionality)
                                      or _PRIOR_BOUNDS_B9[cfg.retrieval.dimensionality])
                        _ll_n      = len(_ll_bds)
                        _ll_prior  = np.array(_ll_bds, dtype=np.float32).T
                        _ll_save   = _os9.path.join(
                            (str(dirs['plots']) + '/'), "likelihood_emulator")

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

                    if sampling_method == "Nested_sampling":
                        _pymn.run(
                            _ll_active, prior,
                            n_dims=n_params,
                            outputfiles_basename=_out_pfx,
                            resume=False, verbose=True,
                            evidence_tolerance=0.5,
                            sampling_efficiency=0.8,
                            n_iter_before_update=100,
                            const_efficiency_mode=cfg.retrieval.constant_efficiency_mode,
                            n_live_points=cfg.retrieval.live_points,
                            max_iter=0,
                        )
                        # ── Analyzer: try the pymultinest Analyzer first.
                        # fall back to reading the plain-text MultiNest
                        # output files directly when the Analyzer's loadtxt
                        # fails on Fortran fixed-point exponents (e.g.
                        # a missing exponent marker: "1.23-313" represents "1.23E-313").
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
                            # post_equal_weights has n_params cols + logL col.
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
                                (False)
                                or (cfg.atmosphere.limb_asymmetries
                                    and cfg.retrieval.dimensionality not in ("2D",))
                            )
                            _truths_1d9  = (None if _no_truths9 else
                                            [np.log10(pm.vmr[2]),
                                             planet.kp_kms, pm.equilibrium_temperature_K,
                                             pm.wind_velocity_kms])
                            _truths_g229 = (None if _no_truths9 else
                                            [np.log10(pm.vmr[2]),
                                             planet.kp_kms, pm.equilibrium_temperature_K,
                                             pm.wind_velocity_kms, 1.0])

                            # Zoom each axis to the posterior bulk (weighted
                            # 0.3 to 99.7 percentile with padding), so the
                            # corner is not stretched to the full prior range.
                            _d9m = _dat9[_mask9, :]
                            _w9m = _wts9[_mask9]
                            _truth_for_range = (
                                _truths_g229
                                if cfg.retrieval.dimensionality in (
                                    "1D_Gibson22", "1D_Blain24_beta")
                                else _truths_1d9)

                            def _wrange9(_col, _tv=None):
                                _o = np.argsort(_col)
                                _c = _col[_o]
                                _cw = np.cumsum(_w9m[_o])
                                _cw = _cw / _cw[-1]
                                _lo = np.interp(0.003, _cw, _c)
                                _hi = np.interp(0.997, _cw, _c)
                                if _tv is not None and np.isfinite(_tv):
                                    _lo = min(_lo, _tv)
                                    _hi = max(_hi, _tv)
                                _pad = 0.10 * (_hi - _lo) if _hi > _lo else 1.0
                                return (_lo - _pad, _hi + _pad)

                            _ranges9 = [
                                _wrange9(
                                    _d9m[:, _k],
                                    (_truth_for_range[_k]
                                     if _truth_for_range is not None
                                     and _k < len(_truth_for_range) else None))
                                for _k in range(_d9m.shape[1])]

                            # Shared "pretty" corner style (filled blue
                            # contours), matching the multi-method comparison
                            # plots.
                            _cstyle9 = dict(
                                color="#3b5ba5",
                                fill_contours=True,
                                levels=(0.68, 0.95),
                                smooth=1.0, smooth1d=1.0,
                                range=_ranges9,
                                contour_kwargs={"linewidths": 1.5},
                                hist_kwargs={"linewidth": 2})

                            if cfg.retrieval.dimensionality == "1D":
                                _fig9 = _corner.corner(
                                    _dat9[_mask9, :],
                                    weights=_wts9[_mask9],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False,
                                    title_fmt=".2E",
                                    truths=_truths_1d9,
                                    quantiles=[0.16, 0.5, 0.84],
                                    truth_color="firebrick",
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18},
                                    **_cstyle9)
                            elif cfg.retrieval.dimensionality in ["1D_Gibson22", "1D_Blain24_beta"]:
                                _fig9 = _corner.corner(
                                    _dat9[_mask9, :],
                                    weights=_wts9[_mask9],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False,
                                    title_fmt=".2E",
                                    truths=_truths_g229,
                                    quantiles=[0.16, 0.5, 0.84],
                                    truth_color="firebrick",
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18},
                                    **_cstyle9)
                            else:
                                _fig9 = _corner.corner(
                                    _dat9[_mask9, :],
                                    weights=_wts9[_mask9],
                                    show_titles=True, labels=parameters,
                                    plot_datapoints=False,
                                    title_fmt=".2E",
                                    quantiles=[0.16, 0.5, 0.84],
                                    label_kwargs={"fontsize": 18},
                                    title_kwargs={"fontsize": 18},
                                    **_cstyle9)
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

                    elif sampling_method == "MCMC":
                        if _emcee is None:
                            return None, None
                        # Serial evaluation (pool=None).
                        # ThreadPool is counterproductive here: pRT uses
                        # OpenMP internally, which fights Python threads and
                        # serialises everything with extra contention.
                        # atmosphere_ret_list is pre-loaded so each loglike
                        # call is only spectrum computation (~0.4 s). With
                        # minimum walkers (2-3× n_params) serial emcee is
                        # fast enough and gives correct autocorrelation stats.
                        _samp9 = _emcee.EnsembleSampler(
                            nwalkers_b9, ndim_b9, log_prob_b9,
                            pool=None)
                        _samp9.run_mcmc(
                            p0_b9, cfg.retrieval.n_steps, progress=True)
                        _samples9 = _samp9.get_chain(
                            discard=cfg.retrieval.burnin, thin=10, flat=True)
                        if _corner is not None:
                            import matplotlib.pyplot as _plt9
                            _PNAMES9 = {
                                "1D": [r"$\log_{10}X_1$", r"$K_p$",
                                       r"$T_{\rm eq}$", r"$v_{\rm wind}$"],
                                "1D_Gibson22": [r"$\log_{10}X_1$", r"$K_p$",
                                           r"$T_{\rm eq}$", r"$v_{\rm wind}$",
                                           r"$\beta$"],
                            }
                            _PNAMES9["1D_Blain24_beta"] = _PNAMES9["1D_Gibson22"]
                            _lbls9 = _PNAMES9.get(cfg.retrieval.dimensionality,
                                                   parameters)
                            _no_truths9 = (
                                (False)
                                or (cfg.atmosphere.limb_asymmetries
                                    and cfg.retrieval.dimensionality
                                    not in ("2D",))
                            )
                            _truths9 = None if _no_truths9 else {
                                "1D": [np.log10(pm.vmr[2]),
                                       planet.kp_kms, pm.equilibrium_temperature_K,
                                       pm.wind_velocity_kms],
                                "1D_Gibson22": [np.log10(pm.vmr[2]),
                                           planet.kp_kms, pm.equilibrium_temperature_K,
                                           pm.wind_velocity_kms, 1.0],
                                "1D_Blain24_beta": [np.log10(pm.vmr[2]),
                                           planet.kp_kms, pm.equilibrium_temperature_K,
                                           pm.wind_velocity_kms, 1.0],
                            }.get(cfg.retrieval.dimensionality)
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
                                (str(dirs['plots']) + '/'),
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
                    (str(dirs['root']) + '/').rstrip("/") + "/plots_SNR"
                    if cfg.cross_correlation.all_significance_metrics
                    else (str(dirs['plots']) + '/').rstrip("/")
                )
                os.makedirs(_base_dir_plot_b9, exist_ok=True)

                if cfg.retrieval.retrieval_choice != 6:
                    _ctx["idx_interval"] = np.arange(n_spectra)
                    _ctx["interval_tracker"] = None

                    for night_index in range(retrieved_nights):
                        _ctx["night_index"] = night_index


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
                                            line_species=retrieval_species[2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[
                                                    retrieval_orders[hh],
                                                    :].min() - 0.01,
                                                wave_star[
                                                    retrieval_orders[hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx["atmosphere_ret_list"].append(_atm9)

                        # 1D_alpha with a telluric template: the fixed nominal
                        # "model" is the Earth-transmittance depth spectrum
                        # (same construction as the CCF template in Block 5b).
                        # the sampler then treats it exactly like a planetary
                        # model (α scale, Doppler shift, lightcurve weighting,
                        # PCA reprocessing, same likelihood).  A contamination
                        # diagnostic: a model-vs-null Bayes factor >> 1 for a
                        # non-planetary template means the detection machinery
                        # responds to residual telluric structure.
                        if (cfg.retrieval.dimensionality in ("1D_alpha", "1D_alpha_Gibson22")
                                and _ctx.get("nominal_spec") is None
                                and cfg.cross_correlation.ccf_template_source
                                    == "telluric"):
                            from astropy.io import fits as _fits_t9
                            with _fits_t9.open(_tell_ref_file) as _th9:
                                _lam9 = np.asarray(
                                    _th9[1].data['lam'], float) * 1e-3
                                _trn9 = np.asarray(
                                    _th9[1].data['trans'], float)
                            try:
                                _am9 = float(np.mean(
                                    np.concatenate([np.atleast_1d(a)
                                                    for a in airmass])
                                    if isinstance(airmass, list) else airmass))
                            except Exception:
                                _am9 = 1.0
                            _ctx["nominal_wave"] = []
                            _ctx["nominal_spec"] = []
                            for hh in range(_n_orders_b9):
                                _wo9 = wave_star[
                                    retrieval_orders[hh], :]
                                _sel9 = ((_lam9 >= _wo9.min() - 0.01)
                                         & (_lam9 <= _wo9.max() + 0.01))
                                _wn9 = np.ascontiguousarray(
                                    _lam9[_sel9], dtype=np.float64)
                                _tn9 = np.clip(
                                    np.asarray(_trn9[_sel9], float),
                                    0.0, 1.0) ** _am9
                                if cfg.instrument.convolve_to_resolution:
                                    _tn9 = convolve(_wn9, _tn9, inst.res)
                                _ctx["nominal_wave"].append(_wn9)
                                _ctx["nominal_spec"].append(1.0 - _tn9)
                            print(f"  [1D_alpha] fixed nominal TELLURIC "
                                  f"template built for {_n_orders_b9} orders "
                                  f"(airmass^{_am9:.3f}; α scales it "
                                  f"in-sampler; no pRT in the loop).")

                        # 1D_alpha with a stellar template: the fixed nominal
                        # "model" is the PHOENIX stellar depth spectrum, treated
                        # exactly like a planetary model by the sampler.  A
                        # contamination diagnostic for structure at the stellar
                        # velocity (which coincides with the planet cell).
                        if (cfg.retrieval.dimensionality in ("1D_alpha", "1D_alpha_Gibson22")
                                and _ctx.get("nominal_spec") is None
                                and cfg.cross_correlation.ccf_template_source
                                    == "stellar"):
                            from astropy.io import fits as _fits_s9
                            _wv_s9 = 1.0e-4 * _fits_s9.getdata(
                                cfg.paths.phoenix_wave_file)
                            _fl_s9 = _fits_s9.getdata(cfg.paths.phoenix_flux_file)
                            _ctx["nominal_wave"] = []
                            _ctx["nominal_spec"] = []
                            for hh in range(_n_orders_b9):
                                _wo9 = wave_star[
                                    retrieval_orders[hh], :]
                                _sel9 = ((_wv_s9 >= _wo9.min() - 0.01)
                                         & (_wv_s9 <= _wo9.max() + 0.01))
                                _wn9 = np.ascontiguousarray(
                                    _wv_s9[_sel9], dtype=np.float64)
                                _sn9 = np.ascontiguousarray(
                                    _fl_s9[_sel9], dtype=np.float64)
                                if cfg.instrument.convolve_to_resolution:
                                    _sn9 = convolve(_wn9, _sn9, inst.res)
                                _sn9 = _sn9 / np.median(_sn9)
                                _ctx["nominal_wave"].append(_wn9)
                                _ctx["nominal_spec"].append(1.0 - _sn9)
                            print(f"  [1D_alpha] fixed nominal STELLAR "
                                  f"template built for {_n_orders_b9} orders "
                                  f"(α scales it in-sampler; no pRT in the "
                                  f"loop).")

                        # 1D_alpha: precompute the fixed nominal model ONCE per
                        # order (no pRT inside the sampler, α just scales its
                        # depth).  Nominal abundance/T = the CCF template.
                        if (cfg.retrieval.dimensionality in ("1D_alpha", "1D_alpha_Gibson22")
                                and _ctx.get("nominal_spec") is None
                                and _Radtrans is not None):
                            _nom_ab = np.asarray([
                                pm.vmr[0], pm.vmr[1],
                                float(cfg.atmosphere.ccf_template.mass_fractions[-1])])
                            _nom_T = float(
                                cfg.atmosphere.ccf_template.isothermal_temperature_K)
                            _ctx["nominal_wave"] = []
                            _ctx["nominal_spec"] = []
                            for hh in range(_n_orders_b9):
                                _wn, _sn, *_ = call_pRT(
                                    rt_inputs, p_ret,
                                    _ctx["atmosphere_ret_list"][hh],
                                    retrieval_species, _nom_ab,
                                    retrieval_atmosphere_config.mean_molecular_weight, retrieval_atmosphere_config.reference_pressure_bar,
                                    retrieval_atmosphere_config.isothermal, _nom_T,
                                    retrieval_atmosphere_config.two_point,
                                    retrieval_atmosphere_config.two_point_pressures_bar,
                                    retrieval_atmosphere_config.two_point_temperatures_K,
                                    retrieval_atmosphere_config.kappa_ir,
                                    retrieval_atmosphere_config.gamma_guillot, _nom_T,
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
                    _ctx["idx_interval"] = np.arange(n_spectra)

                    for night_index in range(retrieved_nights):
                        _ctx["night_index"] = night_index


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
                                            line_species=retrieval_species[2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[
                                                    retrieval_orders[hh],
                                                    :].min() - 0.01,
                                                wave_star[
                                                    retrieval_orders[hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx["atmosphere_ret_list"].append(_atm9)

                        # n_intervals
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
            elif cfg.retrieval.dimensionality == "2D":
                if cfg.retrieval.retrieval_choice != 4:
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

                        abundances_morning = np.asarray(
                            [limb_retrieval.morning_abundances[0],
                             limb_retrieval.morning_abundances[1],
                             10.**log10_X_LL])
                        limb_retrieval.morning_abundances = abundances_morning
                        abundances_evening = np.asarray(
                            [limb_retrieval.evening_abundances[0],
                             limb_retrieval.evening_abundances[1],
                             10.**log10_X_TL])
                        limb_retrieval.evening_abundances = abundances_evening

                        model_mat = np.zeros(
                            (_n_ord_sel_b9, n_spectra, n_pixels))
                        model_mat_prepared = np.zeros_like(model_mat)

                        _same_sp = (
                            _md.species[2:]
                            == _ed.species[2:])
                        if _same_sp:
                            _m_list = _atm_list
                            _e_list = _atm_list
                        else:
                            _m_list = _atm_list_m
                            _e_list = _atm_list_e


                        # Trial abundances use the configured regional temperature
                        # profiles and chemistry choices for this likelihood.
                        retrieval_atmosphere = replace(
                            cfg.atmosphere,
                            morning_day=replace(
                                cfg.atmosphere.morning_day,
                                species=_md.species,
                                mass_fractions=abundances_morning,
                            ),
                            evening_day=replace(
                                cfg.atmosphere.evening_day,
                                species=_ed.species,
                                mass_fractions=abundances_evening,
                            ),
                        )
                        for hh in range(_n_orders_b9):
                            atm_m = _m_list[hh]
                            atm_e = _e_list[hh]
                            atm_m.setup_opa_structure(p_ret)
                            atm_e.setup_opa_structure(p_ret)

                            (wave_pRT_r, syn_spec_LL, _, syn_spec_TL,
                             *_) = call_pRT_limbs(
                                rt_inputs, cfg.atmosphere, p_ret, None, p_ret, None,
                                atm_m, None, atm_e, None,
                                mode="full",
                                retrieval_atmosphere=retrieval_atmosphere)

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
                                K_p, phase, berv, planet.systemic_velocity_kms, v_wind)

                            model_mat[hh], _ = spec_to_mat_fraction(
                                v_planet_r,
                                wave_star[retrieval_orders[hh], :],
                                wave_pRT_r,
                                per_order_results[hh]["syn_spec"],
                                np.ones((n_spectra, n_pixels)),
                                with_signal,
                                without_signal,
                                fraction,
                                event_type=cfg.observation.event_type,
                                signal_scale=cfg.pipeline.inject_scale_factor,
                                limb_asymmetries=retrieval_limb_asymmetries,
                                limb_divisions=cfg.atmosphere.limb_divisions,
                                resolving_power=inst.res,
                                spec_morning_day=syn_LL_r,
                                spec_evening_day=syn_TL_r,
                                sf_morning_day=per_order_results[hh].get("sf_morning_day"),
                                sf_evening_day=per_order_results[hh].get("sf_evening_day"),
                                include_star=False,
                            )
                            if prepare_retrieval_template:
                                if not cfg.pipeline.sysrem_robust_halt:
                                    _sysrem_pass = None
                                model_mat_prepared[hh] = \
                                    prepare_observation(
                                        retrieval_preparation,
                                        model_mat[hh],
                                        _std_noise_ret[_night_index, :n_spectra, hh * _n_px_b9 : (hh + 1) * _n_px_b9],
                                        wave_star[retrieval_orders[hh], :],
                                        np.where(useful_spectral_points_snr_aux[_night_index, hh, :])[0],
                                        np.where(mask_snr_ret_aux[_night_index, hh, :])[0],
                                        airmass,
                                        phase,
                                        without_signal,
                                        _sysrem_pass,
                                        None,
                                        tell_mask_threshold_Blain24=0.8,
                                        max_fit_BL19=False,
                                        sysrem_division=False,
                                        masks=False,
                                        correct_uncertainties=False,
                                        retrieval=True,
                                        mask_inter_retrieval=np.where(mask_inter_ret_aux[_night_index, hh, :])[0],
                                        useful_spectral_points_inter_retrieval=np.where(
                                            useful_spectral_points_inter_aux[_night_index, hh, :]
                                        )[0],
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
                        (str(dirs['root']) + '/').rstrip("/") + "/plots_SNR"
                        if cfg.cross_correlation.all_significance_metrics
                        else (str(dirs['plots']) + '/').rstrip("/")
                    )
                    os.makedirs(_base_dir_plot_b9, exist_ok=True)

                    _ctx["idx_interval"] = np.arange(n_spectra)
                    _ctx["interval_tracker"] = None

                    for night_index in range(retrieved_nights):
                        _ctx["night_index"] = night_index


                        if night_index == 0:
                            _ctx["atmosphere_ret_list"]        = []
                            _ctx["atmosphere_ret_list_morning"] = []
                            _ctx["atmosphere_ret_list_evening"] = []
                            for hh in range(_n_orders_b9):
                                if _Radtrans is not None:
                                    _same_sp_b9 = (
                                        _md.species[2:]
                                        != _ed.species[2:])
                                    if _same_sp_b9:
                                        if hh == 0:
                                            _ctx["atmosphere_ret_list"] = []
                                        _atm9m = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_md.species[2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[retrieval_orders[hh],
                                                    :].min() - 0.01,
                                                wave_star[retrieval_orders[hh],
                                                    :].max() + 0.01,
                                            ],
                                            line_opacity_mode="lbl",
                                        )
                                        _ctx[
                                            "atmosphere_ret_list_morning"
                                        ].append(_atm9m)
                                        _atm9e = _Radtrans(
                                            pressures=p_ret,
                                            line_species=_ed.species[2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[retrieval_orders[hh],
                                                    :].min() - 0.01,
                                                wave_star[retrieval_orders[hh],
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
                                            line_species=_ed.species[2:],
                                            rayleigh_species=["H2", "He"],
                                            gas_continuum_contributors=[
                                                "H2--H2", "H2--He"],
                                            wavelength_boundaries=[
                                                wave_star[retrieval_orders[hh],
                                                    :].min() - 0.01,
                                                wave_star[retrieval_orders[hh],
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
                                const_efficiency_mode=cfg.retrieval.constant_efficiency_mode,
                                n_live_points=cfg.retrieval.live_points,
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
                                    if (False)
                                    else [
                                        np.log10(limb_retrieval.morning_abundances[2]),
                                        np.log10(limb_retrieval.evening_abundances[2]),
                                        planet.kp_kms,
                                        _md.equilibrium_temperature_K,
                                        _ed.equilibrium_temperature_K,
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
        # Store this completed run's arrays and observing metadata. The public
        # result methods validate their selected array layout when called.
        self._completed_run = (
            self._state, tuple(int(i) for i in order_selection),
            {
                "systemic_velocity_kms": float(planet.systemic_velocity_kms),
                "significant_eccentricity": cfg.observation.significant_eccentricity,
                "different_nights": cfg.observation.different_nights,
                "n_nights": cfg.observation.n_nights,
                "gaussian_noise_seed": cfg.noise.noise_seed,
                "throughput_base_seed": cfg.observation.throughput_seed,
                "snr_exclude_kms": cfg.cross_correlation.snr_exclude_kms,
            },
        )
        print("\n  === run() complete ===\n")
