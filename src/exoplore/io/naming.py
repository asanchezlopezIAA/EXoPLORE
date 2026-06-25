"""
exoplore.io.naming
==================

Simulation run-name builder.  Kept here (rather than in ``core.simulator``)
so analysis modules can import it without creating circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exoplore.config.models import SimulationConfig


def _format_number(x: float) -> str:
    """Format a float for use in simulation names.

    Examples: 1.0 → "1",  1.2 → "1p20",  0.5 → "0p50"
    """
    if int(x) == x:
        return str(int(x))
    integer_part = int(x)
    decimal_part = int(round((x - integer_part), 1) * 100)
    return f"{integer_part}p{decimal_part:02d}"


def build_simulation_name(cfg: "SimulationConfig") -> str:
    """Build the simulation run name.

    Matches the logic at lines 1073-1096 of EXOSIMS_2p0_ANDES.py::

        f"{preparing_pipeline}_{signal_flag}_{PCA_opt}{Kp_flag}{Opt_flag}"
        f"{n_nights}nights_{signif_flag}_{stack_flag}_{real_flag}_{noise_flag}_"
        f"stdnoisex{format_number(Noise_scaling_factor)}"
    """
    obs   = cfg.observation
    pipe  = cfg.pipeline
    ccf   = cfg.cross_correlation
    stats = cfg.statistics

    signal_flag = "withsignal" if obs.simulate_planet else "withoutsignal"

    if ccf.all_significance_metrics:
        signif_flag = "AllMetrics"
    elif ccf.ccf_snr:
        signif_flag = "SNR"
    elif ccf.welch_ttest:
        signif_flag = "Welch"
    else:
        signif_flag = ""

    sg = stats.stack_group_size
    stack_flag = f"comb{sg}" if sg is not None else "comb1"
    real_flag  = "realdata" if obs.use_real_data else "simdata"
    noise_flag = "noiseless" if obs.noiseless else "noisy"

    if pipe.optimize_sysrem_order_by_order:
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
        pca_flag = kp_flag = opt_flag = ""

    sf_str = _format_number(obs.noise_scaling_factor)

    return (
        f"{pipe.name}_{signal_flag}_"
        f"{pca_flag}{kp_flag}{opt_flag}"
        f"{obs.n_nights}nights_"
        f"{signif_flag}_{stack_flag}_{real_flag}_"
        f"{noise_flag}_"
        f"stdnoisex{sf_str}"
    )
