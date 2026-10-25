"""Choose seeds for the existing throughput and Gaussian-noise calculations.

The simulator calls these helpers before its existing noise routines. They
choose seeds; the draws and their amplitudes remain in observation.noise or
the runner. Moving this code must preserve the resulting noise in every case.

Different nights must retain their own noise draws. The existing throughput
and Gaussian choices are preserved separately, exactly as implemented before.
"""

import numpy as np

from exoplore.observation.noise import add_throughput

__all__ = ["throughput_child_seed", "apply_configured_throughput", "gaussian_noise_seed"]


def throughput_child_seed(base_seed, order_index, night_index):
    """Return the existing throughput child seed, or None for unseeded draws.

    order_index is the **physical instrument order ID**, not its position in a
    selected subset. Thus selecting/reordering orders preserves their streams.
    The final tag, 1, distinguishes this stream from direct use of the base seed
    by the Gaussian component. Its value and coordinate order must stay stable.
    """
    if base_seed is None:
        return None
    return int(np.random.SeedSequence([
        base_seed, int(order_index), int(night_index), 1,
    ]).generate_state(1)[0])


def apply_configured_throughput(
    flux, observation, order_index, night_index, *, throughput_function=None,
):
    """Apply the existing 2% multiplicative jitter to (exposure, pixel) flux.

    observation supplies noiseless, add_throughput_variations and throughput_seed
    from ObservationConfig. Disabled/noiseless runs return a copy without draws.
    The actual routine and its defaults (including white jitter) are unchanged.

    throughput_function lets the old simulator wrapper pass its original draw
    routine. Tests/tools that replace that routine therefore keep working.
    Ordinary callers should leave this argument unset.
    """
    if observation.noiseless or not observation.add_throughput_variations:
        return np.copy(flux)

    seed = throughput_child_seed(observation.throughput_seed, order_index, night_index)
    if throughput_function is None:
        throughput_function = add_throughput
    return throughput_function(F=flux, jitter_frac=0.02, seed=seed)


def gaussian_noise_seed(
    base_seed, *, n_nights, pipeline_name, selected_order_position, night_index,
):
    """Select the seed used by the runner's existing Gaussian noise draw.

    Single-night runs pass the configured base seed directly. Ordinary
    multi-night runs retain None, even when a base seed is configured.
    Cheverall26 alone derives a multi-night seed from the base, the selected-
    order **loop position** and night index, exactly as before extraction.

    Do not substitute a physical order ID for selected_order_position: unlike
    throughput, this existing policy is not invariant to subset reordering.
    This function selects a seed; it does not generate or scale noise.
    """
    seed = base_seed if n_nights == 1 else None
    if pipeline_name == "Cheverall26" and n_nights != 1 and base_seed is not None:
        seed = (int(base_seed) * 1_000_003
                + selected_order_position * 10_007 + night_index)
    return seed
