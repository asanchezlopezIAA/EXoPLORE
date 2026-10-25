"""Select the throughput and Gaussian-noise seeds used by the observing code.

The seed identifies the order and night receiving a random draw. The simulator
and observation.noise use these seeds when generating the noise arrays.
"""

import numpy as np

from exoplore.observation.noise import add_throughput

__all__ = ["throughput_child_seed", "apply_configured_throughput", "gaussian_noise_seed"]


def throughput_child_seed(base_seed, order_index, night_index):
    """Return a throughput seed for the physical order ID and night index.

    None selects fresh random initialization. SeedSequence combines the base
    seed, physical order ID, night index and the throughput tag 1. Selecting or
    reordering a subset gives each physical order the same seeded stream.
    """
    if base_seed is None:
        return None
    return int(np.random.SeedSequence([
        base_seed, int(order_index), int(night_index), 1,
    ]).generate_state(1)[0])


def apply_configured_throughput(
    flux, observation, order_index, night_index, *, throughput_function=None,
):
    """Apply 2% multiplicative jitter to an (exposure, pixel) flux matrix.

    observation supplies noiseless, add_throughput_variations and throughput_seed
    from ObservationConfig. A noiseless run or disabled throughput returns a
    copy of the input. An enabled draw uses the order/night seed.

    throughput_function supplies the draw function. Its default is
    observation.noise.add_throughput.
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
    """Return the Gaussian-noise seed for an order/night calculation.

    A single night uses base_seed. Ordinary multi-night runs use None for
    fresh draws. Cheverall26 multi-night runs combine the base seed, selected-
    order loop position and night index. Reordering that list changes its
    Gaussian streams because selected_order_position identifies the draw.
    """
    seed = base_seed if n_nights == 1 else None
    if pipeline_name == "Cheverall26" and n_nights != 1 and base_seed is not None:
        seed = (int(base_seed) * 1_000_003
                + selected_order_position * 10_007 + night_index)
    return seed
