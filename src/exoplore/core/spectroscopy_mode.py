"""Select the spectroscopy route independently of the source of the data."""
from __future__ import annotations


def spectroscopy_route(config) -> str:
    """Return standard, helium_simulation or helium_observed without I/O.

    The explicit helium switch chooses the spectroscopy. use_real_data chooses
    its input source. Omission retains older direct-helium configurations.
    """
    selected = config.observation.helium_transmission_spectroscopy
    if selected is not None and type(selected) is not bool:
        raise ValueError('helium_transmission_spectroscopy must be true or false')
    if selected is None:
        selected = (config.atmosphere.helium is not None or config.pipeline.name in ('czesla2024', 'carmenes_helium'))
    if not selected:
        if config.pipeline.name in ('czesla2024', 'carmenes_helium'):
            raise ValueError('czesla2024 prepares helium spectra; enable helium_transmission_spectroscopy')
        return 'standard'
    if config.pipeline.name not in ('czesla2024', 'carmenes_helium'):
        raise ValueError('Select a direct helium preparation pipeline')
    if config.pipeline.name == 'carmenes_helium' and config.observation.use_real_data:
        raise ValueError('CARMENES helium currently supports synthetic extracted spectra only')
    return 'helium_observed' if config.observation.use_real_data else 'helium_simulation'
