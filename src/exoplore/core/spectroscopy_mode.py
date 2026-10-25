"""Select the spectroscopy route independently of the source of the data."""
from __future__ import annotations


def spectroscopy_route(config) -> str:
    """Choose standard, helium_simulation, helium_observed or helium_grid.

    The explicit helium switch chooses the spectroscopy. use_real_data chooses
    its input source. Omission retains older direct-helium configurations.
    """
    # A helium model grid has its own runner; it must be selected before the
    # single-observation routes below. This function selects, rather than runs.
    if config.helium_grid is not None and config.helium_grid.enabled:
        if config.observation.helium_transmission_spectroscopy is False:
            raise ValueError('Enable helium_transmission_spectroscopy for a helium grid')
        return 'helium_grid'
    selected = config.observation.helium_transmission_spectroscopy
    if selected is not None and type(selected) is not bool:
        raise ValueError('helium_transmission_spectroscopy must be true or false')
    if selected is None:
        # Older configs predate the explicit switch. Preserve their meaning by
        # recognizing their helium atmosphere or direct-helium preparation name.
        selected = (config.atmosphere.helium is not None or config.pipeline.name in ('czesla2024', 'carmenes_helium', 'Palle2020'))
    if not selected:
        if config.pipeline.name in ('czesla2024', 'carmenes_helium', 'Palle2020'):
            raise ValueError('czesla2024 prepares helium spectra; enable helium_transmission_spectroscopy')
        return 'standard'
    if config.pipeline.name not in ('czesla2024', 'carmenes_helium', 'Palle2020'):
        raise ValueError('Select a direct helium preparation pipeline')
    # OH is terrestrial emission near helium. Check its configured treatment
    # only for this spectroscopy route; do not apply it to molecular tutorials.
    from exoplore.config.helium_oh import validate_oh_treatment
    validate_oh_treatment(config)
    if config.pipeline.name == 'carmenes_helium' and config.observation.use_real_data:
        raise ValueError('CARMENES helium currently supports synthetic extracted spectra only')
    # Spectroscopy and data origin are separate choices: the same helium
    # analysis family can consume synthetic or existing observed spectra.
    return 'helium_observed' if config.observation.use_real_data else 'helium_simulation'
