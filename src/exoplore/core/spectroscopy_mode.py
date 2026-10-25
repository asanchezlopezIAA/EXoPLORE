"""Select the spectroscopy route independently of the source of the data."""
from __future__ import annotations


def spectroscopy_route(config) -> str:
    """Choose standard, helium_simulation, helium_observed or helium_grid.

    The helium switch selects the spectroscopy and use_real_data selects its
    input source. None infers helium from its atmosphere or pipeline settings.
    """
    # Select the helium-grid runner when a model grid is enabled.
    if config.helium_grid is not None and config.helium_grid.enabled:
        if config.observation.helium_transmission_spectroscopy is False:
            raise ValueError('Enable helium_transmission_spectroscopy for a helium grid')
        return 'helium_grid'
    selected = config.observation.helium_transmission_spectroscopy
    if selected is not None and type(selected) is not bool:
        raise ValueError('helium_transmission_spectroscopy must be true or false')
    if selected is None:
        # Infer helium spectroscopy from the atmosphere or preparation method.
        selected = (config.atmosphere.helium is not None or config.pipeline.name in ('czesla2024', 'carmenes_helium', 'Palle2020'))
    if not selected:
        if config.pipeline.name in ('czesla2024', 'carmenes_helium', 'Palle2020'):
            raise ValueError('czesla2024 prepares helium spectra; enable helium_transmission_spectroscopy')
        return 'standard'
    if config.pipeline.name not in ('czesla2024', 'carmenes_helium', 'Palle2020'):
        raise ValueError('Select a direct helium preparation pipeline')
    # OH is terrestrial emission near helium. Check its configured treatment
    # only for this spectroscopy route. do not apply it to molecular tutorials.
    from exoplore.config.helium_oh import validate_oh_treatment
    validate_oh_treatment(config)
    if config.pipeline.name == 'carmenes_helium' and config.observation.use_real_data:
        raise ValueError('CARMENES helium currently supports synthetic extracted spectra only')
    # Spectroscopy and data origin are separate choices: the same helium
    # analysis family can consume synthetic or existing observed spectra.
    return 'helium_observed' if config.observation.use_real_data else 'helium_simulation'
