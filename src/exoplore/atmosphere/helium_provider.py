"""Select the physical spectrum provider only inside the helium branch."""


def solve_outflow(settings, planet):
    """Return a physical atmosphere from the explicitly selected backend."""
    if settings.backend == 'sunbather':
        from exoplore.atmosphere.helium_sunbather import solve_sunbather
        return solve_sunbather(settings, planet)
    from exoplore.atmosphere.helium_pwinds import solve_outflow as legacy
    return legacy(settings, planet)


def HeliumTransit(settings, planet, science, profile):
    """Provide planet-frame vacuum wavelengths and phase-dependent spectra."""
    if settings.backend == 'sunbather':
        from exoplore.atmosphere.helium_sunbather import SunbatherTransit
        return SunbatherTransit(settings, planet, science, profile)
    from exoplore.atmosphere.helium_pwinds import HeliumTransit as legacy
    return legacy(settings, planet, science, profile)
