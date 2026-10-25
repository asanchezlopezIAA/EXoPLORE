"""Select the physical spectrum provider only inside the helium branch."""

# Reading this file
# The helium runner asks this small selector for its configured physical
# backend. Imports happen inside that choice so a molecular run need not load
# these optional solvers. HeliumTransit here is a factory function: it returns
# the selected backend object, rather than defining another atmosphere model.



def solve_outflow(settings, planet):
    """Return a physical atmosphere from the explicitly selected backend."""
    if settings.backend == 'sunbather':
        from exoplore.atmosphere.helium_sunbather import solve_sunbather
        return solve_sunbather(settings, planet)
    from exoplore.atmosphere.helium_pwinds import solve_outflow as legacy
    return legacy(settings, planet)


def HeliumTransit(settings, planet, science, profile):
    """Provide planet-frame vacuum wavelengths and phase-dependent spectra."""
    # The name is retained for compatibility, but this is a factory: it chooses
    # an object whose backend-specific class caches/evaluates the transit model.
    if settings.backend == 'sunbather':
        from exoplore.atmosphere.helium_sunbather import SunbatherTransit
        return SunbatherTransit(settings, planet, science, profile)
    from exoplore.atmosphere.helium_pwinds import HeliumTransit as legacy
    return legacy(settings, planet, science, profile)
