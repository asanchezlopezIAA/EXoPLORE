"""exoplore.core, simulation orchestration."""

from exoplore.core.simulator import ExoploreSimulator, SimulationSummary

# Observation matrix builders require scipy; import lazily so that the
# config layer can be used without scipy installed.
def __getattr__(name):
    _obs_exports = {
        "spec_to_mat_fraction",
        "get_stellar_matrix",
        "add_throughput",
        "block_parameter",
        "dayside_fraction",
    }
    if name in _obs_exports:
        from exoplore.core import observation as _obs
        return getattr(_obs, name)
    raise AttributeError(f"module 'exoplore.core' has no attribute {name!r}")


__all__ = [
    "ExoploreSimulator",
    "SimulationSummary",
    # observation matrix builders (lazy, require scipy)
    "spec_to_mat_fraction",
    "get_stellar_matrix",
    "add_throughput",
    "block_parameter",
    "dayside_fraction",
]
