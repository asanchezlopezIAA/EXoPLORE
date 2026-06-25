"""
EXoPLORE, High-Resolution Exoplanet Atmosphere Simulator
==========================================================

A community package for simulating high-resolution spectroscopic observations
of exoplanet atmospheres, including cross-correlation analysis and atmospheric
retrieval.

Scientific workflow
-------------------
Planet/system parameters
    → Instrument and observing setup
    → Atmospheric forward model (petitRADTRANS)
    → Stellar + planetary + telluric + instrumental effects
    → Time-series spectral observations
    → Data preparation / masking / SYSREM cleaning
    → Cross-correlation analysis
    → Detection metrics and/or atmospheric retrieval

Quickstart
----------
>>> from exoplore.config import SimulationConfig
>>> cfg = SimulationConfig.from_json("configs/hd189733b_andes_transit_clean.json")
>>> from exoplore.core import ExoploreSimulator
>>> sim = ExoploreSimulator(cfg)
>>> sim.summary()
"""

__version__ = "0.2.0"
__author__ = "Ana Sánchez-López"
