"""Shared explicit OH treatment policy for direct helium spectroscopy.

Validate instrument capabilities before spectra, optional physical models or
external recipes are loaded. Archived baseline/mask configurations retain
their meaning. Fibre transfer factors never have scientific defaults.
"""
from __future__ import annotations

# Role in the calculation
# Validate the OH-emission treatment and its required sky-fibre data. The selected
# correction is applied later by the instrument preparation code.


import math

OH_TREATMENTS = ('ignore', 'mask', 'fiber_b_matched_extraction', 'fiber_b_calibrated', 'fiber_a_oot_profile')


def uses_sky_fibre(treatment: str) -> bool:
    """Whether the OH choice requires observed fibre-B sky spectra."""
    return treatment.startswith('fiber_b_') or treatment == 'fiber_a_oot_profile'


def resolve_oh_treatment(config) -> str:
    """Resolve the public JSON choice, retaining archived OH-mode semantics."""
    treatment = config.pipeline.oh_treatment
    if treatment is None:
        recipe = config.pipeline.palle2020
        if config.pipeline.name == 'Palle2020' and recipe:
            legacy = recipe.get('oh_mode')
        else:
            science = config.pipeline.czesla2024 or config.pipeline.carmenes_helium
            legacy = getattr(science, 'oh_mode', None)
        if legacy not in ('baseline', 'mask'):
            raise ValueError('Supply pipeline.oh_treatment explicitly')
        treatment = 'ignore' if legacy == 'baseline' else 'mask'
    if treatment not in OH_TREATMENTS:
        raise ValueError('pipeline.oh_treatment must be one of: '+', '.join(OH_TREATMENTS))
    return treatment


def validate_oh_treatment(config) -> str:
    """Reject unsupported instruments and missing matched-extraction factors."""
    treatment = resolve_oh_treatment(config)
    if uses_sky_fibre(treatment):
        if config.instrument.name in ('CRIRES+','CRIRES_PLUS'):
            raise ValueError(f'CRIRES+ has no fibre B: oh_treatment={treatment!r} cannot be used with czesla2024. Choose "ignore" or "mask"; CRIRES+ nod subtraction is handled upstream.')
        if (config.instrument.name != 'CARMENES_NIR' or config.pipeline.name != 'Palle2020'
                or not config.observation.use_real_data):
            raise ValueError('Fibre-B OH subtraction requires observed CARMENES_NIR data and Palle2020; synthetic spectra contain no observed sky fibre')
        fit = (config.pipeline.palle2020 or {}).get('oh_fit')
        if not isinstance(fit, dict) or 'profile' not in fit:
            raise ValueError('Fibre-B subtraction requires explicit pipeline.palle2020.oh_fit.profile settings')
        from exoplore.instruments.carmenes_oh import OHFitConfig
        OHFitConfig(**fit['profile'])
        if treatment == 'fiber_a_oot_profile':
            from exoplore.instruments.carmenes_oh_oot import OOTProfileConfig
            if not isinstance(fit.get('oot_profile'), dict):
                raise ValueError('fiber_a_oot_profile requires explicit oh_fit.oot_profile settings')
            OOTProfileConfig(**fit['oot_profile'])
            threshold = fit.get('maximum_anchor_change_sigma')
            if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold) or threshold <= 0):
                raise ValueError('fiber_a_oot_profile requires a positive oh_fit.maximum_anchor_change_sigma')
        if treatment == 'fiber_b_matched_extraction' and fit.get('fiber_b_scale') is not None:
            scale = fit.get('fiber_b_scale')
            values = scale if isinstance(scale, list) else [scale]
            if not values or any(isinstance(v, bool) or not isinstance(v, (int,float))
                                 or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('Optional oh_fit.fiber_b_scale must be a positive B-to-A flux multiplier or one positive multiplier per exposure. Omit it to measure the factor from out-of-transit spectra.')
    return treatment
