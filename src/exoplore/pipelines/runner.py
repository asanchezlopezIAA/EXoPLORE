"""Present preparation results in the shapes consumed by the observing runner.

The fixed-telluric calculation returns a flux/error pair. The observing loop
also needs column masks and detrending information. This adapter supplies
that bookkeeping while keeping the calculated flux and uncertainties intact.
"""

import numpy as np

from .prepare import preparing_pipeline


def prepare_observation(inputs, data, noise, wave, useful, mask, airmass, phase,
                        without_signal, *args, **kwargs):
    """Run preparation and supply the observing loop's result structure.

    With tellurics disabled, pipeline_fixedTellurics performs the existing
    wavelength/time normalization. No additional masking or SYSREM is applied
    by that calculation, so the input column selection is retained and the
    stored detrending vectors are zero. Model-only calls receive its prepared
    flux matrix. The underlying preparing_pipeline interface is preserved.
    """
    result = preparing_pipeline(inputs, data, noise, wave, useful, mask,
                                airmass, phase, without_signal, *args, **kwargs)
    # Seidel22 already returns the flux, errors and masks expected by the
    # observing loop, including its stellar-core and interpolation coverage masks.
    if inputs.tellurics.include_tellurics or inputs.pipeline.name == 'Seidel22':
        return result
    prepared, uncertainty = result
    if kwargs.get("masks", True) and kwargs.get("correct_uncertainties", True):
        useful = np.asarray(useful, dtype=int).copy()
        mask = np.asarray(mask, dtype=int).copy()
        if inputs.pipeline.sysrem_robust_halt:
            return prepared, uncertainty, useful, mask, 0
        vectors = np.zeros((data.shape[0], inputs.sysrem_iterations))
        return (prepared, uncertainty, useful, mask, 0,
                mask.copy(), useful.copy(), None, vectors)
    return prepared
