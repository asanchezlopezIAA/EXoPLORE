"""Full-array ETC noise conversion and helium-order selection checks.

Wavelengths are vacuum nm. S/N is per exposure. These helpers do not draw
noise, reconstruct a blaze profile, or change the selected instrument orders.
"""
from __future__ import annotations

import warnings
import numpy as np


def etc_pixel_noise(snr_per_resel: np.ndarray,
                    pixels_per_resel: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel S/N and normalized-flux noise standard deviation.

    Accept an entire (..., orders, pixels) S/N array. Sampling can be a
    scalar or an array broadcastable to the S/N shape; use (orders, 1)
    for per-order sampling. Equal signal and independent equal-noise pixels
    within each resolution element give S/N_pixel = S/N_resel / sqrt(n).
    Invalid or zero ETC S/N gives zero pixel S/N and infinite uncertainty,
    so unavailable pixels must be masked before scientific preparation.
    No exposure-time, stellar, blaze or telluric scaling is applied here.
    """
    snr = np.asarray(snr_per_resel, dtype=float)
    sampling = np.broadcast_to(np.asarray(pixels_per_resel, dtype=float), snr.shape)
    if np.any(~np.isfinite(sampling)) or np.any(sampling <= 0):
        raise ValueError('Pixels per resolution element must be finite and positive')
    valid = np.isfinite(snr) & (snr > 0)
    pixel_snr = np.zeros_like(snr)
    np.divide(snr, np.sqrt(sampling), out=pixel_snr, where=valid)
    std_noise = np.full_like(snr, np.inf)
    np.divide(1., pixel_snr, out=std_noise, where=valid)
    return pixel_snr, std_noise


def require_helium_orders(wave_orders_nm: np.ndarray, included_orders: list[int],
                          helium_lines_nm: list[float]) -> list[int]:
    """Warn on missing triplet coverage; refuse a selection with no helium.

    Inspect actual vacuum wavelength grids (orders, pixels), not a fixed
    instrument order number. Empty selection means all orders. Detect gaps
    larger than five times the median native pixel spacing. Coverage here
    is at rest wavelengths; Doppler-track edge coverage needs a later check.
    Return native zero-based indices covering at least one triplet component.
    Never silently replace the user's included_orders.
    """
    wave = np.asarray(wave_orders_nm, dtype=float)
    lines = np.asarray(helium_lines_nm, dtype=float)
    if wave.ndim != 2 or lines.ndim != 1 or not len(lines) or not np.isfinite(lines).all():
        raise ValueError('Supply an orders-by-pixels vacuum-nm grid and finite helium wavelengths')
    selected = list(included_orders) if len(included_orders) else list(range(len(wave)))
    if any(i < 0 or i >= len(wave) for i in selected):
        raise ValueError('included_orders contains an index outside the instrument grid')
    coverage = np.zeros((len(wave), len(lines)), dtype=bool)
    for i, row in enumerate(wave):
        spacing = np.diff(row)
        good = np.isfinite(row[:-1]) & np.isfinite(row[1:]) & (spacing > 0)
        if not np.any(good):
            continue
        intervals = good & (spacing <= 5 * np.median(spacing[good]))
        for j, line in enumerate(lines):
            coverage[i, j] = np.any(intervals & (row[:-1] <= line) & (row[1:] >= line))
    available = np.flatnonzero(coverage.any(axis=1)).tolist()
    covered = coverage[selected].any(axis=0)
    if not covered.all():
        message = (f'Helium simulation: included_orders={selected} omits vacuum He I '
                   f'components {lines[~covered].tolist()} nm. Orders covering helium: '
                   f'{available} (zero-based). Update included_orders for your dataset.')
        warnings.warn(message, UserWarning, stacklevel=2)
        if not covered.any():
            raise ValueError('No helium line is covered by the selected orders; simulation stopped')
    return available
