"""CARMENES ETC anchors and the ANDES relative S/N prescription.

io/carmenes_etc.py writes the input bundle from downloaded ETC responses.
This file converts its order maxima to native-pixel continuum S/N. The
direct sodium noise module then uses that continuum anchor to calculate
photon uncertainties on the simulated flux scale.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exoplore.instruments.wavegrid import pixel_snr_one_order


@dataclass(frozen=True)
class CarmenesETC:
    """Full ETC curves, retained alongside their observing conditions.

    Wavelengths are in nm, S/N is per resolution element and per exposure.
    Curve axes are (airmass, physical order index, wavelength sample).
    """

    # One wavelength vector per physical order, in nm.
    wave_nm: np.ndarray
    # ETC S/N with axes (downloaded airmass, order, wavelength).
    snr_per_resel: np.ndarray
    # Increasing airmasses at which the ETC was evaluated.
    airmass: np.ndarray
    # Exposure time common to those downloaded ETC calculations.
    exposure_seconds: float
    # VIS or NIR identifies the instrument channel covered by the bundle.
    channel: str

    @classmethod
    def load(cls, path):
        """Read a bundle produced by scripts/carmenes_etc_to_fits.py."""
        # Read the numerical response curves and their observing conditions.
        with np.load(path, allow_pickle=False) as data:
            table = cls(data['wave_nm'], data['snr_per_resel'],
                        data['airmass'], float(data['exposure_seconds']),
                        str(data['channel']))
        # Check dimensions, wavelength ordering and the supported instrument channel together.
        w, s, x = table.wave_nm, table.snr_per_resel, table.airmass
        if (w.ndim != 2 or s.shape != (x.size, *w.shape)
                or x.ndim != 1 or x.size == 0
                or not np.isfinite(w).all() or np.any(np.diff(w, axis=1) <= 0)
                or not np.isfinite(x).all() or np.any(x < 1)
                or np.any(np.diff(x) <= 0)
                or table.channel not in ('VIS', 'NIR')
                or not np.isfinite(table.exposure_seconds)
                or table.exposure_seconds <= 0):
            raise ValueError('Invalid CARMENES ETC bundle axes or observing conditions')
        # Return the validated ETC input to the observation builder.
        return table

    def check_airmass(self, airmass):
        """Check the observing sequence before calculating atmospheric spectra."""
        # Read the simulated exposure airmasses as one finite vector.
        x = np.asarray(airmass, float)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError('Supply finite exposure airmasses for the CARMENES ETC')
        # Require exposure conditions within the downloaded airmass range.
        if np.any(x < self.airmass[0]-1e-8) or np.any(x > self.airmass[-1]+1e-8):
            raise ValueError('Exposure airmass is outside the supplied CARMENES ETC range')

    def pixel_snr(self, order, wave_um, tellurics, airmass,
                  exposure_seconds, resolving_power):
        """Anchor an entire native order at each ETC curve's maximum.

        Use the existing ANDES sinc-squared blaze and relative telluric
        scaling. Native wavelength spacing sets pixels per resolution element.
        The stellar-line and random throughput factors keep the ANDES defaults.
        """
        # Use the complete native order, in micrometres, to position its blaze.
        wave = np.asarray(wave_um, float)
        # Terrestrial transmission has one row per exposure.
        tell = np.asarray(tellurics, float)
        x = np.asarray(airmass, float)
        # Require the ETC anchor and simulation to use the same exposure duration.
        if not np.isclose(exposure_seconds, self.exposure_seconds, rtol=0, atol=1e-6):
            raise ValueError('CARMENES ETC exposure time differs from the simulation')
        if (not isinstance(order, (int, np.integer)) or order < 0
                or order >= self.wave_nm.shape[0]):
            raise ValueError('Physical order index is outside the CARMENES ETC bundle')
        if (wave.ndim != 1 or wave.size < 3 or np.any(np.diff(wave) <= 0)
                or not np.isfinite(wave).all() or np.any(wave <= 0)
                or tell.shape != (x.size, wave.size) or x.ndim != 1
                or not np.isfinite(x).all() or not np.isfinite(tell).all()
                or np.any(tell < 0) or np.any(tell > 1)
                or not np.isfinite(resolving_power) or resolving_power <= 0):
            raise ValueError('Supply a complete native order and exposure-dependent transmissions')
        # Check atmospheric-condition coverage before interpolating the downloaded responses.
        self.check_airmass(x)
        # Read the selected physical order at every ETC reference airmass.
        curves = self.snr_per_resel[:, order]
        # ETC curves contain unavailable samples at detector edges. The anchor
        # uses the maximum among valid samples, retaining those gaps in the file.
        curves = np.where(np.isfinite(curves) & (curves > 0), curves, 0.)
        # Locate each reference airmass's maximum-S/N wavelength.
        peaks = np.argmax(curves, axis=1)
        # Read those continuum S/N anchors in per-resolution-element units.
        maxima = curves[np.arange(len(curves)), peaks]
        if np.any(maxima <= 0):
            raise ValueError('Selected CARMENES ETC order has no positive S/N anchor')
        # Each downloaded airmass has its own maximum and peak wavelength.
        # Interpolate the resulting S/N matrices in airmass, avoiding a second
        # extinction law on top of the ETC observing conditions.
        # Estimate pixels per resolution element from lambda / R divided by local pixel spacing.
        sampling = wave/(resolving_power*np.gradient(wave))
        # Collect a native-pixel S/N prediction for each downloaded ETC airmass.
        models = []
        # Each ETC maximum has its own anchor wavelength and S/N.
        for peak, maximum in zip(peaks, maxima):
            # Convert that ETC wavelength from nm to micrometres.
            anchor_um = self.wave_nm[order, peak]/1000
            if not wave[0] <= anchor_um <= wave[-1]:
                raise ValueError('Use the complete native order so the ETC maximum is covered')
            # Find the native pixel closest to the ETC anchor.
            center = int(np.argmin(abs(wave-anchor_um)))
            # Apply the shared blaze and relative telluric shape, then divide S/N by
            # the square root of pixels per resolution element.
            models.append(pixel_snr_one_order(
                wave, tell, maximum, px_per_resel=1, center_pix=center)
                / np.sqrt(sampling)[None, :])
        # Stack reference models with axes (ETC airmass, exposure, pixel).
        models = np.asarray(models)
        # Use the physical order ID to retain its ETC anchor across different
        # order selections.
        # With one downloaded airmass, its checked model supplies all exposure rows.
        if self.airmass.size == 1:
            return models[0]
        # Bracket every exposure airmass between neighbouring downloaded references.
        upper = np.clip(np.searchsorted(self.airmass, x), 1, len(self.airmass)-1)
        # The preceding reference provides the lower bracket.
        lower = upper-1
        # Calculate the linear fraction between the two reference airmasses.
        weight = np.clip((x-self.airmass[lower]) /
                         (self.airmass[upper]-self.airmass[lower]), 0, 1)
        # Select the matching exposure row within each reference model.
        rows = np.arange(x.size)
        # Blend the two reference S/N predictions for each exposure and wavelength.
        return ((1-weight[:, None])*models[lower, rows]
                + weight[:, None]*models[upper, rows])


def configured_carmenes_etc(config):
    """Validate the master-JSON selection before starting a simulation."""
    # Read the optional ETC-bundle path from the shared noise configuration.
    path = config.noise.carmenes_etc_file
    # An unset ETC bundle leaves the existing instrument S/N-input choice available.
    if path is None:
        return None
    # Check the instrument, synthetic-data selection and ETC-based S/N choice.
    if (config.instrument.name not in ('CARMENES_VIS', 'CARMENES_NIR')
            or config.observation.use_real_data
            or config.noise.noise_choice != 'SNR'
            or config.noise.fixed_snr is not None
            or config.noise.use_mean_snr):
        raise ValueError('carmenes_etc_file requires synthetic CARMENES, noise_choice=SNR, '
                         'fixed_snr=null and use_mean_snr=false')
    # Load and validate the selected response bundle.
    table = CarmenesETC.load(Path(path))
    # Require agreement between the configured instrument and downloaded ETC channel.
    if config.instrument.name != 'CARMENES_'+table.channel:
        raise ValueError('CARMENES ETC channel differs from instrument.name')
    if not np.isclose(config.observation.exposure_time_seconds,
                      table.exposure_seconds, rtol=0, atol=1e-6):
        raise ValueError('CARMENES ETC exposure time differs from the simulation')
    return table
