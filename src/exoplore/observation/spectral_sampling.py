"""Apply instrumental broadening and integrate a spectrum over native pixels.

The sodium observation builder calls this for the complete spectrum, the
continuum and the diagnostic components. Wavelengths must share one unit.
The output is the mean flux density within each native detector pixel.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d


def sample_native_pixels(flux, fine_wave, native_log_edges, log_step, resolving_power):
    """Use the Gaussian response and pixel integration of the helium simulator.

    fine_wave is uniformly sampled in log wavelength. Native pixel boundaries
    are supplied in log wavelength, so the same operation supports irregular
    instrument grids. Fluxes are integrated before forming normalized ratios.
    """
    # A Gaussian has FWHM = 2 sqrt(2 ln 2) sigma. Convert 1/R from
    # log-wavelength width into a standard deviation in fine-grid samples.
    sigma = 1/(resolving_power*2*np.sqrt(2*np.log(2))*log_step)
    # Apply the Gaussian instrumental line-spread function to the fine spectrum.
    blurred = gaussian_filter1d(flux, sigma, mode='nearest')
    # Integrate flux density with respect to wavelength along the fine grid.
    integral = cumulative_trapezoid(blurred, fine_wave, initial=0)
    # Convert log-wavelength detector boundaries back to wavelength.
    pixel_edges = np.exp(native_log_edges)
    # Evaluate the cumulative integral at each edge, subtract adjacent values,
    # then divide by pixel width to obtain the mean flux density in each pixel.
    return np.diff(np.interp(pixel_edges, fine_wave, integral))/np.diff(pixel_edges)
