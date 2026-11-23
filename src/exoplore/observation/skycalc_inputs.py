"""PWV interpolation and reusable SkyCalc transmission inputs.

Time interpolation describes the user's chosen PWV history. Spectral
interpolation combines transmissions at neighbouring SkyCalc PWV nodes.
All cache entries retain the complete downloaded FITS model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

# ESO SkyCalc's explicit PWV selector, in millimetres.
PWV_GRID_MM = np.array([0.05, 0.1, 0.25, 0.5, 1., 1.5, 2.5, 3.5,
                        5., 7.5, 10., 20., 30.])


def exposure_pwv(hours, series):
    """Interpolate user-supplied time/PWV knots without extrapolation."""
    if not isinstance(series, dict) or set(series) != {"hours_from_observation_start", "pwv_mm"}:
        raise ValueError("pwv_time_series needs hours_from_observation_start and pwv_mm arrays.")
    t = np.asarray(series["hours_from_observation_start"], dtype=float)
    p = np.asarray(series["pwv_mm"], dtype=float)
    h = np.asarray(hours, dtype=float)
    if (t.ndim != 1 or p.shape != t.shape or len(t) < 2
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p))
            or not np.all(np.diff(t) > 0)):
        raise ValueError("PWV time series needs at least two finite, increasing times and matching PWV values.")
    if np.any(p < PWV_GRID_MM[0]) or np.any(p > PWV_GRID_MM[-1]):
        raise ValueError("PWV must lie between 0.05 and 30 mm.")
    if not np.all(np.isfinite(h)) or np.any(h < t[0]) or np.any(h > t[-1]):
        raise ValueError("PWV time series must cover every exposure time (hours from observation start).")
    return np.interp(h, t, p)


def pwv_bracket(pwv):
    """Return lower node, upper node and the upper node's linear weight."""
    if not np.isfinite(pwv) or not PWV_GRID_MM[0] <= pwv <= PWV_GRID_MM[-1]:
        raise ValueError("SkyCalc PWV must lie between 0.05 and 30 mm.")
    i = int(np.searchsorted(PWV_GRID_MM, pwv))
    if i < len(PWV_GRID_MM) and PWV_GRID_MM[i] == pwv:
        return float(pwv), float(pwv), 0.0
    lo, hi = PWV_GRID_MM[i-1:i+1]
    return float(lo), float(hi), float((pwv-lo)/(hi-lo))


def skycalc_directory(tellurics, inputs_dir, event, night=None):
    """Resolve a shared single-night directory or a numbered night directory."""
    root = getattr(tellurics, "skycalc_directory", None)
    if root:
        path = Path(root)
        return path if night is None else path / f"night_{night}"
    path = Path(inputs_dir) / f"Skycalc_{event}"
    if night is not None:
        path /= f"night_{night}"
    return path / ("Fixed_PWV" if tellurics.constant_pwv else "Variable_PWV")


class SkyCalcCache:
    """Cache complete model queries on disk, including all request settings.

    The caller supplies the effective query parameters. A changed airmass,
    wavelength range, resolution or query recipe therefore gets its own file.
    Failed downloads never become cache entries.
    """
    def __init__(self, directory, query, settings):
        self.directory = Path(directory)
        self.query = query
        self.settings = dict(settings)

    def model(self, airmass, pwv):
        request = dict(self.settings, airmass=float(airmass), pwv=float(pwv))
        encoded = json.dumps(request, sort_keys=True, allow_nan=False)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        path = self.directory / f"{key}.fits"
        if not path.exists():
            hdul = self.query(airmass, pwv, **self.settings["arguments"])
            if hdul is None:
                raise RuntimeError(f"SkyCalc returned no model for airmass={airmass}, PWV={pwv} mm.")
            self.directory.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            try:
                hdul.writeto(temp, overwrite=True)
                temp.replace(path)
                path.with_suffix(".json").write_text(encoded + "\n")
            finally:
                hdul.close()
                temp.unlink(missing_ok=True)
        with fits.open(path) as hdul:
            return fits.HDUList([hdu.copy() for hdu in hdul])

    def transmission(self, airmass, pwv):
        """Interpolate transmission only, on identical native wavelength grids.

        Other SkyCalc columns remain those of the lower node. EXoPLORE consumes
        lam/trans only. Headers explicitly identify this transmission product.
        """
        lo, hi, weight = pwv_bracket(pwv)
        result = self.model(airmass, lo)
        if hi != lo:
            with self.model(airmass, hi) as upper:
                if not np.array_equal(result[1].data["lam"], upper[1].data["lam"]):
                    result.close()
                    raise ValueError("SkyCalc PWV neighbours must share an identical wavelength grid.")
                result[1].data["trans"] = ((1-weight)*result[1].data["trans"]
                                            + weight*upper[1].data["trans"])
        result[0].header["PWVREQ"] = float(pwv)
        result[0].header["PWVLO"] = lo
        result[0].header["PWVHI"] = hi
        result[0].header["PWVWGHT"] = weight
        result[0].header["PWVINT"] = "linear transmission"
        return result


def configured_exposure_dates(observation, planet, inputs_dir, night=None,
                              tellurics=None, observatory="paranal"):
    """Plan the molecular simulator's dates without atmosphere calculations.

    Single-night sampling calls get_event directly. Synthetic multiple nights
    follow the runner's epoch selection and optional per-night cadence branch.
    Measured nights read their existing JD arrays.
    """
    from .timing import get_event
    if observation.specific_event:
        index = night if night is not None else 0
        path = Path(inputs_dir) / "reference_night" / f"julian_date_{index}.fits"
        with fits.open(path) as hdul:
            return np.asarray(hdul[0].data, dtype=float)
    if not observation.different_nights:
        return get_event(observation, planet, None)[0]
    if night is None or not 0 <= night < observation.n_nights:
        raise ValueError("Multiple-night SkyCalc planning requires --night N.")
    midpoint = planet.transit_epoch_bjd + night*planet.orbital_period_days
    if (tellurics.use_accurate_airmass and planet.ra_deg is not None
            and planet.dec_deg is not None):
        from .ephemeris import find_observable_transit_epochs
        # Fail clearly if planning cannot establish the observing epochs.
        midpoint = find_observable_transit_epochs(
            planet.transit_epoch_bjd, planet.orbital_period_days,
            planet.ra_deg, planet.dec_deg, observatory, observation.n_nights)[night]
    duration = planet.transit_duration_hours
    pre = observation.pre_event_hours if observation.pre_event_hours > 0 else duration/2
    post = observation.post_event_hours if observation.post_event_hours > 0 else duration/2
    exp = observation.exposure_time_seconds
    if observation.exposure_time_seconds_per_night is not None:
        exp = observation.exposure_time_seconds_per_night[night]
        step = (exp+observation.readout_time_seconds+observation.overhead_time_seconds)/86400.
        return np.arange(midpoint-pre/24., midpoint+post/24., step)
    step = (exp+observation.readout_time_seconds+observation.overhead_time_seconds)/86400.
    return np.arange(midpoint-duration/48.-pre/24.,
                     midpoint+duration/48.+post/24.+step, step)
