"""Load independent stellar components and interpolate their local intensities.

Every component uses the same LocalIntensityGrid contract. Spectral line
shapes, continuum contrast and CLV come from those inputs. Region labels
select a spectrum through regions.py. They carry no atmosphere-code choice.
"""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import hashlib
import numpy as np

from exoplore.io.stellar_surface import load_local_intensities
from exoplore.config.stellar_surface import SurfaceRegionConfig
from .surface import LocalIntensityGrid, StellarTransitSpectra, C_KMS
from .regions import component_map


@dataclass(frozen=True)
class StellarComponent:
    """A local spectrum, its continuum, optional velocity table and provenance."""
    name: str
    grid: LocalIntensityGrid
    resolution: float
    metadata: dict
    velocity_offset_kms: float = 0.
    velocity_mu: np.ndarray | None = None
    velocity_kms: np.ndarray | None = None

    def velocity(self, mu):
        """Positive component velocities shift local profiles to the red."""
        if self.velocity_mu is None:
            return np.full(np.shape(mu), self.velocity_offset_kms)
        if np.min(mu) < self.velocity_mu[0] or np.max(mu) > self.velocity_mu[-1]:
            raise ValueError(f'Component {self.name} velocity table lacks required mu coverage')
        return np.interp(mu, self.velocity_mu, self.velocity_kms)


def quiet_correction_config(settings):
    """Use the declared default component everywhere for an incomplete correction."""
    return replace(settings, correction_surface='matched', regions=[
        SurfaceRegionConfig(settings.default_component, x=0., y=0., radius=2.)])


class StellarIntensityField:
    """Select one preloaded spectrum at each projected stellar position."""
    def __init__(self, settings, choices, planet):
        self.settings, self.choices = settings, choices
        self.obliquity = choices.projected_obliquity_deg or 0.
        self.components = {}
        units, resolutions = set(), set()
        for name, item in settings.components.items():
            grid, resolution, metadata = load_local_intensities(
                item.intensity_model_file, wavelength_medium=choices.wavelength_medium)
            if grid.continuum is None:
                raise ValueError('Heterogeneous components require their absolute local continua')
            # Apply the existing RM-only definition independently to each component.
            if choices.effects == 'rm':
                if grid.mu[-1] != 1:
                    raise ValueError('RM-only components require a disk-centre spectrum')
                grid = LocalIntensityGrid(grid.wavelength_um, grid.mu,
                    grid.continuum*(grid.intensity[-1]/grid.continuum[-1]), grid.continuum)
            velocity_mu = velocity = None
            if item.velocity_model_file:
                with np.load(item.velocity_model_file, allow_pickle=False) as table:
                    velocity_mu, velocity = table['mu'].copy(), table['velocity_kms'].copy()
                if (velocity_mu.ndim != 1 or len(velocity_mu) < 2 or velocity.shape != velocity_mu.shape
                        or not np.isfinite(velocity_mu).all() or not np.isfinite(velocity).all()
                        or np.any(np.diff(velocity_mu) <= 0) or velocity_mu[0] < 0 or velocity_mu[-1] > 1
                        or np.any(abs(velocity) >= C_KMS)):
                    raise ValueError('Velocity tables require increasing mu and finite velocity_kms vectors')
                velocity_mu.setflags(write=False)
                velocity.setflags(write=False)
            metadata = dict(metadata, user=item.metadata,
                filename=str(item.intensity_model_file),
                sha256=hashlib.sha256(Path(item.intensity_model_file).read_bytes()).hexdigest())
            if item.velocity_model_file:
                metadata['velocity_sha256'] = hashlib.sha256(Path(item.velocity_model_file).read_bytes()).hexdigest()
            self.components[name] = StellarComponent(name, grid, resolution, metadata,
                item.velocity_offset_kms, velocity_mu, velocity)
            units.add(metadata['intensity_unit'])
            resolutions.add(resolution)
        if len(units) != 1 or len(resolutions) != 1:
            raise ValueError('Stellar components require common intensity units and prior resolving power')
        if choices.coupling == 'spatial' and resolutions != {0.}:
            raise ValueError('Spatial stellar components must be intrinsic spectra with resolving_power=0')
        if settings.time_dependent:
            # Pattern motion and line Doppler shifts describe one solid-body star.
            expected = 2*np.pi*planet.R_star/(settings.rotation_period_days*86400)/1e5
            expected *= np.sin(np.deg2rad(settings.stellar_inclination_deg))
            if not np.isclose(expected, planet.v_rotsini_kms,
                              rtol=settings.rotation_consistency_rtol, atol=1e-6):
                raise ValueError('Rotation period, stellar inclination, radius and v_rotsini_kms are inconsistent')
        self.default = self.components[settings.default_component]
        # Retain the finest native wavelength structure supplied by any component.
        self.log_wavelength_step = min(float(np.min(np.diff(np.log(c.grid.wavelength_um))))
                                       for c in self.components.values())

    def labels(self, x, y, time_bjd=None):
        return component_map(x, y, self.settings, self.obliquity, time_bjd)

    def times(self, times_bjd, count):
        """Physical maps require actual exposure times supplied by the caller."""
        if times_bjd is None:
            if self.settings.time_dependent:
                raise ValueError('Supply times_bjd for rotating stellar regions')
            return np.zeros(count)
        values = np.asarray(times_bjd, float)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError('times_bjd must be a finite vector matching the exposure phases')
        return values

    def sample(self, wave, mu, rotational_velocity, labels):
        """Return ray-by-wavelength intensities on independent native model grids."""
        wave = np.asarray(wave, float)
        if (wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all()
                or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
            raise ValueError('Local sampling wavelengths must be finite, positive and increasing')
        mu, rotation, labels = map(np.asarray, (mu, rotational_velocity, labels))
        if (mu.ndim != 1 or rotation.shape != mu.shape or labels.shape != mu.shape
                or not np.isfinite(mu).all() or not np.isfinite(rotation).all()):
            raise ValueError('Local rays require matching finite mu and velocity vectors')
        unknown = set(labels)-set(self.components)
        if unknown:
            raise ValueError(f'Unknown local stellar components: {sorted(unknown)}')
        intensity = np.empty((len(mu), len(wave)))
        continuum = np.empty_like(intensity)
        for name, component in self.components.items():
            rows = np.flatnonzero(labels == name)
            if not len(rows):
                continue
            grid, angles = component.grid, mu[rows]
            if angles.min() < grid.mu[0] or angles.max() > grid.mu[-1]:
                raise ValueError(f'Component {name} lacks required mu coverage')
            velocity = rotation[rows]+component.velocity(angles)
            if np.any(abs(velocity) >= C_KMS):
                raise ValueError('Combined local velocity must be subluminal')
            rest = np.asarray(wave)[None, :]/(1+velocity[:, None]/C_KMS)
            if rest.min() < grid.wavelength_um[0] or rest.max() > grid.wavelength_um[-1]:
                raise ValueError(f'Component {name} needs wider wavelength coverage for its local velocity')
            upper = np.clip(np.searchsorted(grid.mu, angles, side='right'), 1, len(grid.mu)-1)
            a = ((angles-grid.mu[upper-1])/(grid.mu[upper]-grid.mu[upper-1]))[:, None]
            right = np.clip(np.searchsorted(grid.wavelength_um, rest, side='right'), 1, len(grid.wavelength_um)-1)
            t = (rest-grid.wavelength_um[right-1])/(grid.wavelength_um[right]-grid.wavelength_um[right-1])
            for values, target in [(grid.intensity, intensity), (grid.continuum, continuum)]:
                low = (1-t)*values[upper[:, None]-1, right-1]+t*values[upper[:, None]-1, right]
                high = (1-t)*values[upper[:, None], right-1]+t*values[upper[:, None], right]
                target[rows] = (1-a)*low+a*high
        return intensity, continuum

    def angular_flux(self, wave, rays, count, kernel):
        """Accumulate each component through the existing compiled ray integrator."""
        intensity = np.zeros((count, len(wave)))
        continuum = np.zeros_like(intensity)
        for name, component in self.components.items():
            selected = np.flatnonzero(rays.labels == name)
            if not len(selected):
                continue
            grid, mu = component.grid, rays.mu[selected]
            if mu.min() < grid.mu[0] or mu.max() > grid.mu[-1]:
                raise ValueError(f'Component {name} lacks required mu coverage')
            velocity = rays.velocity[selected]+component.velocity(mu)
            if not np.isfinite(velocity).all() or np.any(abs(velocity) >= C_KMS):
                raise ValueError('Combined local velocities must be finite and subluminal')
            if (wave[0]/(1+velocity.max()/C_KMS) < grid.wavelength_um[0]
                    or wave[-1]/(1+velocity.min()/C_KMS) > grid.wavelength_um[-1]):
                raise ValueError(f'Component {name} needs wider wavelength coverage')
            upper = np.clip(np.searchsorted(grid.mu, mu, side='right'), 1, len(grid.mu)-1)
            fraction = (mu-grid.mu[upper-1])/(grid.mu[upper]-grid.mu[upper-1])
            # One angle per selected row lets the same kernel accumulate irregular arcs.
            values, continua = kernel(wave, rays.rings[selected], upper, fraction, velocity,
                rays.weights[selected, None], grid.wavelength_um, grid.intensity, grid.continuum, count)
            intensity += values
            continuum += continua
        return intensity, continuum

    def integrate_surface(self, surface, wave, px, py, radius, times, chunk):
        """Use the existing Cartesian projected areas and opaque occultation masks."""
        times = self.times(times, len(px))
        outside = np.empty((len(px), len(wave)))
        inside = np.empty_like(outside)
        cout, cin = np.empty_like(outside), np.empty_like(outside)
        masks = (surface.x[None, :]-px[:, None])**2+(surface.y[None, :]-py[:, None])**2 < radius**2
        static_labels = None if self.settings.time_dependent else self.labels(surface.x, surface.y)
        # A static projected map has one set of spectra shared by all positions.
        for start in range(0, len(wave), chunk):
            section = slice(start, start+chunk)
            cached = None
            for i, time in enumerate(times):
                if cached is None or self.settings.time_dependent:
                    labels = static_labels if static_labels is not None else self.labels(surface.x, surface.y, time)
                    cached = self.sample(wave[section], surface.mu, surface.velocity_kms, labels)
                spectra, continua = cached
                outside[i, section] = spectra.sum(axis=0)*surface.area
                cout[i, section] = continua.sum(axis=0)*surface.area
                inside[i, section] = outside[i, section]-spectra[masks[i]].sum(axis=0)*surface.area
                cin[i, section] = cout[i, section]-continua[masks[i]].sum(axis=0)*surface.area
        return StellarTransitSpectra(wave.copy(), outside, inside, cout, cin)

    def provenance(self):
        return dict(configuration=asdict(self.settings),
                    components={k: v.metadata for k, v in self.components.items()})
