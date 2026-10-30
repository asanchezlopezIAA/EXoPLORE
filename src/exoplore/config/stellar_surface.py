"""Stellar spectra and the regions that select them on the visible surface.

Components describe the emitted light. Regions describe its location. Names
such as quiet, spot or plage are user labels and carry no temperature rule.
"""
from dataclasses import dataclass, field, fields
import math


def _construct(cls, value):
    """Reject misspelled fields before constructing nested configuration."""
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ValueError(f'{cls.__name__} requires a JSON object')
    unknown = set(value)-{f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} fields: {sorted(unknown)}')
    return cls(**value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


@dataclass
class StellarComponentConfig:
    """A local intensity file, optional velocity shift and user provenance."""
    intensity_model_file: str
    velocity_offset_kms: float = 0.
    velocity_model_file: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.intensity_model_file, str) or not self.intensity_model_file:
            raise ValueError('Supply intensity_model_file for every stellar component')
        _finite(self.velocity_offset_kms, 'velocity_offset_kms')
        if abs(self.velocity_offset_kms) >= 299792.458:
            raise ValueError('Component velocity must be subluminal')
        if self.velocity_model_file is not None and (
                not isinstance(self.velocity_model_file, str) or not self.velocity_model_file):
            raise ValueError('velocity_model_file must be a path or null')
        if self.velocity_model_file and self.velocity_offset_kms != 0:
            raise ValueError('Choose either a constant or a tabulated component velocity')
        if not isinstance(self.metadata, dict):
            raise ValueError('Component metadata must be a JSON object')


@dataclass
class SurfaceRegionConfig:
    """Select a component inside a projected circle or rotating spherical cap."""
    component: str
    geometry: str = 'projected'
    name: str = ''
    priority: int = 0
    x: float | None = None
    y: float | None = None
    radius: float | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    angular_radius_deg: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component:
            raise ValueError('Every surface region requires a component name')
        if type(self.priority) is not int:
            raise ValueError('Region priority must be an integer')
        if not isinstance(self.name, str) or not isinstance(self.metadata, dict):
            raise ValueError('Region name must be a string and metadata a JSON object')
        if self.geometry == 'projected':
            for name in ('x', 'y', 'radius'):
                _finite(getattr(self, name), name)
            if self.radius <= 0:
                raise ValueError('Projected region radius must be positive, in stellar radii')
            if any(getattr(self, n) is not None for n in
                   ('latitude_deg', 'longitude_deg', 'angular_radius_deg')):
                raise ValueError('Projected regions use x, y and radius')
        elif self.geometry == 'stellar':
            for name in ('latitude_deg', 'longitude_deg', 'angular_radius_deg'):
                _finite(getattr(self, name), name)
            if not -90 <= self.latitude_deg <= 90 or not 0 < self.angular_radius_deg <= 180:
                raise ValueError('Use latitude in [-90, 90] and angular radius in (0, 180] degrees')
            if any(getattr(self, n) is not None for n in ('x', 'y', 'radius')):
                raise ValueError('Stellar regions use latitude, longitude and angular radius')
        else:
            raise ValueError('Region geometry must be projected or stellar')


@dataclass
class StellarSurfaceConfig:
    """Compose local spectra, region geometry and the assumed correction map.

    Highest priority wins. A later region wins a priority tie. Physical caps
    require an explicit spin inclination, period and BJD_TDB reference epoch.
    """
    components: dict[str, StellarComponentConfig] = field(default_factory=dict)
    default_component: str = 'quiet'
    regions: list[SurfaceRegionConfig] = field(default_factory=list)
    stellar_inclination_deg: float | None = None
    rotation_period_days: float | None = None
    reference_bjd: float | None = None
    reference_phase_deg: float = 0.
    rotation_consistency_rtol: float = .01
    correction_surface: str = 'matched'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.components, dict) or not isinstance(self.regions, list):
            raise ValueError('Supply a component object and an ordered region list')
        self.components = {k: _construct(StellarComponentConfig, v) for k, v in self.components.items()}
        self.regions = [_construct(SurfaceRegionConfig, r) for r in self.regions]
        if not isinstance(self.default_component, str) or not self.default_component:
            raise ValueError('default_component must be a component name')
        if self.regions:
            names = {self.default_component, *(r.component for r in self.regions)}
            if names-set(self.components):
                raise ValueError(f'Unknown stellar components: {sorted(names-set(self.components))}')
        if any(not isinstance(k, str) or not k for k in self.components):
            raise ValueError('Component names must be nonempty strings')
        if self.correction_surface not in ('matched', 'quiet'):
            raise ValueError('correction_surface must be matched or quiet')
        if not isinstance(self.metadata, dict):
            raise ValueError('Surface metadata must be a JSON object')
        _finite(self.reference_phase_deg, 'reference_phase_deg')
        _finite(self.rotation_consistency_rtol, 'rotation_consistency_rtol')
        if not 0 <= self.rotation_consistency_rtol <= .1:
            raise ValueError('rotation_consistency_rtol must lie in [0, 0.1]')
        if self.time_dependent:
            for name in ('stellar_inclination_deg', 'rotation_period_days', 'reference_bjd'):
                _finite(getattr(self, name), name)
            if not 0 <= self.stellar_inclination_deg <= 180 or self.rotation_period_days <= 0:
                raise ValueError('Supply inclination in [0, 180] degrees and a positive rotation period')

    @property
    def active(self):
        return bool(self.regions)

    @property
    def time_dependent(self):
        return any(r.geometry == 'stellar' for r in self.regions)

    @classmethod
    def from_dict(cls, value):
        return _construct(cls, value)
