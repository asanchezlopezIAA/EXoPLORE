"""Physical limiting cases for the RM and CLV surface integrator."""

import numpy as np
import pytest

from exoplore.stellar import LocalIntensityGrid, StellarSurface


def grid(*, clv=False, continuum_darkening=False):
    """Use an analytic local line so each limiting case has a known behaviour."""
    wave = np.linspace(0.588, 0.592, 801)
    mu = np.array([0.0, 0.5, 1.0])
    continuum = np.ones((len(mu), len(wave)))
    if continuum_darkening:
        continuum *= (0.4 + 0.6*mu[:, None])
    depth = (0.2 + 0.5*mu[:, None]) if clv else np.full((len(mu), 1), 0.5)
    profile = np.exp(-0.5*((wave - 0.59)/0.000025)**2)
    return LocalIntensityGrid(wave, mu, continuum*(1 - depth*profile), continuum)


def surface(local=None, *, speed=0, angle=0, cells=48, subpixels=1):
    return StellarSurface(grid() if local is None else local, vsini_kms=speed,
                          projected_obliquity_deg=angle,
                          cells_across_diameter=cells, subpixels_per_axis=subpixels)


def test_unocculted_and_zero_radius_spectra_match_disk():
    model = surface(grid(clv=True), speed=15)
    wave = np.linspace(0.589, 0.591, 101)
    outside = model.integrate(wave, [2], [0], 0.15)
    zero = model.integrate(wave, [0], [0], 0)
    np.testing.assert_allclose(outside.inside[0], outside.outside, rtol=1e-14)
    np.testing.assert_allclose(zero.inside[0], zero.outside, rtol=1e-14)
    np.testing.assert_allclose(outside.line_distortion(), 1, atol=1e-14)


def test_static_identical_line_profiles_have_no_line_distortion():
    # Continuum limb darkening changes the light curve while the line shape
    # remains the same everywhere in this particular control experiment.
    result = surface(grid(continuum_darkening=True)).integrate(
        [0.5899, 0.59, 0.5901], [-0.7, 0, 0.7], [0, 0, 0], 0.2)
    assert np.all(result.continuum_inside < result.continuum_outside)
    np.testing.assert_allclose(result.line_distortion(), 1, atol=1e-14)


def test_uniform_disk_depth_converges_to_radius_ratio_squared():
    # A central opaque disk blocks its area fraction on a uniform star.
    errors = []
    for cells in [20, 200]:
        result = surface(cells=cells).integrate([0.589], [0], [0], 0.2)
        depth = 1 - result.continuum_inside[0, 0]/result.continuum_outside[0]
        errors.append(abs(depth - 0.2**2))
    assert errors[-1] < 0.0005
    assert errors[-1] < errors[0]


def test_limb_darkened_continuum_integral():
    # For I(mu)=0.4+0.6*mu, the analytic disk integral is pi*(0.4+0.6*2/3).
    result = surface(grid(continuum_darkening=True), cells=200).integrate(
        [0.589], [2], [0], 0.2)
    np.testing.assert_allclose(result.continuum_outside, np.pi*0.8, rtol=0.001)


def test_clv_produces_line_distortion_without_rotation():
    result = surface(grid(clv=True)).integrate([0.589, 0.59, 0.591], [0], [0], 0.2)
    # Blocking the deeper centre line leaves a shallower disk-integrated line.
    assert result.line_distortion()[0, 1] > 1.005
    np.testing.assert_allclose(result.line_distortion()[0, [0, 2]], 1, atol=1e-14)


def test_rotation_gives_mirror_symmetric_shadows():
    result = surface(speed=20).integrate(np.linspace(0.5897, 0.5903, 121),
                                         [-0.5, 0.5], [0, 0], 0.2)
    distortion = result.line_distortion()
    assert np.ptp(distortion[0]) > 0.005
    # Lambda-linear shifts on a wavelength grid give approximate velocity
    # symmetry at this narrow line. This tolerance covers that approximation.
    np.testing.assert_allclose(distortion[0], distortion[1, ::-1], atol=0.00005)


def test_rotation_coordinate_is_finite_and_obliquity_rotates_axis():
    aligned = surface(speed=20, cells=21)
    polar = surface(speed=20, angle=90, cells=21)
    assert np.isfinite(aligned.velocity_kms).all()
    np.testing.assert_allclose(aligned.velocity_kms, 20*aligned.x)
    np.testing.assert_allclose(polar.velocity_kms, -20*polar.y, atol=1e-14)


def test_intensity_scaling_cancels_in_normalized_distortion():
    original = grid(clv=True, continuum_darkening=True)
    scaled = LocalIntensityGrid(original.wavelength_um, original.mu,
                                original.intensity*137, original.continuum*137)
    kwargs = dict(planet_x=[0.4], planet_y=[0.1], radius_ratio=0.2)
    wave = np.linspace(0.5895, 0.5905, 51)
    a = surface(original, speed=20).integrate(wave, **kwargs)
    b = surface(scaled, speed=20).integrate(wave, **kwargs)
    np.testing.assert_allclose(a.line_distortion(), b.line_distortion(), atol=1e-14)


def test_chunk_size_and_subpixel_representation_preserve_result():
    wave = np.linspace(0.5895, 0.5905, 51)
    a = surface(grid(clv=True), speed=10, cells=20, subpixels=2)
    b = surface(grid(clv=True), speed=10, cells=40)
    first = a.integrate(wave, [0.3], [0.1], 0.2, wavelength_chunk_size=3)
    second = b.integrate(wave, [0.3], [0.1], 0.2, wavelength_chunk_size=51)
    np.testing.assert_allclose(first.inside, second.inside, atol=1e-14)


def test_missing_mu_coverage_fails_instead_of_wrapping_to_last_angle():
    original = grid()
    clipped = LocalIntensityGrid(original.wavelength_um, [0.5, 1],
                                 original.intensity[1:], original.continuum[1:])
    with pytest.raises(ValueError, match="mu grid"):
        surface(clipped)


def test_missing_wavelength_margin_fails_instead_of_clamping_edges():
    with pytest.raises(ValueError, match="wider wavelength"):
        surface(speed=100).integrate([0.588, 0.592], [0], [0], 0.2)


@pytest.mark.parametrize("attribute,value", [
    ("mu", [0, 1, 0.5]), ("mu", [0, 0.5, 1.1]),
    ("intensity", np.zeros((2, 5))), ("continuum", np.zeros((3, 801))),
    ("wavelength_um", [0.59, 0.58]),
])
def test_invalid_local_grid_fails(attribute, value):
    original = grid()
    values = {name: getattr(original, name) for name in
              ("wavelength_um", "mu", "intensity", "continuum")}
    values[attribute] = value
    with pytest.raises(ValueError):
        LocalIntensityGrid(**values)


def test_local_grid_owns_readonly_arrays():
    original = grid()
    source = original.intensity.copy()
    copied = LocalIntensityGrid(original.wavelength_um, original.mu,
                                source, original.continuum)
    source[:] = 0
    assert copied.intensity.max() > 0
    with pytest.raises(ValueError):
        copied.intensity[:] = 0
