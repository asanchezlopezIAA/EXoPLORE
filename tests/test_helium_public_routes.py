"""Self-contained checks of opt-in helium policies and OOT OH separation.

These checks create no files, download no data and launch no external solver.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from exoplore.config.helium_grid import HeliumGridConfig
from exoplore.config.helium_oh import validate_oh_treatment, uses_sky_fibre
from exoplore.core.spectroscopy_mode import spectroscopy_route
from exoplore.instruments.carmenes_oh import OHFitConfig, _sampled_profile
from exoplore.instruments.carmenes_oh_oot import OOTProfileConfig, fit_oot_profile


def oh_profile():
    return OHFitConfig((1083.2103,1083.2412),1083.43338,
        (1083.17,1083.28),(1083.39,1083.48),.015,(.002,.015),
        (0.,.015),.007,.001,6.,5,(-.012,.012),5)


def policy_config():
    profile = oh_profile()
    return SimpleNamespace(instrument=SimpleNamespace(name='CARMENES_NIR'),
        observation=SimpleNamespace(use_real_data=True,helium_transmission_spectroscopy=True),
        atmosphere=SimpleNamespace(helium=None),helium_grid=None,atomic_transmission=None,
        pipeline=SimpleNamespace(name='Palle2020',oh_treatment='fiber_a_oot_profile',
            palle2020={'oh_fit':{'profile':dict(profile.__dict__),
                'oot_profile':dict(fit_window_nm=[1083.39,1083.48],relative_svd_cutoff=1e-10,
                    minimum_sky_amplitude_cv=.05,maximum_reduced_chi2=3.,
                    maximum_profile_condition_number=1e4),'maximum_anchor_change_sigma':.01}},
            czesla2024=None,carmenes_helium=None))


class HeliumPolicyTests(unittest.TestCase):
    def test_fibre_a_option_requires_sky_and_observed_carmenes(self):
        cfg=policy_config()
        self.assertEqual(validate_oh_treatment(cfg),'fiber_a_oot_profile')
        self.assertTrue(uses_sky_fibre('fiber_a_oot_profile'))
        cfg.instrument.name='CRIRES+'
        with self.assertRaisesRegex(ValueError,'no fibre B'):
            validate_oh_treatment(cfg)
        cfg.instrument.name='CARMENES_NIR'
        cfg.observation.use_real_data=False
        with self.assertRaisesRegex(ValueError,'observed CARMENES'):
            validate_oh_treatment(cfg)

    def test_replaced_name_and_missing_settings_rejected(self):
        cfg=policy_config()
        cfg.pipeline.oh_treatment='fiber_b_oot_profile'
        with self.assertRaisesRegex(ValueError,'must be one of'):
            validate_oh_treatment(cfg)
        cfg=policy_config()
        cfg.pipeline.palle2020['oh_fit'].pop('oot_profile')
        with self.assertRaisesRegex(ValueError,'oot_profile settings'):
            validate_oh_treatment(cfg)

    def test_existing_choices_and_optional_matched_scale(self):
        for mode in ('ignore','mask','fiber_b_calibrated','fiber_b_matched_extraction'):
            cfg=policy_config();cfg.pipeline.oh_treatment=mode
            if mode=='fiber_b_matched_extraction':
                self.assertEqual(validate_oh_treatment(cfg),mode)
                cfg.pipeline.palle2020['oh_fit']['fiber_b_scale']=[.88,.89]
            self.assertEqual(validate_oh_treatment(cfg),mode)

    def test_standard_route_unchanged(self):
        cfg=policy_config()
        cfg.pipeline.name='Gibson2020'
        cfg.observation.helium_transmission_spectroscopy=False
        self.assertEqual(spectroscopy_route(cfg),'standard')

    def test_oot_observed_route(self):
        self.assertEqual(spectroscopy_route(policy_config()),'helium_observed')

    def test_explicit_sixteen_workers(self):
        atmosphere=dict(hydrogen_number_fraction=.998,radial_max_rp=30,
            irradiation_spectrum_path='incident_spectrum.dat',irradiation_source='Explicit test input',
            model_wavelength_range_nm=[1082.5,1084.5],model_wavelength_points=2001,
            line_of_sight_points=240,limb_darkening_law='none',limb_darkening_coefficients=[],
            model_edge_absorption_tolerance=1e-4,helium_vacuum_lines_nm=[1083.2057,1083.3217,1083.3306],
            helium_oscillator_strengths=[.0599,.1797,.2995],helium_atomic_source='Explicit test input',
            sunbather=dict(cloudy_path='cloudy',heating_cooling_factor=1.1,maximum_iterations=20,
                starting_temperature='constant',metal_scale=0.,cosmic_rays=True,stellar_tidal_gravity=True))
        fields=dict(enabled=True,signal_reviewed=True,input=dict(kind='exoplore',path='prepared',uncertainty='diagonal'),
            temperature_grid_K=[3000,4000],log10_mass_loss_grid_g_s=[10,11],atmosphere=atmosphere,
            resolving_power=80400.,fit_window_nm=[1083.1,1083.48],velocity_bounds_kms=[-20,20],
            velocity_grid_points=81,exposure_quadrature_points=5,instrumental_oversampling=4,
            covariance_eigenvalue_rtol=1e-10,workers=16,output_name='grid')
        self.assertEqual(HeliumGridConfig(**fields).workers,16)
        for invalid in (0,-1,True,1.5):
            with self.assertRaisesRegex(ValueError,'positive integer'):
                HeliumGridConfig(**dict(fields,workers=invalid))


class OOTSeparationTests(unittest.TestCase):
    def test_known_profile_and_transit_independence(self):
        n,pixels=12,2001
        wave=np.tile(np.linspace(1081,1084,pixels),(n,1))
        amplitudes=np.linspace(700,1600,n)
        centre,sigma,gamma,scale=1083.429,.0065,0.,1.4e-5
        template=np.array([_sampled_profile(w,[centre],sigma,gamma,5) for w in wave])
        raw=.1+scale*amplitudes[:,None]*template
        errors=np.full_like(raw,.0003)
        sky=dict(amplitudes=amplitudes,amplitude_variance=np.ones(n),
            master_parameters={'strong':dict(centres_nm=[centre],sigma_nm=.007,lorentz_hwhm_nm=.001)})
        initial_profile=np.array([_sampled_profile(w,[centre],.007,.001,5) for w in wave])
        base=dict(transfer_scale=1.5e-5,transfer_shift_nm=0.,model_a=1.5e-5*amplitudes[:,None]*initial_profile)
        science=SimpleNamespace(gamma_kms=0.,telluric_min_transmission=.15,
            minimum_continuum_pixels=30,continuum_stellar_windows_nm=[[1081.5962,1082.7624]])
        config=OOTProfileConfig([1083.39,1083.48],1e-10,.05,3.,1e4)
        refs=np.arange(8)
        def fit(flux):
            return fit_oot_profile(wave,wave,flux,errors,np.ones_like(raw),np.zeros(n),
                refs,science,sky,base,oh_profile(),config)
        value=fit(raw)
        self.assertTrue(value['accepted'],value['rejection_reasons'])
        self.assertAlmostEqual(value['transfer_scale']/scale,1.,places=3)
        self.assertAlmostEqual(value['gaussian_sigma_nm']/sigma,1.,places=3)
        self.assertLess(abs(value['strong_centre_nm']-centre),1e-6)
        altered=raw.copy();altered[8:]+=10*errors[8:]
        other=fit(altered)
        for key in ('transfer_scale','strong_centre_nm','gaussian_sigma_nm','lorentz_hwhm_nm'):
            self.assertEqual(value[key],other[key])
        np.testing.assert_array_equal(raw,.1+scale*amplitudes[:,None]*template)


if __name__=='__main__':
    unittest.main()
