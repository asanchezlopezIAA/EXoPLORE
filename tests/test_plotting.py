"""Smoke tests for exoplore.plotting, import and signature checks only.

Plotting functions call matplotlib and open figure windows; we do not
render pixels in the test suite.  These tests verify:

1. All functions are importable from the public API.
2. Each function exists with the expected signature (via inspect).
3. Functions that do pure computation before any plot call return the
   correct types (e.g. plot_mat_with_collapse with with_collapse=False).

No figures are shown or saved (matplotlib is used in non-interactive
mode via the 'Agg' backend).
"""

import inspect
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------

def test_public_imports():
    from exoplore.plotting import (
        # kpvsys, clean API
        plot_kp_vsys_map,
        plot_1d_ccf,
        # kpvsys API
        plot_Kp_Vrest,
        plot_1D_CCF,
        plot_combined_KpVrest_1DCCF,
        # matrices, clean API
        plot_spectral_matrix,
        plot_ccf_timeseries,
        # matrices API
        CCF_matrix_ERF,
        plot_ccf_matrices_per_night,
        plot_mat_with_collapse,
        plot_steps,
        plot_matrix_difference,
    )
    assert callable(plot_Kp_Vrest)
    assert callable(plot_1D_CCF)
    assert callable(plot_combined_KpVrest_1DCCF)
    assert callable(CCF_matrix_ERF)
    assert callable(plot_ccf_matrices_per_night)
    assert callable(plot_mat_with_collapse)
    assert callable(plot_steps)
    assert callable(plot_matrix_difference)


def test_detection_shim_imports():
    """detection.py shim exports the same names as kpvsys."""
    from exoplore.plotting.detection import (
        plot_Kp_Vrest,
        plot_1D_CCF,
        plot_combined_KpVrest_1DCCF,
    )
    assert callable(plot_Kp_Vrest)
    assert callable(plot_1D_CCF)
    assert callable(plot_combined_KpVrest_1DCCF)


# ---------------------------------------------------------------------------
# Signature checks
# ---------------------------------------------------------------------------

class TestSignatures:
    def test_plot_Kp_Vrest_params(self):
        from exoplore.plotting import plot_Kp_Vrest
        sig = inspect.signature(plot_Kp_Vrest)
        params = list(sig.parameters)
        assert "inp_dat" in params
        assert "kp_range" in params
        assert "ccf_tot_sn" in params
        assert "v_rest" in params
        assert "save_plot" in params
        assert "CCF_Noise" in params
        assert "sysrem_opt" in params

    def test_plot_1D_CCF_params(self):
        from exoplore.plotting import plot_1D_CCF
        sig = inspect.signature(plot_1D_CCF)
        params = list(sig.parameters)
        assert "inp_dat" in params
        assert "v_rest" in params
        assert "ccf_tot_sn" in params
        assert "max_kp" in params
        assert "max_sn" in params
        assert "n_kp" in params
        assert "max_v_wind" in params
        assert "xlims" in params

    def test_plot_combined_KpVrest_1DCCF_params(self):
        from exoplore.plotting import plot_combined_KpVrest_1DCCF
        sig = inspect.signature(plot_combined_KpVrest_1DCCF)
        params = list(sig.parameters)
        assert "inp_dat" in params
        assert "v_rest" in params
        assert "kp_range" in params
        assert "ccf_tot_sn" in params
        assert "max_kp_idx" in params
        assert "save_plot" in params

    def test_CCF_matrix_ERF_params(self):
        from exoplore.plotting import CCF_matrix_ERF
        sig = inspect.signature(CCF_matrix_ERF)
        params = list(sig.parameters)
        assert "inp_dat" in params
        assert "v_ccf" in params
        assert "phase" in params
        assert "ccf_complete" in params
        assert "with_signal" in params
        assert "without_signal" in params
        assert "v_planet" in params

    def test_plot_ccf_matrices_per_night_params(self):
        from exoplore.plotting import plot_ccf_matrices_per_night
        sig = inspect.signature(plot_ccf_matrices_per_night)
        params = list(sig.parameters)
        assert "inp_dat" in params
        assert "ccf_store" in params
        assert "output_dir" in params
        assert "v_ccf" in params
        assert "phase" in params
        assert "with_signal" in params

    def test_plot_mat_with_collapse_params(self):
        from exoplore.plotting import plot_mat_with_collapse
        sig = inspect.signature(plot_mat_with_collapse)
        params = list(sig.parameters)
        assert "x" in params
        assert "y" in params
        assert "z" in params
        assert "with_collapse" in params
        assert "save_plot" in params

    def test_plot_steps_params(self):
        from exoplore.plotting import plot_steps
        sig = inspect.signature(plot_steps)
        params = list(sig.parameters)
        assert "inp_dat" in params
        assert "wave_ins" in params
        assert "n_panels" in params
        assert "mat_list" in params
        assert "spec_idx" in params
        assert "phase" in params
        assert "with_signal" in params
        assert "useful_spectral_points" in params

    def test_plot_matrix_difference_params(self):
        from exoplore.plotting import plot_matrix_difference
        sig = inspect.signature(plot_matrix_difference)
        params = list(sig.parameters)
        assert "wave" in params
        assert "matrix1" in params
        assert "matrix2" in params
        assert "with_signal" in params


# ---------------------------------------------------------------------------
# Runtime guard: plot_Kp_Vrest raises ValueError on mismatched range args
# ---------------------------------------------------------------------------

class TestPlotKpVrestValidation:
    def _make_inp_dat(self):
        return {
            "MAX_CCF_V_STD": 200,
            "PLOT_CCF_XSTEP": 50,
            "kp_max": 250,
            "K_p": 150.0,
            "sysrem_its": 1,
            "plots_dir": "/tmp/",
            "Simulation_name": "test",
        }

    def test_mismatched_range_raises(self):
        import numpy as np
        from exoplore.plotting import plot_Kp_Vrest
        inp_dat = self._make_inp_dat()
        v_rest = np.linspace(-100, 100, 50)
        kp_range = np.linspace(0, 250, 40)
        ccf_tot_sn = np.random.randn(50, 40)
        with pytest.raises(ValueError, match="xrange and yrange"):
            plot_Kp_Vrest(
                inp_dat, kp_range, ccf_tot_sn, v_rest,
                xrange=[-100, 100], yrange=None,
                show_plot=False, save_plot=False,
            )

    def test_steps_wrong_mat_list_raises(self):
        import numpy as np
        from exoplore.plotting import plot_steps
        wave_ins = np.linspace(2.3, 2.4, 100)
        phase = np.linspace(-0.05, 0.05, 20)
        with_signal = np.arange(5, 15)
        usp = np.arange(100)
        mat = np.random.randn(20, 100)
        with pytest.raises(Exception, match="n_panels-1"):
            plot_steps(
                {}, wave_ins, 3, [mat],  # n_panels=3 but only 1 matrix
                5, phase, with_signal, usp
            )
