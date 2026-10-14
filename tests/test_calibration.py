"""Tests of the sonic-to-seismic calibration (iwt.calibration)."""

import numpy as np
import pytest

from iwt import calibration as C
from iwt.qmodel import phase_velocity

Z_LOG = np.arange(500.0, 1500.01, 0.5)


def _blocky(z, edges, values):
    return np.asarray(values, float)[np.clip(np.searchsorted(edges, z, side="right") - 1,
                                             0, len(values) - 1)]


def test_sonic_time_constant_velocity():
    t = C.sonic_time(Z_LOG, np.full(Z_LOG.size, 2500.0))
    assert np.allclose(t, (Z_LOG - Z_LOG[0]) / 2500.0)
    assert np.isnan(C.sonic_time(Z_LOG, np.full(Z_LOG.size, 2500.0), [400.0])[0])


def test_drift_recovers_known_dispersion():
    """Sonic at 12 kHz, VSP at 30 Hz, known Q: the drift and its physical correction are exact."""
    edges = np.array([500.0, 800.0, 1500.0])
    v30 = _blocky(Z_LOG, edges, [2400.0, 3000.0])
    q = _blocky(Z_LOG, edges, [40.0, 80.0])
    v_sonic = phase_velocity(v30, q, 12e3, 30.0)
    z_vsp = np.arange(500.0, 1500.1, 10.0)
    t_vsp = 0.4 + C.sonic_time(Z_LOG, v30, z_vsp)
    d = C.drift(Z_LOG, v_sonic, z_vsp, t_vsp)
    expected = C.sonic_time(Z_LOG, v30, z_vsp) - C.sonic_time(Z_LOG, v_sonic, z_vsp)
    assert np.allclose(d, expected, atol=1e-12)
    assert d[-1] > 0.010                     # about 4 % over 0.4 s: a sizeable drift
    v_cal = C.physical_calibration(Z_LOG, v_sonic, np.full(Z_LOG.size, 2300.0), q, 12e3, 30.0)
    assert np.allclose(v_cal, v30, rtol=1e-12)
    assert np.max(np.abs(C.drift(Z_LOG, v_cal, z_vsp, t_vsp))) < 1e-12


@pytest.mark.parametrize("mode", ["shift", "scale"])
def test_knee_calibration_removes_piecewise_linear_drift(mode):
    edges = np.array([500.0, 800.0, 1100.0, 1500.0])
    v_true = _blocky(Z_LOG, edges, [2400.0, 3000.0, 2700.0])
    grad = np.array([10e-6, 4e-6, 7e-6])               # drift gradient, s/m
    s_sonic = 1.0 / v_true - _blocky(Z_LOG, edges, grad)
    z_vsp = np.arange(500.0, 1500.1, 10.0)
    t_vsp = C.sonic_time(Z_LOG, v_true, z_vsp)
    kc = C.knee_calibration(Z_LOG, 1.0 / s_sonic, z_vsp, t_vsp, knees=[800.0, 1100.0], mode=mode)
    assert np.allclose(kc.nodes, [500.0, 800.0, 1100.0, 1500.0])
    assert np.allclose(np.diff(kc.node_drift) / np.diff(kc.nodes), grad, rtol=1e-6)
    # exact up to the trapezoidal integration across the block boundaries
    # (about 1 us on a drift of 7.5 ms)
    assert np.max(np.abs(kc.residual)) < 2e-6
    assert np.allclose(kc.velocity, v_true, rtol=1e-5)


def test_knee_calibration_without_knees_is_linear():
    v = np.full(Z_LOG.size, 2500.0)
    z_vsp = np.arange(500.0, 1500.1, 10.0)
    t_vsp = C.sonic_time(Z_LOG, v, z_vsp) * 1.01
    kc = C.knee_calibration(Z_LOG, v, z_vsp, t_vsp)
    assert kc.nodes.size == 2
    assert np.allclose(kc.velocity, 2500.0 / 1.01)


def test_backus_vertical_velocity():
    """Alternating thin layers: Backus velocity = harmonic C33 average, below the time average."""
    z = np.arange(0.0, 198.0, 0.5)
    vp = np.where(np.arange(z.size) % 3 == 0, 2000.0, 3000.0)   # 0.5 m and 1 m layers
    rho = np.where(vp > 2500.0, 2400.0, 2100.0)
    vb = C.backus_vertical_velocity(z, vp, rho, 16.5)          # 33 samples = 11 periods
    m33 = 1.0 / np.mean(1.0 / (rho * vp ** 2))
    v_backus = np.sqrt(m33 / np.mean(rho))
    v_time = 1.0 / np.mean(1.0 / vp)
    mid = slice(100, 300)
    assert np.allclose(vb[mid], v_backus, rtol=1e-12)
    assert v_backus < v_time
    assert np.allclose(C.backus_vertical_velocity(z, vp, rho, 0.0), vp)


def test_physical_calibration_anisotropy_term():
    v = np.full(10, 3000.0)
    th = np.radians(30.0)
    out = C.physical_calibration(np.arange(10.0), v, np.full(10, 2300.0), np.inf, 12e3, 30.0,
                                 inclination_deg=30.0, eps=0.1, delta=0.05)
    s2, c2 = np.sin(th) ** 2, np.cos(th) ** 2
    assert np.allclose(out, 3000.0 / (1 + 0.05 * s2 * c2 + 0.1 * s2 * s2))
    same = C.physical_calibration(np.arange(10.0), v, np.full(10, 2300.0), np.inf, 12e3, 30.0)
    assert np.allclose(same, v)


def test_knee_calibration_mode_check():
    v = np.full(Z_LOG.size, 2500.0)
    with pytest.raises(ValueError):
        C.knee_calibration(Z_LOG, v, [600.0, 900.0], [0.1, 0.2], mode="stretch")


# -- gamma -> (eps, delta) calibration (WP5) ------------------------------------------


def test_upscale_gamma_backus():
    z = np.arange(0.0, 200.0, 0.1)
    c44 = np.where((z // 1.0) % 2 == 0, 5e9, 15e9)            # 1 m beds, isotropic each
    g = C.upscale_gamma(z, c44, c44, 20.0)
    # C66 arithmetic, C44 harmonic mean: gamma = (10 / 7.5 - 1) / 2 = 1/6
    np.testing.assert_allclose(g[500:1500], 1.0 / 6.0, rtol=1e-2)
    flat = C.upscale_gamma(z, np.full(z.size, 4e9), np.full(z.size, 5e9), 20.0)
    np.testing.assert_allclose(flat[100:-100], 0.125, rtol=1e-12)


def test_bayes_linear_recovers_line_and_errors():
    rng = np.random.default_rng(0)
    x = np.linspace(0.05, 0.30, 6)
    s = np.full(6, 0.01)
    z = []
    for _ in range(400):
        y = 0.01 + 0.8 * x + s * rng.standard_normal(6)
        cal = C.bayes_linear(x, y, s, scatter=0.0)
        z.append((cal.coef - [0.01, 0.8]) / np.sqrt(np.diag(cal.coef_cov)))
    z = np.array(z)
    assert np.all(np.abs(z.mean(axis=0)) < 0.15) and np.all(np.abs(z.std(axis=0) - 1) < 0.1)
    # prediction error grows away from the calibrated range (extrapolation)
    _, sd = cal.predict(np.array([0.175, 0.30, 0.6]), scatter=False)
    assert sd[0] < sd[1] < sd[2]
    # leave-one-out predictions are honest: z-scores of order one
    assert np.all(np.abs(cal.loo_z) < 4.0) and cal.loo_mean.shape == (6,)


def test_bayes_linear_estimates_extra_scatter():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 0.3, 60)
    s = np.full(60, 0.005)
    y = 0.02 + 0.7 * x + np.hypot(0.02, 0.005) * rng.standard_normal(60)
    cal = C.bayes_linear(x, y, s)
    assert 0.014 < cal.scatter < 0.028
    assert np.all(np.abs(cal.loo_z) < 4.0) and 0.7 < cal.loo_z.std() < 1.3
    # with the scatter ignored the leave-one-out residuals are far too large
    assert C.bayes_linear(x, y, s, scatter=0.0).loo_z.std() > 2.5
