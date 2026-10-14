"""Tests of the WP7 tools: well tie (iwt.welltie), the integrated loop (iwt.workflow),
contamination by multiples (iwt.multiples), linear resolution (iwt.avo) and the
dipping reservoir section (iwt.synthetic)."""

import numpy as np
import pytest

from iwt import avo
from iwt import multiples as M
from iwt import raytrace as RT
from iwt import welltie as T
from iwt import workflow as WF
from iwt.model import ElasticModel1D, phase_velocities
from iwt.modeling.convolution import ricker
from iwt.modeling.reflection import exact_pp, exact_pp_slowness
from iwt.qmodel import phase_velocity
from iwt.synthetic import (OWC_TVDSS, SEA_WATER, WellLogs, dipping_reservoir_models,
                           wisting_like_well)

DT = 0.002


def _reflectivity(n, seed=1):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    k = rng.choice(np.arange(30, n - 30), 25, replace=False)
    r[k] = rng.normal(0.0, 0.1, k.size)
    return r


def _convolve(r, w):
    L = (w.size - 1) // 2
    return np.convolve(r, w)[L:L + r.size]


# -- well tie ---------------------------------------------------------------------------


def test_estimate_wavelet_recovers_a_rotated_wavelet():
    t, w = ricker(25.0, DT)
    w = T.rotate_phase(np.pad(w, 20), np.radians(40.0))[20:-20]   # padded: no wrap-around
    rs = np.array([_reflectivity(500, s) for s in (1, 2)])
    d = np.array([_convolve(r, w) for r in rs])
    tw, we = T.estimate_wavelet(d, rs, DT, t[-1], damping=1e-8)
    np.testing.assert_allclose(tw, t, atol=1e-12)
    np.testing.assert_allclose(we, w, atol=1e-6 * np.abs(w).max())
    # noise: the estimate degrades gracefully
    rng = np.random.default_rng(3)
    dn = d + 0.05 * d.std() * rng.standard_normal(d.shape)
    _, wn = T.estimate_wavelet(dn, rs, DT, t[-1])
    assert np.linalg.norm(wn - w) / np.linalg.norm(w) < 0.1


def test_tie_metrics_separates_shift_and_phase():
    _, w = ricker(30.0, DT)
    s = _convolve(_reflectivity(600), w)
    for k, phi in ((0, 0.0), (4, np.radians(35.0)), (-3, np.radians(-60.0))):
        o = 2.0 * T.rotate_phase(np.roll(s, k), phi)
        m = T.tie_metrics(o, s, DT, max_shift=0.02)
        assert m.shift == pytest.approx(k * DT, abs=0.05 * DT)
        assert m.phase == pytest.approx(phi, abs=np.radians(1.0))
        assert m.max_correlation == pytest.approx(1.0, abs=1e-3)
        if k == 0 and phi == 0.0:
            assert m.correlation == pytest.approx(1.0)
            assert m.scale == pytest.approx(2.0)
            assert m.pep == pytest.approx(1.0)
        else:
            assert m.correlation < 0.9
    # a pure time shift is not mistaken for a phase rotation (and vice versa)
    m = T.tie_metrics(np.roll(s, 5), s, DT)
    assert abs(m.phase) < np.radians(1.0)
    m = T.tie_metrics(T.rotate_phase(s, np.radians(90.0)), s, DT)
    assert abs(m.shift) < 0.05 * DT


# -- model from logs --------------------------------------------------------------------


def _logs():
    z = np.arange(470.0, 520.01, 0.5)
    rng = np.random.default_rng(0)
    vp = 2400.0 * (1.0 + 0.03 * rng.standard_normal(z.size))
    vs = 1000.0 * (1.0 + 0.03 * rng.standard_normal(z.size))
    rho = 2200.0 + 20.0 * rng.standard_normal(z.size)
    return WellLogs(z, vp, vs, rho, meta=dict(f_sonic_p=12e3, f_sonic_s=2e3))


def test_model_from_logs_preserves_sonic_time_and_disperses():
    logs = _logs()
    z = logs.depth
    m = WF.model_from_logs(logs, 418.0, SEA_WATER.vp0, SEA_WATER.rho)
    np.testing.assert_allclose(m.interfaces, np.concatenate([[0.0, 418.0], z]))
    # layers between samples: the vertical time is the trapezoidal integral of slowness
    t = m.vertical_time(z[-1]) - m.vertical_time(z[0])
    assert float(t) == pytest.approx(np.sum(0.5 * (1.0 / logs.vp[1:] + 1.0 / logs.vp[:-1]) * np.diff(z)), rel=1e-12)
    # the gap: means of the first 10 m, or the given velocity
    top = z < z[0] + 10.0
    assert m.vp0[1] == pytest.approx(np.mean(logs.vp[top]))
    assert m.rho[1] == pytest.approx(np.mean(logs.rho[top]))
    assert m.qp[0] == np.inf and m.vs0[0] == 0.0
    mg = WF.model_from_logs(logs, 418.0, SEA_WATER.vp0, SEA_WATER.rho, gap_vp=2100.0)
    assert mg.vp0[1] == 2100.0
    np.testing.assert_array_equal(mg.vp0[2:], m.vp0[2:])
    # Q: Vp moved from 12 kHz to 30 Hz (slower), Vs too unless disperse_vs=False
    q = WF.model_from_logs(logs, 418.0, SEA_WATER.vp0, SEA_WATER.rho, qp=50.0, qs=25.0,
                           eps=0.1, delta=np.full(z.size, 0.05))
    vp30 = phase_velocity(logs.vp[:2], 50.0, 30.0, 12e3)
    assert q.vp0[2] == pytest.approx(2.0 / np.sum(1.0 / vp30))
    assert np.all(q.vp0[2:] < m.vp0[2:]) and np.all(q.vs0[2:] < m.vs0[2:])
    nv = WF.model_from_logs(logs, 418.0, SEA_WATER.vp0, SEA_WATER.rho, qp=50.0, qs=25.0,
                            disperse_vs=False)
    np.testing.assert_allclose(nv.vs0[2:], m.vs0[2:])
    assert np.all(q.qp[1:] == 50.0) and np.all(q.qs[1:] == 25.0)
    assert np.all(q.eps[2:] == 0.1) and np.all(q.delta[2:] == 0.05) and q.eps[1] == 0.0


def test_register_and_criteria():
    reg = WF.Register()
    m = ElasticModel1D([0.0, 100.0, 200.0], vp0=[1500.0, 2000.0], vs0=[0.0, 800.0],
                       rho=[1000.0, 2000.0])
    v0 = reg.new_version("logs", m, "as logged")
    v1 = reg.new_version("calibrated", m.copy(), "Q | dispersion")
    assert (v0.number, v1.number) == (0, 1) and reg.current is v1
    reg.log("late synthetic", "gap velocity", "VSP gap", "shift 1 ms", "adopt", 1)
    reg.log("AVA erratic", "multiples", "Born vs full", "0.9 x Born", "explain")
    assert reg.check("tie", 0.93, 0.8, ">=").passed
    assert not reg.check("drift", 1.5, 1.0, "<=", "ms").passed
    with pytest.raises(ValueError):
        reg.check("bad", 1.0, 1.0, "==")
    assert len(reg.criteria) == 2
    md = reg.to_markdown()
    assert "| v1 | calibrated |" in md
    assert "| 1 | late synthetic |" in md and "| adopt | v1 |" in md and "| – |" in md
    assert "| tie | 0.93 | >= 0.8 | yes |" in md and "| drift | 1.5 ms | <= 1 ms | **no** |" in md
    reg.check("|ratio - 1|", 0.5, 0.1)
    assert "| \\|ratio - 1\\| | 0.5 |" in reg.to_markdown()      # pipes escaped


# -- contamination ----------------------------------------------------------------------


def test_relative_rms():
    b = np.sin(np.linspace(0.0, 10.0, 200))
    assert M.relative_rms(1.1 * b, b) == pytest.approx(0.1)
    a = b.copy()
    a[:100] += 1.0
    assert M.relative_rms(a, b, mask=np.arange(200) >= 100) == pytest.approx(0.0)
    a = 1.2 * b
    a[5] = np.nan
    assert M.relative_rms(a, b) == pytest.approx(0.2, rel=0.02)


def test_gathers_switch_physics():
    m = ElasticModel1D([0.0, 200.0, 260.0, 600.0], vp0=[1500.0, 2200.0, 3000.0],
                       vs0=[0.0, 900.0, 1500.0], rho=[1000.0, 2100.0, 2400.0])
    dt, nt = 0.002, 400
    g = M.gathers(m, [50.0], 5.0, 5.0, dt, nt, ricker(25.0, dt), attenuation=False, f_max=60.0)
    assert list(g) == list(M.PHYSICS)
    t = np.arange(nt) * dt
    seabed = (t > 0.24) & (t < 0.295)       # 0.262 s: identical in every variant
    base = (t > 0.30) & (t < 0.33)          # 0.316 s: below the seabed transmission
    for k in M.PHYSICS:
        assert M.relative_rms(g[k][0, seabed], g["full"][0, seabed]) < 0.03
    # unit transmission (Born) strengthens the deeper primary by about 1 / (1 - R^2)
    r = (2100.0 * 2200.0 - 1000.0 * 1500.0) / (2100.0 * 2200.0 + 1000.0 * 1500.0)
    ratio = (np.abs(g["Born primaries"][0, base]).max()
             / np.abs(g["primaries with transmission loss"][0, base]).max())
    assert ratio == pytest.approx(1.0 / (1.0 - r * r), rel=0.15)


# -- resolution -------------------------------------------------------------------------


def test_linear_resolution():
    J = np.eye(3)
    r = avo.linear_resolution(J, [1.0, 2.0, 3.0], 0.5 ** 2 * np.eye(3))
    np.testing.assert_allclose(r.posterior_std ** 2, 1.0 / (4.0 + 1.0 / np.array([1, 4, 9])))
    np.testing.assert_allclose(r.data_weight, 1.0 - r.posterior_std ** 2 / np.array([1, 4, 9]))
    np.testing.assert_allclose(r.correlation, np.eye(3), atol=1e-12)
    # a parameter the data cannot see keeps its prior; two the data confuse trade off
    x = np.linspace(-1.0, 1.0, 20)
    J = np.column_stack([x, 1.001 * x, np.zeros_like(x)])
    r = avo.linear_resolution(J, [1.0, 1.0, 1.0], 0.01 ** 2 * np.eye(x.size))
    assert r.data_weight[2] == pytest.approx(0.0)
    assert r.correlation[0, 1] < -0.99


# -- kinematic and coefficient helpers --------------------------------------------------


def test_reflection_table_zero_offset_fast_path():
    m = ElasticModel1D([0.0, 300.0, 340.0, 420.0, 600.0], vp0=[1500.0, 2400.0, 2900.0, 2600.0],
                       vs0=[0.0, 1000.0, 1400.0, 1200.0], rho=[1000.0, 2200.0, 2350.0, 2300.0],
                       eps=[0.0, 0.15, 0.05, 0.1], delta=[0.0, 0.05, 0.02, 0.04],
                       qp=[np.inf, 40.0, 80.0, 60.0])
    z = np.array([320.0, 340.0, 400.0, 500.0])
    fast = RT.reflection_table(m, z, [0.0], 6.0, 8.0)
    general = RT.reflection_table(m, z, [0.0, 200.0], 6.0, 8.0)
    for a, b in zip(fast[2:], general[2:]):
        np.testing.assert_allclose(a[:, 0], b[:, 0], rtol=1e-9, atol=1e-15)


def test_exact_pp_slowness_matches_angle_form():
    up, lo = (2400.0, 1000.0, 2200.0, 0.15, 0.05), (2900.0, 1500.0, 2350.0, 0.05, 0.02)
    th = np.radians(np.linspace(0.0, 40.0, 9))
    V = phase_velocities(th, up[0], up[1], up[3], up[4], 0.0)[0]
    np.testing.assert_allclose(exact_pp_slowness(np.sin(th) / V, up, lo), exact_pp(th, up, lo),
                               rtol=1e-12)
    # beyond the critical slowness of the lower medium the coefficient is complex
    pc = 1.0 / (lo[0] * np.sqrt(1.0 + 2.0 * lo[3]))
    r = exact_pp_slowness([0.5 * pc, 1.05 * pc], up, lo)
    assert abs(r[0].imag) < 1e-12 and abs(r[1].imag) > 1e-3


# -- the Fig. 5 analogue ----------------------------------------------------------------


def test_dipping_reservoir_models():
    xs = np.array([0.0, 500.0, 1000.0])
    oil = dipping_reservoir_models(xs, dz=0.5, td=760.0)
    brine = dipping_reservoir_models(xs, dz=0.5, td=760.0, oil=False)
    shift = xs * np.tan(np.radians(1.2))
    for m, b, s in zip(oil, brine, shift):
        zc = np.arange(560.0, 759.0, 0.5) + 0.25
        po, pb = m.at(zc), b.at(zc)
        fluid = (zc > 672.0 + s) & (zc < OWC_TVDSS)
        # only the oil column differs; the hydrocarbon lowers Vp and density
        assert np.all(po["vp0"][fluid] < pb["vp0"][fluid])
        assert np.all(po["rho"][fluid] < pb["rho"][fluid])
        np.testing.assert_array_equal(po["vp0"][~fluid], pb["vp0"][~fluid])
        # the reservoir top dips with the wedge, the contact stays flat
        assert m.interfaces[np.searchsorted(m.interfaces, 672.0 + s - 0.1)] == pytest.approx(
            672.0 + s, abs=0.5)
        assert np.any(np.isclose(m.interfaces, OWC_TVDSS))
    # units above the wedge are untouched
    a, b = oil[0].at([500.0, 590.0]), oil[2].at([500.0, 590.0])
    np.testing.assert_array_equal(a["vp0"], b["vp0"])
    # the oil column (22.5 m at x = 0) thins to 1.5 m at 1 km
    assert OWC_TVDSS - 672.0 == pytest.approx(22.5)
    assert OWC_TVDSS - (672.0 + shift[-1]) == pytest.approx(1.5, abs=0.1)


def test_model_from_wisting_logs_matches_truth_time():
    model, logs = wisting_like_well()
    m = WF.model_from_logs(logs, 418.0, SEA_WATER.vp0, SEA_WATER.rho, disperse_vs=False)
    z = logs.depth
    t_true = model.vertical_time(z[-1]) - model.vertical_time(z[0])
    t_logs = m.vertical_time(z[-1]) - m.vertical_time(z[0])
    # logged at 12 kHz: faster than the 30 Hz truth (sonic drift of order 10 ms)
    assert 0.0 < float(t_true - t_logs) < 0.02
