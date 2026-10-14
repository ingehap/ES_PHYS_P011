"""Tests of the reflection ray tables (iwt.raytrace), Q-compensated migration (iwt.imaging)
and AVA extraction (iwt.avo)."""

import numpy as np
import pytest
from scipy.optimize import brentq

from iwt import avo
from iwt import imaging as I
from iwt import raytrace as RT
from iwt.model import ElasticModel1D, phase_velocities
from iwt.modeling.convolution import ricker
from iwt.modeling.elastic import complex_velocity
from iwt.modeling.reflection import exact_pp
from iwt.modeling.reflectivity import Reflectivity

UP = (2400.0, 1000.0, 2200.0, 0.15, 0.05)
LO = (2900.0, 1500.0, 2350.0, 0.05, 0.02)
OFF = np.arange(100.0, 1101.0, 100.0)
DT, NT = 0.001, 1100


# -- kinematics -------------------------------------------------------------------------


def test_group_velocity_matches_tsvankin():
    a = RT.velocity_stiffness(2800.0, 1300.0, 0.2, 0.05)
    th = np.radians(np.linspace(1.0, 80.0, 12))
    vg, psi = RT.qp_group_velocity(th, *a)
    V = phase_velocities(th, 2800.0, 1300.0, 0.2, 0.05, 0.0)[0]
    h = 1e-6
    dV = (phase_velocities(th + h, 2800.0, 1300.0, 0.2, 0.05, 0.0)[0]
          - phase_velocities(th - h, 2800.0, 1300.0, 0.2, 0.05, 0.0)[0]) / (2 * h)
    np.testing.assert_allclose(vg, np.hypot(V, dV), rtol=1e-9)
    np.testing.assert_allclose(np.tan(psi), (np.tan(th) + dV / V) / (1 - np.tan(th) * dV / V),
                               rtol=1e-6)
    # second derivative of q against finite differences
    p = np.linspace(1e-5, 3e-4, 7)
    _, _, d2 = RT.qp_vertical_slowness(p, *a, derivative=2)
    d1 = lambda pp: RT.qp_vertical_slowness(pp, *a, derivative=1)[1]   # noqa: E731
    np.testing.assert_allclose(d2, (d1(p + 1e-10) - d1(p - 1e-10)) / 2e-10, rtol=1e-6)


def test_reflection_table_homogeneous_vti():
    m = ElasticModel1D([0.0, 2000.0], vp0=2800.0, vs0=1300.0, rho=2400.0, eps=0.2, delta=0.05,
                       qp=50.0)
    a = RT.velocity_stiffness(2800.0, 1300.0, 0.2, 0.05)
    x = np.array([0.0, 300.0, 800.0, 1500.0, 2500.0])
    tab = RT.reflection_table(m, [500.0, 900.0], x)
    for i, z in enumerate((500.0, 900.0)):
        for j, xx in enumerate(x):
            psi = np.arctan2(xx / 2, z)
            th = 0.0 if xx == 0 else brentq(lambda t: RT.qp_group_velocity(t, *a)[1] - psi, 0.0,
                                            np.radians(89.0))
            vg = RT.qp_group_velocity(th, *a)[0]
            T = 2 * np.hypot(z, xx / 2) / vg
            assert tab.time[i, j] == pytest.approx(T, rel=1e-10)
            assert tab.theta[i, j] == pytest.approx(th, abs=1e-8)
            assert tab.tstar[i, j] == pytest.approx(T / 50.0, rel=1e-10)
    # zero offset: spreading 2 z (1 + 2 delta) (NMO focusing); isotropic: the distance
    assert tab.spreading[0, 0] == pytest.approx(1000.0 * 1.1, rel=1e-6)
    iso = ElasticModel1D([0.0, 2000.0], vp0=2800.0, vs0=1300.0, rho=2400.0)
    ti = RT.reflection_table(iso, [500.0], x[:4])
    np.testing.assert_allclose(ti.spreading[0], np.hypot(1000.0, x[:4]), rtol=1e-8)


def test_reflection_table_fermat_in_layers():
    m = ElasticModel1D([0.0, 400.0, 550.0, 700.0, 1200.0],
                       vp0=[1480.0, 2300.0, 3100.0, 2700.0], vs0=[0.0, 1000.0, 1650.0, 1250.0],
                       rho=[1030.0, 2200.0, 2400.0, 2350.0], eps=[0.0, 0.25, 0.04, 0.12],
                       delta=[0.0, 0.10, 0.01, 0.05], qp=[np.inf, 40.0, 90.0, 60.0])
    z = np.array([500.0, 650.0, 800.0])
    x = np.array([0.0, 200.0, 600.0, 1000.0])
    tab = RT.reflection_table(m, z, x, 6.0, 8.0)
    assert np.all(np.isfinite(tab.time))
    h = 0.05
    dT = (RT.reflection_table(m, z, x + h, 6.0, 8.0).time
          - RT.reflection_table(m, z, np.abs(x - h), 6.0, 8.0).time) / (2 * h)
    np.testing.assert_allclose(dT[:, 1:], tab.p[:, 1:], rtol=1e-6)
    # zero offset: the vertical two-way time minus the source and receiver depths in the sea
    np.testing.assert_allclose(tab.time[:, 0], m.twt(z) - 14.0 / 1480.0, rtol=1e-12)
    np.testing.assert_allclose(tab.tstar[:, 0], [
        (2 * 100 / 2300) / 40, 2 * (150 / 2300 / 40 + 100 / 3100 / 90),
        2 * (150 / 2300 / 40 + 150 / 3100 / 90 + 100 / 2700 / 60)], rtol=1e-9)
    with pytest.raises(ValueError):
        RT.reflection_table(m, [5.0], x, 6.0, 8.0)


@pytest.mark.parametrize("eps, delta", [(0.2, 0.05), (0.25, 0.0), (0.1, 0.1)])
def test_nonhyperbolic_moveout(eps, delta):
    """Alkhalifah & Tsvankin (1995): within 1 % to offset/depth 2, far better than a hyperbola."""
    m = ElasticModel1D([0.0, 3000.0], vp0=2800.0, vs0=1300.0, rho=2400.0, eps=eps, delta=delta)
    z = 1000.0
    x = np.linspace(1.0, 2 * z, 41)
    T = RT.reflection_table(m, [z], x).time[0]
    t0, vn, eta = 2 * z / 2800.0, 2800.0 * np.sqrt(1 + 2 * delta), (eps - delta) / (1 + 2 * delta)
    at = np.sqrt(t0 ** 2 + x ** 2 / vn ** 2
                 - 2 * eta * x ** 4 / (vn ** 2 * (t0 ** 2 * vn ** 2 + (1 + 2 * eta) * x ** 2)))
    hyp = np.sqrt(t0 ** 2 + x ** 2 / vn ** 2)
    err_at, err_hyp = np.abs(at / T - 1).max(), np.abs(hyp / T - 1).max()
    assert err_at < 0.01
    if eta > 0:
        assert err_at < 0.2 * err_hyp
    else:                                  # elliptical: both exact
        assert err_at < 1e-12


# -- Q operators ------------------------------------------------------------------------


def test_attenuation_operator_is_kjartansson_to_first_order():
    c0, Q, z, fr = 2500.0, 40.0, 1000.0, 30.0
    f = np.linspace(5.0, 60.0, 50)
    om = 2 * np.pi * f
    k = om / complex_velocity(c0, Q, om, 2 * np.pi * fr)   # exp(i(kz - wt)) convention
    exact = np.conj(np.exp(1j * k * z) * np.exp(-1j * om * z / c0))
    H = I.attenuation_operator(f, z / c0 / Q, fr)
    np.testing.assert_allclose(np.abs(H), np.abs(exact), rtol=0.012)
    assert np.max(np.abs(np.angle(H / exact))) < 0.01


@pytest.mark.parametrize("G", [2.0, 10.0, 100.0])
def test_stabilization_sets_the_gain_limit(G):
    s = I.stabilization(G)
    beta = np.geomspace(1e-9, 1.0, 200001)
    assert ((beta + s) / (beta ** 2 + s)).max() == pytest.approx(G, rel=1e-4)
    with pytest.raises(ValueError):
        I.stabilization(0.5)


def test_compensation_recovers_attenuated_wavelet():
    t, w = ricker(30.0, DT)
    tr = np.zeros(1024)
    tr[400:400 + w.size] = w
    att = I.attenuate(tr, DT, 0.008)
    assert att.max() < 0.6
    rec = I.compensate(att, DT, 0.008, gain_limit=1000.0)
    assert np.max(np.abs(rec - tr)) < 0.01
    weak = I.compensate(att, DT, 0.008, gain_limit=2.0)
    assert att.max() < weak.max() < rec.max()
    phase = I.compensate(att, DT, 0.008, mode="phase")
    assert np.argmax(phase) == np.argmax(tr) and phase.max() < 0.6
    with pytest.raises(ValueError):
        I.compensate(att, DT, 0.008, mode="spectral")


# -- migration --------------------------------------------------------------------------


def _model(q=True):
    return ElasticModel1D([0.0, 400.0, 700.0, 1500.0], vp0=[1480.0, UP[0], LO[0]],
                          vs0=[0.0, UP[1], LO[1]], rho=[1030.0, UP[2], LO[2]],
                          eps=[0.0, UP[3], LO[3]], delta=[0.0, UP[4], LO[4]],
                          qp=[np.inf, 40.0, 80.0] if q else np.inf,
                          qs=[np.inf, 20.0, 40.0] if q else np.inf)


@pytest.fixture(scope="module")
def gathers():
    """Born gathers with and without Q; the direct wave and the seabed reflection muted
    (their energy would otherwise leak through the inverse Q filter)."""
    t_sb = RT.reflection_table(_model(), [399.99], OFF, 6.0, 8.0).time[0]
    w = np.clip((np.arange(NT) * DT - (t_sb[:, None] + 0.035)) / 0.01, 0.0, 1.0)
    mute = 0.5 - 0.5 * np.cos(np.pi * w)
    out = {}
    for att in (False, True):
        r = Reflectivity(_model(), attenuation=att, free_surface=False, internal_multiples=False,
                         converted_waves=False, transmission_loss=False)
        out[att] = r.record(OFF, 6.0, [8.0], DT, NT, ricker(30.0, DT), "pressure",
                            f_max=90.0).data[0] * mute
    return out


def _ava(data, model, mode=None, z0=700.0):
    z = np.arange(z0 - 40.0, z0 + 40.0, 0.5)
    img = I.migrate_cmp(data, DT, OFF, model, z, 6.0, 8.0, q_compensation=mode, gain_limit=30.0,
                        high_cut=75.0)
    pk = avo.pick_event(z, img.data, img.theta, z0, 6.0)
    # the incidence angle is that of the layer just above the (sharp) reflector
    return pk._replace(theta=img.theta[np.searchsorted(z, z0) - 1])


def test_true_amplitude_migration(gathers):
    """Migrated amplitude / exact plane-wave R is constant (point-source spreading removed)."""
    pk = _ava(gathers[False], _model())
    ok = np.degrees(pk.theta) < 32.0
    ratio = pk.amplitude[ok] / exact_pp(pk.theta[ok], UP, LO).real
    assert np.ptp(ratio) < 0.01 * np.median(ratio)
    np.testing.assert_allclose(pk.depth[ok], 700.0, atol=1.0)


def test_q_compensation_in_migration(gathers):
    ref = _ava(gathers[False], _model())
    ok = np.degrees(ref.theta) < 35.0
    comp = _ava(gathers[True], _model(), "amplitude")
    none = _ava(gathers[True], _model())
    phase = _ava(gathers[True], _model(), "phase")
    near = np.degrees(ref.theta) < 25.0
    np.testing.assert_allclose(comp.amplitude[near], ref.amplitude[near], rtol=0.01)
    np.testing.assert_allclose(comp.amplitude[ok], ref.amplitude[ok], rtol=0.03)
    r_none = none.amplitude[ok] / ref.amplitude[ok]
    assert np.all(r_none < 0.8) and r_none[-1] < r_none[0] - 0.05       # decays with angle
    r_phase = phase.amplitude[ok] / ref.amplitude[ok]
    np.testing.assert_allclose(r_phase, r_none, atol=0.05)
    assert np.abs(phase.depth[ok] - ref.depth[ok]).max() < np.abs(none.depth[ok] - ref.depth[ok]).max()


def test_isotropic_migration_leaves_residual_moveout(gathers):
    vti = _ava(gathers[False], _model())
    iso_m = _model(q=False)
    iso_m.eps[:] = 0.0
    iso_m.delta[:] = 0.0
    iso = _ava(gathers[False], iso_m)
    ok = np.degrees(vti.theta) < 40.0
    assert np.ptp(vti.depth[ok]) < 1.5
    assert np.ptp(iso.depth[ok]) > 4.0


# -- AVA --------------------------------------------------------------------------------


def test_intercept_gradient_and_event_tracking():
    th = np.radians(np.arange(5.0, 50.0, 2.5))
    amp = -0.05 - 0.12 * np.sin(th) ** 2
    fit = avo.intercept_gradient(th, amp, (20.0, 45.0))
    assert fit.intercept == pytest.approx(-0.05) and fit.gradient == pytest.approx(-0.12)
    assert fit.n == np.sum((th >= np.radians(20)) & (th <= np.radians(45))) and fit.rms < 1e-12
    w = avo.intercept_gradient(th, amp + 0.001 * np.cos(7 * th), sigma=0.001)
    assert np.sqrt(w.cov[1, 1]) > 0
    # a trough dipping 0.4 m per trace, under a stronger peak 15 m above it
    z = np.arange(600.0, 700.0, 0.5)
    n = th.size
    depth = 650.0 + 0.4 * np.arange(n)
    d = np.array([-np.abs(a) * np.exp(-((z - zz) / 3.0) ** 2)
                  + 2 * np.abs(a) * np.exp(-((z - zz + 15) / 3.0) ** 2)
                  for a, zz in zip(amp, depth)]).T
    pk = avo.pick_event(z, d, np.tile(th, (z.size, 1)), 651.0, 6.0)
    np.testing.assert_allclose(pk.depth, depth, atol=0.05)
    np.testing.assert_allclose(pk.amplitude, -np.abs(amp), rtol=0.01)
