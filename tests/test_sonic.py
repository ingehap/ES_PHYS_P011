"""Tests of the sonic anisotropy (iwt.sonic) and the synthetic Stoneley and caliper logs."""

import numpy as np
import pytest

from iwt import sonic as S
from iwt.model import thomsen_to_cij
from iwt.qmodel import phase_velocity

MUD = S.Mud(1540.0, 1250.0)
B = 0.108
A = 0.046
SHALE = (2400.0, 950.0, 2280.0, 0.20, 0.08, 0.24)       # leaky: slow vertical shear
SAND = (3350.0, 1800.0, 2420.0, 0.05, 0.02, 0.06)       # fast


def _cij(vp0, vs0, rho, eps, delta, gamma):
    c = thomsen_to_cij(vp0, vs0, rho, eps, delta, gamma)
    return (c.c11, c.c13, c.c33, c.c44, c.c66)


def _iso(vp, vs, rho):
    return S.ti_stiffness(vp, vs, rho, rho * vs ** 2)


# -- tube wave ------------------------------------------------------------------------


@pytest.mark.parametrize("a, kt", [(0.0, np.inf), (A, np.inf), (A, 8e9)])
def test_tube_wave_round_trip(a, kt):
    c66 = np.array([2.0e9, 5.0e9, 9.0e9])
    s = S.tube_wave_slowness(c66, MUD.slowness, MUD.density, B, a, kt)
    np.testing.assert_allclose(S.c66_from_tube_wave(s, MUD.slowness, MUD.density, B, a, kt),
                               c66, rtol=1e-12)
    if a == 0.0:   # White's formula
        np.testing.assert_allclose(s ** 2, MUD.slowness ** 2 + MUD.density / c66, rtol=1e-12)
    else:          # the tool always slows the tube wave
        assert np.all(s > S.tube_wave_slowness(c66, MUD.slowness, MUD.density, B))


def test_gamma_is_zero_in_an_isotropic_formation():
    vp, vs, rho = 2800.0, 1400.0, 2350.0
    s0 = S.tube_wave_slowness(rho * vs ** 2, MUD.slowness, MUD.density, B, A)
    r = S.sonic_anisotropy(vp, vs, rho, s0, MUD, S.Borehole(B, A))
    assert abs(float(r.gamma)) < 1e-12
    s1 = S.stoneley_slowness(500.0, _iso(vp, vs, rho), rho, MUD.slowness, MUD.density, B, A)
    r = S.sonic_anisotropy(vp, vs, rho, s1, MUD, S.Borehole(B, A), frequency=500.0)
    assert abs(float(r.gamma)) < 1e-8
    assert float(r.dispersion) != 1.0


# -- the exact borehole mode --------------------------------------------------------


@pytest.mark.parametrize("a, kt", [(0.0, np.inf), (A, np.inf), (A, 5e9)])
def test_stoneley_low_frequency_limit_is_the_tube_wave(a, kt):
    for p in (SHALE, SAND):
        cij = _cij(*p)
        s = S.stoneley_slowness(0.5, cij, p[2], MUD.slowness, MUD.density, B, a, kt)
        st = S.tube_wave_slowness(cij[4], MUD.slowness, MUD.density, B, a, kt)
        assert s == pytest.approx(st, rel=1e-6)


def test_stoneley_high_frequency_limit_is_the_scholte_wave():
    from scipy.optimize import brentq

    vp, vs, rho = 3000.0, 1600.0, 2400.0
    sp2, ss2, sf2 = 1 / vp ** 2, 1 / vs ** 2, MUD.slowness ** 2

    def scholte(x):
        a, b, f = np.sqrt(x ** 2 - sp2), np.sqrt(x ** 2 - ss2), np.sqrt(x ** 2 - sf2)
        return (2 * x ** 2 - ss2) ** 2 - 4 * x ** 2 * a * b + MUD.density / rho * ss2 ** 2 * a / f

    lo = max(np.sqrt(ss2), np.sqrt(sf2)) * (1 + 1e-9)
    grid = np.linspace(lo, 2 * lo, 4000)
    d = scholte(grid)
    j = np.flatnonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    s_sch = brentq(scholte, grid[j], grid[j + 1])
    s = S.stoneley_slowness(1e6, _iso(vp, vs, rho), rho, MUD.slowness, MUD.density, B)
    assert s == pytest.approx(s_sch, rel=1e-4)


def test_leaky_stoneley_in_slow_anisotropic_shale():
    """The tube wave (C66) outruns the axial shear wave (C44): the mode radiates."""
    f = np.array([200.0, 500.0, 1000.0])
    s = S.stoneley_slowness(f, _cij(*SHALE), SHALE[2], MUD.slowness, MUD.density, B, A, full=True)
    assert np.all(s.imag > 0) and np.all(np.diff(s.imag) > 0)
    s_tube = S.tube_wave_slowness(_cij(*SHALE)[4], MUD.slowness, MUD.density, B, A)
    assert s_tube < 1.0 / SHALE[1]
    sand = S.stoneley_slowness(f, _cij(*SAND), SAND[2], MUD.slowness, MUD.density, B, A, full=True)
    np.testing.assert_allclose(sand.imag, 0.0, atol=1e-12)
    # the measured C44 matters more than the unknown eps and delta at low frequency
    vp0, vs0, rho, eps, delta, gamma = SHALE
    c66 = _cij(*SHALE)[4]
    ti0 = S.stoneley_slowness(500.0, S.ti_stiffness(vp0, vs0, rho, c66), rho, MUD.slowness,
                              MUD.density, B, A)
    iso = S.stoneley_slowness(500.0, S.ti_stiffness(vp0, np.sqrt(c66 / rho), rho, c66), rho,
                              MUD.slowness, MUD.density, B, A)
    err_ti0, err_iso = abs(ti0 / s[1].real - 1), abs(iso / s[1].real - 1)
    assert err_ti0 < 4e-3 and err_iso > 8e-3 and err_iso > 3 * err_ti0


@pytest.mark.parametrize("p", [SHALE, SAND])
def test_dispersion_corrected_inversion_recovers_c66(p):
    cij = _cij(*p)
    vp0, vs0, rho, eps, delta, gamma = p
    s = S.stoneley_slowness(500.0, cij, rho, MUD.slowness, MUD.density, B, A)
    bh = S.Borehole(B, A)
    r = S.sonic_anisotropy(vp0, vs0, rho, s, MUD, bh, frequency=500.0, eps=eps, delta=delta)
    assert float(r.c66) == pytest.approx(cij[4], rel=1e-7)
    assert float(r.gamma) == pytest.approx(gamma, abs=1e-7)
    naive = S.sonic_anisotropy(vp0, vs0, rho, s, MUD, bh)
    if p is SHALE:
        assert float(naive.gamma) < gamma - 0.05     # dispersion ignored: gamma far too low


def test_inversion_of_a_log_at_once():
    rng = np.random.default_rng(3)
    n = 40
    vp = 2800 + 400 * rng.random(n)
    vs = 1300 + 400 * rng.random(n)
    rho = 2300 + 100 * rng.random(n)
    gam = 0.15 * rng.random(n)
    c66 = rho * vs ** 2 * (1 + 2 * gam)
    s = S.stoneley_slowness(500.0, S.ti_stiffness(vp, vs, rho, c66), rho, MUD.slowness,
                            MUD.density, B, A)
    assert s.shape == (n,) and np.all(np.isfinite(s))
    r = S.sonic_anisotropy(vp, vs, rho, s, MUD, S.Borehole(B, A), frequency=500.0)
    np.testing.assert_allclose(r.gamma, gam, atol=1e-7)


def test_shear_frequency_correction():
    vp0, vs0, rho, eps, delta, gamma = SAND
    qs = 30.0
    vs_dipole = phase_velocity(vs0, qs, 2000.0, 500.0)     # dipole sees a stiffer C44
    s = S.stoneley_slowness(500.0, _cij(*SAND), rho, MUD.slowness, MUD.density, B, A)
    bh = S.Borehole(B, A)
    biased = S.sonic_anisotropy(vp0, vs_dipole, rho, s, MUD, bh, 500.0, eps=eps, delta=delta)
    fixed = S.sonic_anisotropy(vp0, vs_dipole, rho, s, MUD, bh, 500.0, eps=eps, delta=delta,
                               shear_frequency=2000.0, qs=qs)
    assert float(biased.gamma) < gamma - 0.01
    assert float(fixed.gamma) == pytest.approx(gamma, abs=1e-6)


# -- uncertainty and calibration ------------------------------------------------------


def test_gamma_jacobian_against_finite_differences():
    x0 = dict(s_stoneley=9.3e-4, s_shear=1 / 1100.0, rho=2300.0, mud_slowness=1 / 1500.0,
              mud_density=1200.0, radius=0.108, tool_radius=0.046, tool_compliance=1 / 20e9)

    def gamma(**kw):
        c66 = S.c66_from_tube_wave(kw["s_stoneley"], kw["mud_slowness"], kw["mud_density"],
                                   kw["radius"], kw["tool_radius"], 1 / kw["tool_compliance"])
        return S.gamma_from_moduli(c66, kw["rho"] / kw["s_shear"] ** 2)

    J = S.gamma_jacobian(**x0)
    for i, k in enumerate(S.GAMMA_INPUTS):
        h = abs(x0[k]) * 1e-6
        xp, xm = dict(x0), dict(x0)
        xp[k] += h
        xm[k] -= h
        assert J[i] == pytest.approx((gamma(**xp) - gamma(**xm)) / (2 * h), rel=1e-6), k


def test_mud_slowness_dominates_the_gamma_error_in_fast_rock():
    s_sand = S.tube_wave_slowness(_cij(*SAND)[4], MUD.slowness, MUD.density, B, A)
    J = S.gamma_jacobian(s_sand, 1 / SAND[1], SAND[2], MUD.slowness, MUD.density, B, A, 0.0)
    sig = {"s_stoneley": 0.003 * s_sand, "s_shear": 0.01 / SAND[1], "rho": 15.0,
           "mud_slowness": 0.02 * MUD.slowness, "mud_density": 0.01 * MUD.density,
           "radius": 0.001, "tool_radius": 0.001}
    total, contrib = S.gamma_uncertainty(J, sig)
    assert np.isclose(total, np.sqrt(sum(c ** 2 for c in contrib.values())))
    assert max(contrib, key=contrib.get) == "mud_slowness"
    assert contrib["mud_slowness"] > 2 * sorted(contrib.values())[-2]


def test_calibrate_mud_slowness_with_a_known_reference():
    rng = np.random.default_rng(4)
    vp = 3000 + 200 * rng.random(20)
    vs = 1600 + 100 * rng.random(20)
    rho = np.full(20, 2300.0)
    gam = 0.03
    s = S.tube_wave_slowness(rho * vs ** 2 * (1 + 2 * gam), MUD.slowness, MUD.density, B, A)
    sf = S.calibrate_mud_slowness(vp, vs, rho, s, MUD.density, S.Borehole(B, A), gamma_ref=gam)
    assert sf == pytest.approx(MUD.slowness, rel=1e-9)
    # assuming the reference is isotropic when it is not makes the mud too fast
    sf0 = S.calibrate_mud_slowness(vp, vs, rho, s, MUD.density, S.Borehole(B, A), gamma_ref=0.0)
    assert sf0 < MUD.slowness * 0.995


def test_running_average_harmonic_matches_backus_c44():
    z = np.arange(0.0, 99.0, 0.5)
    c44 = np.where(np.arange(z.size) % 3 == 0, 2e9, 6e9)
    h = S.running_average(z, c44, 16.5, harmonic=True)        # 33 samples = 11 periods
    assert np.allclose(h[50:150], 1.0 / np.mean(1.0 / c44), rtol=1e-12)
    assert np.allclose(S.running_average(z, c44, 0.0), c44)


# -- synthetic logs -----------------------------------------------------------------


def test_synthetic_caliper_and_stoneley_logs():
    from iwt.synthetic import caliper_log, stoneley_log, wisting_like_model

    model = wisting_like_model().subset(0.0, 800.0)
    z = np.arange(470.0, 780.0, 2.0)
    rad, cal = caliper_log(model, z)
    labels = model.at(z)["label"]
    sand = np.array([str(lab).startswith(("Stø", "Nordmela", "Fruholmen")) for lab in labels])
    assert np.all(rad[sand] == 0.108) and np.all(rad[~sand] >= 0.108)
    assert np.max(rad) > 0.112
    st = stoneley_log(model, z, MUD, rad, seed=None)
    assert np.isfinite(st.slowness).mean() > 0.95
    shale = np.array([str(lab).startswith("Lower Cretaceous") for lab in labels])
    assert np.nanmedian(st.inv_q[shale]) > 1e-3            # leaky in the slow shale
    assert np.nanmax(np.abs(st.inv_q[sand])) < 1e-9
    st2 = stoneley_log(model, z, MUD, rad, seed=11)
    ok = np.isfinite(st.slowness)
    assert np.array_equal(ok, np.isfinite(st2.slowness))
    assert 0.002 < np.std(st2.slowness[ok] / st.slowness[ok] - 1) < 0.004
