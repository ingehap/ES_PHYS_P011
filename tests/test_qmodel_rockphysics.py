import numpy as np
import pytest

from iwt.qmodel import attenuation_coefficient, dispersion_exponent, phase_velocity, t_star
from iwt.rockphysics import (
    fluid_substitution,
    gassmann_dry,
    gassmann_saturated,
    reuss_fluid,
)


# -- constant Q ----------------------------------------------------------------


def test_phase_velocity_reference_and_power_law():
    assert phase_velocity(2500.0, 50.0, 30.0, 30.0) == pytest.approx(2500.0)
    g = dispersion_exponent(50.0)
    assert g == pytest.approx(np.arctan(1 / 50.0) / np.pi)
    ratio = phase_velocity(1.0, 50.0, 12e3, 30.0) / phase_velocity(1.0, 50.0, 2e3, 30.0)
    assert ratio == pytest.approx((12e3 / 2e3) ** g)


def test_high_q_dispersion_approximation():
    # c(f)/c0 ~ 1 + ln(f/f0) / (pi Q) for Q >> 1 (Kjartansson, 1979)
    for q in (30.0, 100.0, 300.0):
        c = phase_velocity(1.0, q, 1e4, 30.0)
        approx = 1.0 + np.log(1e4 / 30.0) / (np.pi * q)
        assert c == pytest.approx(approx, rel=2.0 / q**2 * np.log(1e4 / 30) ** 2)


def test_infinite_q_is_non_dispersive():
    assert phase_velocity(2000.0, np.inf, 1e4, 30.0) == pytest.approx(2000.0)
    assert attenuation_coefficient(2000.0, np.inf, 50.0, 30.0) == 0.0
    with pytest.raises(ValueError):
        phase_velocity(2000.0, 50.0, 0.0, 30.0)


def test_attenuation_coefficient_high_q_limit():
    q, f, c0 = 200.0, 40.0, 3000.0
    alpha = attenuation_coefficient(c0, q, f, f)
    assert alpha == pytest.approx(np.pi * f / (q * c0), rel=1e-4)


def test_t_star():
    assert t_star([100.0, 200.0], [2000.0, 2500.0], [50.0, np.inf]) == pytest.approx(0.001)


# -- Gassmann ------------------------------------------------------------------

K_MIN = 36.6e9


def test_gassmann_round_trip():
    k_dry = np.array([2e9, 7e9, 15e9])
    k_sat = gassmann_saturated(k_dry, K_MIN, 2.4e9, 0.25)
    np.testing.assert_allclose(gassmann_dry(k_sat, K_MIN, 2.4e9, 0.25), k_dry, rtol=1e-12)
    assert np.all(k_sat > k_dry)


def test_gassmann_limits():
    # a fluid as stiff as the mineral makes the rock as stiff as the mineral
    assert gassmann_saturated(5e9, K_MIN, K_MIN, 0.3) == pytest.approx(K_MIN)
    # a vanishingly soft fluid leaves the dry frame unchanged
    assert gassmann_saturated(5e9, K_MIN, 1e-3, 0.3) == pytest.approx(5e9, rel=1e-6)


def test_fluid_substitution_shear_and_round_trip():
    vp, vs, rho = 3050.0, 1650.0, 2210.0
    k_oil, rho_oil = 1.2e9, 840.0
    vp2, vs2, rho2 = fluid_substitution(vp, vs, rho, 0.27, K_MIN, 2.35e9, 1030.0, k_oil, rho_oil)
    assert vp2 < vp
    assert rho2 == pytest.approx(rho + 0.27 * (rho_oil - 1030.0))
    # shear modulus is unchanged: Vs changes only through density
    assert rho2 * vs2**2 == pytest.approx(rho * vs**2)
    back = fluid_substitution(vp2, vs2, rho2, 0.27, K_MIN, k_oil, rho_oil, 2.35e9, 1030.0)
    np.testing.assert_allclose(back, (vp, vs, rho), rtol=1e-12)


def test_fluid_substitution_rejects_impossible_frame():
    with pytest.raises(ValueError):
        fluid_substitution(1600.0, 200.0, 2000.0, 0.3, K_MIN, 2.35e9, 1030.0, 1e9, 800.0)


def test_reuss_fluid():
    k, r = reuss_fluid([2.35e9, 1.2e9], [1030.0, 840.0], [0.15, 0.85])
    assert k == pytest.approx(1.0 / (0.15 / 2.35e9 + 0.85 / 1.2e9))
    assert r == pytest.approx(0.15 * 1030 + 0.85 * 840)
    with pytest.raises(ValueError):
        reuss_fluid([1e9, 2e9], [800.0, 1000.0], [0.5, 0.6])
