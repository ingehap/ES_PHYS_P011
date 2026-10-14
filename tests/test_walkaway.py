"""Tests of the qP ray tracing (iwt.raytrace) and the walkaway VSP inversion (iwt.vsp.walkaway)."""

import dataclasses

import numpy as np
import pytest

from iwt import raytrace as RT
from iwt.model import ElasticModel1D, phase_velocities
from iwt.vsp import walkaway as W

TRUTH = np.array([2800.0, 1300.0, 0.12, 0.05])
OFFSETS = np.concatenate([-np.arange(2500.0, 199.0, -25.0), np.arange(200.0, 2501.0, 25.0)])
DEPTHS = np.arange(600.0, 901.0, 10.0)
SIGMA_T = 0.2e-3
SIGMA_PSI = np.radians(1.5)


def _model(vp0=2800.0, vs0=1300.0, eps=0.12, delta=0.05):
    """400 m of water over a homogeneous VTI half-space."""
    return ElasticModel1D([0.0, 400.0, 2000.0], vp0=[1480.0, vp0], vs0=[0.0, vs0],
                          rho=[1030.0, 2400.0], eps=[0.0, eps], delta=[0.0, delta],
                          gamma=[0.0, 0.1])


@pytest.fixture(scope="module")
def clean():
    return W.synthetic_walkaway(_model(), OFFSETS, DEPTHS)


def _noisy(survey, rng, scale=1.0):
    return dataclasses.replace(
        survey, time=survey.time + scale * SIGMA_T * rng.standard_normal(survey.time.shape),
        psi=survey.psi + scale * SIGMA_PSI * rng.standard_normal(survey.psi.shape))


def _measure(survey, level=750.0, **kw):
    return W.measure_slowness(survey, level, time_noise=SIGMA_T, psi_noise=SIGMA_PSI, **kw)


def _angles(data, lo, hi):
    th = np.degrees(data.theta)
    return data.select((th >= lo) & (th <= hi))


def _random_prior(rng, truth=TRUTH):
    """Sonic-like priors (Vp0 1 %, Vs0 5 %) whose means scatter as their errors say."""
    s = (0.01 * truth[0], 0.05 * truth[1])
    return {"vp0": (truth[0] + s[0] * rng.standard_normal(), s[0]),
            "vs0": (truth[1] + s[1] * rng.standard_normal(), s[1])}


# -- exact qP kinematics -------------------------------------------------------------


def test_phase_velocity_slowness_and_derivative():
    a = RT.velocity_stiffness(*TRUTH)
    th = np.radians(np.linspace(0.0, 89.0, 60))
    v = RT.qp_phase_velocity(th, *a)
    np.testing.assert_allclose(v, phase_velocities(th, *TRUTH, 0.1)[0], rtol=1e-12)
    p, q = np.sin(th) / v, np.cos(th) / v
    qq, dq = RT.qp_vertical_slowness(p, *a, derivative=True)
    np.testing.assert_allclose(qq, q, rtol=1e-10)
    h = 1e-10
    np.testing.assert_allclose(dq, (RT.qp_vertical_slowness(p + h, *a)
                                    - RT.qp_vertical_slowness(p - h, *a)) / (2 * h),
                               rtol=1e-5, atol=1e-8)
    # beyond the horizontal slowness the qP wave is evanescent
    assert np.isnan(RT.qp_vertical_slowness(1.01 / np.sqrt(a[0]), *a))
    # fluid: q = sqrt(1/V^2 - p^2)
    np.testing.assert_allclose(RT.qp_vertical_slowness(1e-4, 1480.0 ** 2, 1480.0 ** 2,
                                                       1480.0 ** 2, 0.0),
                               np.sqrt(1480.0 ** -2 - 1e-8), rtol=1e-12)


def test_polarization_is_christoffel_eigenvector():
    a11, a13, a33, a44 = RT.velocity_stiffness(*TRUTH)
    th = np.radians(np.linspace(1.0, 89.0, 30))
    v = RT.qp_phase_velocity(th, a11, a13, a33, a44)
    p, q = np.sin(th) / v, np.cos(th) / v
    psi = RT.qp_polarization(p, q, a11, a13, a33, a44)
    u = np.stack([np.sin(psi), np.cos(psi)])
    G = np.array([[a11 * p * p + a44 * q * q, (a13 + a44) * p * q],
                  [(a13 + a44) * p * q, a44 * p * p + a33 * q * q]])
    np.testing.assert_allclose(np.einsum("ijn,jn->in", G, u), u, atol=1e-10)
    assert np.all(psi > th)                     # eps > delta: motion leans towards horizontal
    iso = RT.velocity_stiffness(2800.0, 1300.0, 0.0, 0.0)
    viso = RT.qp_phase_velocity(th, *iso)
    np.testing.assert_allclose(RT.qp_polarization(np.sin(th) / viso, np.cos(th) / viso, *iso),
                               th, atol=1e-12)


def test_two_point_satisfies_fermat():
    """dT/dx = p and dT/dz = q (receiver side) in a layered VTI model with a water layer."""
    m = ElasticModel1D([0.0, 300.0, 520.0, 700.0, 1200.0],
                       vp0=[1480.0, 2300.0, 3100.0, 2700.0], vs0=[0.0, 1000.0, 1650.0, 1250.0],
                       rho=[1030.0, 2200.0, 2400.0, 2350.0], eps=[0.0, 0.25, 0.04, 0.12],
                       delta=[0.0, 0.10, 0.01, 0.05])
    x = np.array([150.0, 600.0, 1000.0, 1400.0])
    z = np.array([600.0, 650.0, 900.0])
    r = RT.two_point(m, 6.0, z, x)
    assert np.all(np.isfinite(r.time))
    # far beyond the critical offset of the fast layer: no direct ray below 85 degrees
    assert np.isnan(RT.two_point(m, 6.0, [600.0], [3000.0]).time[0, 0])
    assert np.isfinite(RT.two_point(m, 6.0, [600.0], [3000.0], max_angle=89.9).time[0, 0])
    h = 0.05
    dtdx = (RT.two_point(m, 6.0, z, x + h).time - RT.two_point(m, 6.0, z, x - h).time) / (2 * h)
    dtdz = (RT.two_point(m, 6.0, z + h, x).time - RT.two_point(m, 6.0, z - h, x).time) / (2 * h)
    np.testing.assert_allclose(dtdx, r.p, rtol=1e-6)
    np.testing.assert_allclose(dtdz, r.q, rtol=1e-6)
    # homogeneous isotropic check: straight rays
    iso = ElasticModel1D([0.0, 2000.0], vp0=2500.0, vs0=1200.0, rho=2300.0)
    ri = RT.two_point(iso, 0.0, [800.0], x)
    np.testing.assert_allclose(ri.time[0], np.hypot(x, 800.0) / 2500.0, rtol=1e-12)
    np.testing.assert_allclose(ri.psi[0], np.arctan2(x, 800.0), atol=1e-10)


def test_hodogram_polarization():
    rng = np.random.default_rng(0)
    t = np.linspace(-1.0, 1.0, 81)
    w = (1 - 2 * (3 * t) ** 2) * np.exp(-(3 * t) ** 2)
    for psi in np.radians([-40.0, 0.0, 25.0, 60.0]):
        ux = np.sin(psi) * w + 0.01 * rng.standard_normal(t.size)
        uz = np.cos(psi) * w + 0.01 * rng.standard_normal(t.size)
        assert abs(W.hodogram_polarization(ux, uz) - psi) < np.radians(1.0)
        # the axis, not the polarity: reversed polarity gives the same angle
        assert abs(W.hodogram_polarization(-ux, -uz) - psi) < np.radians(1.0)


# -- measurements ---------------------------------------------------------------------


def test_measurement_covariance_matches_monte_carlo(clean):
    rng = np.random.default_rng(1)
    d0 = _measure(clean)
    X = np.array([np.concatenate(_measure(_noisy(clean, rng))[:3]) for _ in range(300)])
    ratio = X.var(axis=0) / np.diag(d0.cov)
    n = d0.p.size
    for block in range(3):
        assert abs(np.median(ratio[block * n:(block + 1) * n]) - 1.0) < 0.1
    # overlapping source windows: neighbouring p estimates are correlated
    j = n // 3
    rho_mc = np.corrcoef(X[:, j], X[:, j + 1])[0, 1]
    rho_model = d0.cov[j, j + 1] / np.sqrt(d0.cov[j, j] * d0.cov[j + 1, j + 1])
    assert rho_model > 0.3 and abs(rho_mc - rho_model) < 0.12
    # without given noise levels they are estimated from the data
    est = W.measure_slowness(_noisy(clean, rng), 750.0)
    assert abs(np.median(est.sigma_q / d0.sigma_q) - 1.0) < 0.15
    assert abs(np.median(est.sigma_psi / d0.sigma_psi) - 1.0) < 0.15


def test_missing_pick_voids_only_its_windows(clean):
    full = _measure(clean)
    t = clean.time.copy()
    t[0, 100, 15] = np.nan                       # one pick at the middle receiver of 750 m
    d = _measure(dataclasses.replace(clean, time=t))
    assert full.p.size - d.p.size == 5           # the source itself and its four neighbours
    assert np.all(np.isfinite(d.cov))


# -- inversion ------------------------------------------------------------------------


def test_noise_free_inversion_recovers_eps_delta(clean):
    d = _measure(clean)
    post = W.invert_level(d)
    assert abs(post["eps"][0] - 0.12) < 0.005 and abs(post["delta"][0] - 0.05) < 0.005
    assert abs(post["vp0"][0] - 2800.0) < 5.0
    assert post.chi2 / post.n_data < 0.05
    # article geometry: 30-60 degrees only, with sonic priors
    post = W.invert_level(_angles(d, 30, 60), {"vp0": (2800.0, 28.0), "vs0": (1300.0, 65.0)})
    assert abs(post["eps"][0] - 0.12) < 0.005 and abs(post["delta"][0] - 0.05) < 0.005


def test_posterior_coverage_slowness_only(clean):
    """The Laplace errors match the scatter of the estimates (priors drawn around the truth)."""
    rng = np.random.default_rng(3)
    z = []
    for _ in range(60):
        d = _angles(_measure(_noisy(clean, rng)), 30, 60)
        post = W.invert_level(d, _random_prior(rng), use_polarization=False)
        z.append((post.mean - TRUTH) / post.std)
    z = np.array(z)
    assert np.all((z.std(axis=0) > 0.75) & (z.std(axis=0) < 1.3))
    assert np.all(np.abs(z.mean(axis=0)) < 0.4)
    assert np.mean(np.abs(z[:, 2:]) < 2.0) > 0.88


def test_polarization_fit_bias_is_second_order(clean):
    """With polarizations the estimator has an errors-in-variables bias that scales with the
    noise variance: about half a standard deviation at the nominal noise, a quarter at half."""
    def mean_z(scale, n=40):
        rng = np.random.default_rng(5)
        ref = W.invert_level(_measure(clean)).mean          # noise-free estimate
        z = []
        for _ in range(n):
            d = W.measure_slowness(_noisy(clean, rng, scale), 750.0,
                                   time_noise=scale * SIGMA_T, psi_noise=scale * SIGMA_PSI)
            post = W.invert_level(d)
            z.append((post.mean - ref) / post.std)
        return np.array(z).mean(axis=0), np.array(z).std(axis=0)

    m1, s1 = mean_z(1.0)
    m2, s2 = mean_z(0.5)
    assert np.all(np.abs(m1) < 0.8) and np.all((s1 > 0.6) & (s1 < 1.3))
    assert abs(m2[2]) < 0.75 * abs(m1[2])          # eps bias shrinks with the noise


def test_near_angle_cutoff_increases_tradeoff(clean):
    d = _measure(_noisy(clean, np.random.default_rng(2)))
    prior = {"vp0": (2800.0, 28.0), "vs0": (1300.0, 65.0)}
    posts = [W.invert_level(_angles(d, lo, 90), prior) for lo in (0, 30, 40)]
    sd = [p["delta"][1] for p in posts]
    corr = [abs(p.corr[2, 3]) for p in posts]
    assert sd[0] < sd[1] < sd[2] and corr[0] < corr[1] < corr[2]


def test_weak_anisotropy_bias_for_strong_anisotropy():
    m = _model(2300.0, 1050.0, 0.26, 0.10)
    d = _measure(W.synthetic_walkaway(m, OFFSETS, DEPTHS))
    prior = {"vp0": (2300.0, 23.0), "vs0": (1050.0, 52.5)}
    exact = W.invert_level(d, prior, use_polarization=False)
    weak = W.invert_level(d, prior, method="weak")
    assert abs(exact["delta"][0] - 0.10) < exact["delta"][1]
    assert (weak["delta"][0] - 0.10) / weak["delta"][1] > 2.0
    assert not np.isfinite(W.invert_level(d, method="weak")["vs0"][1])   # Vs0 not in the data
    with pytest.raises(ValueError):
        W.invert_level(d, method="thomsen")


def test_vs0_prior_matters_only_with_polarization(clean):
    """P-wave slowness-polarization data are governed by Vp0 and two combinations of eps
    and delta that absorb Vs0 (Grechka & Mateeva, 2007): with polarizations an error in the
    Vs0 prior maps into eps and delta.  The qP phase velocity alone hardly depends on Vs0."""
    d = _angles(_measure(_noisy(clean, np.random.default_rng(4))), 30, 60)
    good = {"vp0": (2800.0, 28.0), "vs0": (1300.0, 65.0)}
    bad = {"vp0": (2800.0, 28.0), "vs0": (1430.0, 65.0)}           # 10 % too fast
    slow_good, slow_bad = (W.invert_level(d, p, use_polarization=False) for p in (good, bad))
    pol_good, pol_bad = (W.invert_level(d, p) for p in (good, bad))
    assert slow_good.data_weight[1] < 0.05 and pol_good.data_weight[1] < 0.3
    shift_slow = np.abs(slow_bad.mean[2:] - slow_good.mean[2:])
    shift_pol = np.abs(pol_bad.mean[2:] - pol_good.mean[2:])
    assert np.all(shift_slow < 0.1 * slow_good.std[2:])
    assert np.all(shift_pol > 0.003) and shift_pol[0] > 0.5 * pol_good["eps"][1]
    # ... while the polarization halves the eps and delta errors
    assert np.all(pol_good.std[2:] < 0.6 * slow_good.std[2:])


def test_azimuthal_consistency():
    models = [_model(), _model(eps=0.17)]
    kw = dict(azimuths=(0.0, 90.0), time_noise=SIGMA_T, psi_noise=SIGMA_PSI, seed=3)
    prior = {"vp0": (2800.0, 28.0), "vs0": (1300.0, 65.0)}
    for azm, vti in ((None, True), (models, False)):
        s = W.synthetic_walkaway(_model(), OFFSETS, DEPTHS, azimuth_models=azm, **kw)
        posts = [W.invert_level(_measure(s, azimuth=i), prior) for i in range(2)]
        chi2, dof, pval = W.azimuthal_consistency(posts)
        assert dof == 2
        assert (pval > 0.01) if vti else (pval < 1e-6)


def test_polarization_check_flags_source_statics():
    prior = {"vp0": (2800.0, 28.0), "vs0": (1300.0, 65.0)}
    pvals = []
    for k in (0.0, 1e-5):                  # source delay growing 10 ms per km of offset
        s = W.synthetic_walkaway(_model(), OFFSETS, DEPTHS, statics=lambda x: k * x,
                                 time_noise=SIGMA_T, psi_noise=SIGMA_PSI, seed=8)
        d = _measure(s)
        pvals.append(W.polarization_check(d, W.invert_level(d, prior, use_polarization=False))[1])
    assert pvals[0] > 0.01 and pvals[1] < 1e-6
