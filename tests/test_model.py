import numpy as np
import pytest

from iwt.model import (
    ElasticModel1D,
    Stiffness,
    backus_average,
    cij_to_thomsen,
    delta_min,
    is_stable,
    phase_velocities,
    thomsen_to_cij,
)

RNG = np.random.default_rng(1)


def random_vti(n, rng=RNG, anisotropic=True):
    vp0 = rng.uniform(1600.0, 5500.0, n)
    vs0 = vp0 / rng.uniform(1.5, 3.5, n)
    rho = rng.uniform(1800.0, 2800.0, n)
    if not anisotropic:
        z = np.zeros(n)
        return vp0, vs0, rho, z, z.copy(), z.copy()
    eps = rng.uniform(0.0, 0.35, n)
    delta = rng.uniform(-0.15, 0.25, n)
    gamma = rng.uniform(0.0, 0.4, n)
    return vp0, vs0, rho, eps, delta, gamma


# -- Thomsen <-> Cij ---------------------------------------------------------


def test_round_trip_thomsen_cij():
    vp0, vs0, rho, eps, delta, gamma = random_vti(500)
    c = thomsen_to_cij(vp0, vs0, rho, eps, delta, gamma)
    out = cij_to_thomsen(*c, rho)
    for a, b in zip(out, (vp0, vs0, eps, delta, gamma)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_isotropic_stiffness():
    c = thomsen_to_cij(3000.0, 1500.0, 2300.0, 0.0, 0.0, 0.0)
    lam_2mu = 2300.0 * 3000.0**2
    mu = 2300.0 * 1500.0**2
    np.testing.assert_allclose([c.c11, c.c33, c.c44, c.c66, c.c13],
                               [lam_2mu, lam_2mu, mu, mu, lam_2mu - 2 * mu])


def test_delta_lower_bound():
    vp0, vs0 = 3000.0, 1200.0
    dmin = delta_min(vp0, vs0)
    c = thomsen_to_cij(vp0, vs0, 2300.0, 0.1, dmin, 0.1)
    np.testing.assert_allclose(c.c13 + c.c44, 0.0, atol=1e-3 * c.c33)
    with pytest.raises(ValueError):
        thomsen_to_cij(vp0, vs0, 2300.0, 0.1, dmin - 0.01, 0.1)


def born_criteria(c):
    """Born stability criteria for transverse isotropy, written out by hand."""
    c12 = c.c11 - 2.0 * c.c66
    return ((c.c11 > np.abs(c12)) & (c.c44 > 0.0)
            & ((c.c11 + c12) * c.c33 > 2.0 * c.c13**2))


def test_stability_matches_born_criteria():
    # wide random draws include inadmissible combinations (e.g. large delta, small eps)
    c = thomsen_to_cij(*random_vti(2000))
    stable = is_stable(*c)
    np.testing.assert_array_equal(stable, born_criteria(c))
    assert 0 < stable.sum() < stable.size
    # typical sedimentary values (Vp/Vs >= 1.6, 0 <= delta <= eps) are admissible
    rng = np.random.default_rng(2)
    vp0 = rng.uniform(1800.0, 5000.0, 500)
    vs0 = vp0 / rng.uniform(1.6, 3.0, 500)
    eps = rng.uniform(0.0, 0.3, 500)
    delta = eps * rng.uniform(0.0, 1.0, 500)
    gamma = rng.uniform(0.0, 0.3, 500)
    assert is_stable(*thomsen_to_cij(vp0, vs0, 2300.0, eps, delta, gamma)).all()
    # C66 < 0 is not admissible
    bad = thomsen_to_cij(3000.0, 1500.0, 2300.0, 0.1, 0.05, -0.6)
    assert not is_stable(*bad)
    # a fluid is admissible only when allowed
    fl = thomsen_to_cij(1500.0, 0.0, 1030.0, 0.0, 0.0, 0.0)
    assert is_stable(*fl)
    assert not is_stable(*fl, allow_fluid=False)


# -- phase velocities ----------------------------------------------------------


def test_phase_velocity_limits():
    vp0, vs0, eps, delta, gamma = 2400.0, 950.0, 0.2, 0.08, 0.24
    vp, vsv, vsh = phase_velocities(0.0, vp0, vs0, eps, delta, gamma)
    np.testing.assert_allclose([vp, vsv, vsh], [vp0, vs0, vs0])
    vp, vsv, vsh = phase_velocities(np.pi / 2, vp0, vs0, eps, delta, gamma)
    np.testing.assert_allclose([vp, vsv, vsh],
                               [vp0 * np.sqrt(1 + 2 * eps), vs0, vs0 * np.sqrt(1 + 2 * gamma)])


def test_phase_velocity_weak_anisotropy_limit():
    # Thomsen (1986) weak-anisotropy form: Vp = Vp0 (1 + delta s^2 c^2 + eps s^4)
    theta = np.radians(np.linspace(0.0, 90.0, 19))
    vp0, vs0, eps, delta = 3000.0, 1500.0, 0.01, 0.005
    vp, _, _ = phase_velocities(theta, vp0, vs0, eps, delta, 0.0)
    s2, c2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    weak = vp0 * (1 + delta * s2 * c2 + eps * s2**2)
    np.testing.assert_allclose(vp, weak, rtol=2e-4)


def test_isotropic_phase_velocity_is_constant():
    theta = np.radians(np.linspace(0, 90, 7))
    vp, vsv, vsh = phase_velocities(theta, 3000.0, 1500.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(vp, 3000.0)
    np.testing.assert_allclose(vsv, 1500.0)
    np.testing.assert_allclose(vsh, 1500.0)


# -- Backus averaging --------------------------------------------------------


def test_backus_identical_layers_is_identity():
    c = thomsen_to_cij(2400.0, 950.0, 2280.0, 0.2, 0.08, 0.24)
    stack = Stiffness(*(np.full(5, x) for x in c))
    s, rho = backus_average(np.full(5, 0.5), stack, np.full(5, 2280.0))
    np.testing.assert_allclose(s, c, rtol=1e-12)
    assert rho == pytest.approx(2280.0)


def test_backus_of_isotropic_layers_has_gamma_and_eta_non_negative():
    # Fine layering of isotropic layers gives gamma >= 0 and eps >= delta
    # (eta >= 0), but eps itself can be negative (Berryman et al., 1999).
    rng = np.random.default_rng(3)
    n_negative_eps = 0
    for _ in range(2000):
        n = rng.integers(2, 8)
        vp0, vs0, rho, eps, delta, gamma = random_vti(n, rng, anisotropic=False)
        h = rng.uniform(0.1, 3.0, n)
        s, r = backus_average(h, thomsen_to_cij(vp0, vs0, rho, eps, delta, gamma), rho)
        _, _, e, d, g = cij_to_thomsen(*s, r)
        assert g >= -1e-12
        assert e - d >= -1e-12
        assert is_stable(*s)
        n_negative_eps += e < 0.0
    assert n_negative_eps > 0


def test_backus_two_layer_hand_calculation():
    # equal thicknesses of two isotropic layers
    c1 = thomsen_to_cij(2000.0, 800.0, 2100.0, 0, 0, 0)
    c2 = thomsen_to_cij(4000.0, 2200.0, 2600.0, 0, 0, 0)
    stack = Stiffness(*(np.array([a, b]) for a, b in zip(c1, c2)))
    s, rho = backus_average([1.0, 1.0], stack, [2100.0, 2600.0])
    assert s.c33 == pytest.approx(1.0 / (0.5 / c1.c33 + 0.5 / c2.c33))
    assert s.c44 == pytest.approx(1.0 / (0.5 / c1.c44 + 0.5 / c2.c44))
    assert s.c66 == pytest.approx(0.5 * (c1.c66 + c2.c66))
    assert rho == pytest.approx(2350.0)


def test_backus_velocity_below_ray_velocity():
    # Cauchy-Schwarz: <1/V>^2 = <sqrt(rho/M)>^2 <= <rho><1/M>, so the long-wave
    # (Backus) vertical velocity never exceeds the time-average (ray) velocity.
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = rng.integers(2, 10)
        vp0, vs0, rho, eps, delta, gamma = random_vti(n, rng)
        h = rng.uniform(0.1, 3.0, n)
        s, r = backus_average(h, thomsen_to_cij(vp0, vs0, rho, eps, delta, gamma), rho)
        v_backus = np.sqrt(s.c33 / r)
        v_ray = h.sum() / np.sum(h / vp0)
        assert v_backus <= v_ray * (1 + 1e-12)


def test_backus_rejects_mixed_fluid_solid():
    c1 = thomsen_to_cij(1500.0, 0.0, 1030.0, 0, 0, 0)
    c2 = thomsen_to_cij(2000.0, 800.0, 2100.0, 0, 0, 0)
    stack = Stiffness(*(np.array([a, b]) for a, b in zip(c1, c2)))
    with pytest.raises(ValueError):
        backus_average([1.0, 1.0], stack, [1030.0, 2100.0])


# -- the model class -----------------------------------------------------------


def three_layer():
    return ElasticModel1D(
        [0.0, 100.0, 150.0, 300.0],
        vp0=[1500.0, 2000.0, 3000.0],
        vs0=[0.0, 800.0, 1500.0],
        rho=[1030.0, 2100.0, 2400.0],
        eps=[0.0, 0.2, 0.05],
        delta=[0.0, 0.08, 0.02],
        gamma=[0.0, 0.25, 0.05],
        qp=[np.inf, 40.0, 100.0],
        qs=[np.inf, 20.0, 60.0],
        label=["water", "shale", "sand"],
    )


def test_model_validation():
    m = three_layer().validate()
    assert m.n_layers == 3 and len(m) == 3
    with pytest.raises(ValueError):
        ElasticModel1D([0.0, 10.0, 5.0], 2000.0, 800.0, 2100.0)
    with pytest.raises(ValueError):
        ElasticModel1D([0.0, 10.0], 2000.0, 2500.0, 2100.0).validate()
    with pytest.raises(ValueError):
        ElasticModel1D([0.0, 10.0], 1500.0, 0.0, 1030.0, eps=0.1).validate()
    with pytest.raises(ValueError, match="delta"):
        ElasticModel1D([0.0, 10.0], 3000.0, 1500.0, 2300.0, delta=-0.5).validate()


def test_layer_index_and_at():
    m = three_layer()
    np.testing.assert_array_equal(m.layer_index([0.0, 99.9, 100.0, 150.0, 300.0]),
                                  [0, 0, 1, 2, 2])
    assert m.at(120.0)["label"] == "shale"
    with pytest.raises(ValueError):
        m.layer_index(301.0)


def test_vertical_time_exact():
    m = three_layer()
    np.testing.assert_allclose(m.vertical_time(100.0), 100.0 / 1500.0)
    np.testing.assert_allclose(m.vertical_time(200.0),
                               100 / 1500 + 50 / 2000 + 50 / 3000)
    np.testing.assert_allclose(m.twt(300.0), 2 * (100 / 1500 + 50 / 2000 + 150 / 3000))
    z = np.linspace(0.0, 300.0, 31)
    np.testing.assert_allclose(m.depth_at_time(m.vertical_time(z)), z, atol=1e-9)
    # S time is infinite through the water
    assert np.isinf(m.vertical_time(200.0, "S"))
    with pytest.raises(ValueError):
        m.depth_at_time(0.1, "S")


def test_split_subset_regular_preserve_traveltime():
    m = three_layer()
    s = m.split_at([50.0, 120.0, 151.0])
    assert s.n_layers == 6
    np.testing.assert_allclose(s.interface_times()[-1], m.interface_times()[-1])
    sub = m.subset(120.0, 200.0)
    np.testing.assert_allclose(sub.interfaces, [120.0, 150.0, 200.0])
    np.testing.assert_array_equal(sub.label, ["shale", "sand"])
    r = m.regular(10.0)
    assert r.n_layers == 30
    np.testing.assert_allclose(r.interface_times()[-1], m.interface_times()[-1])
    with pytest.raises(ValueError):
        m.regular(7.0)


def test_block_preserves_constant_layers_and_q_weighting():
    m = three_layer()
    fine = m.regular(0.5)
    b = fine.block([100.0, 150.0])
    for name in ("vp0", "vs0", "rho", "eps", "delta", "gamma", "qp", "qs"):
        np.testing.assert_allclose(getattr(b, name), getattr(m, name), rtol=1e-10)
    np.testing.assert_array_equal(b.label, m.label)
    # two layers with equal vertical P time: 1/Q* is the mean of 1/Q
    two = ElasticModel1D([0.0, 20.0, 50.0], vp0=[2000.0, 3000.0], vs0=[1000.0, 1500.0],
                         rho=2200.0, qp=[20.0, 80.0], qs=[10.0, 40.0])
    blk = two.block([])
    assert blk.qp[0] == pytest.approx(1.0 / (0.5 / 20.0 + 0.5 / 80.0))
    assert blk.qs[0] == pytest.approx(1.0 / (0.5 / 10.0 + 0.5 / 40.0))


def test_block_keeps_fluid_and_solid_apart():
    b = three_layer().regular(1.0).block([150.0])
    np.testing.assert_allclose(b.interfaces, [0.0, 100.0, 150.0, 300.0])
    assert b.is_fluid[0] and not b.is_fluid[1:].any()


def test_block_label_is_the_dominant_unit():
    # 50 one-metre shale cells and 150 one-metre sand cells in one block
    b = three_layer().regular(1.0).block([])
    np.testing.assert_array_equal(b.label, ["water", "sand"])


def test_window_mean_exact():
    m = ElasticModel1D([0.0, 1.0, 3.0, 6.0], vp0=2000.0, vs0=1000.0, rho=2000.0)
    v = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(m.window_mean(v, [2.0], 2.0), [2.0])
    np.testing.assert_allclose(m.window_mean(v, [3.0], 2.0), [3.0])
    # truncated at the model top: [0, 1.5] -> (1*1 + 0.5*2) / 1.5
    np.testing.assert_allclose(m.window_mean(v, [0.0], 3.0), [2.0 / 1.5])
    # a segment boundary at 3 m stops the window
    np.testing.assert_allclose(m.window_mean(v, [2.5], 2.0, segments=[3.0]), [2.0])
    # zero-length window returns the layer value
    np.testing.assert_allclose(m.window_mean(v, [4.0], 0.0), [4.0])


def test_backus_running_average_limits():
    rng = np.random.default_rng(9)
    n = 40
    vp0, vs0, rho, eps, delta, gamma = random_vti(n, rng)
    z = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 1.0, n))])
    m = ElasticModel1D(z, vp0, vs0, rho, eps, delta, gamma, qp=50.0, qs=30.0)
    # a window shorter than every layer returns the model
    tiny = m.backus(0.0)
    for name in ("vp0", "vs0", "rho", "eps", "delta", "gamma"):
        np.testing.assert_allclose(getattr(tiny, name), getattr(m, name), rtol=1e-9, atol=1e-12)
    # a window longer than twice the model gives the global Backus average everywhere
    huge = m.backus(4.0 * z[-1])
    s, r = backus_average(m.thickness, m.cij(), m.rho)
    vp0g, vs0g, eg, dg, gg = cij_to_thomsen(*s, r)
    np.testing.assert_allclose(huge.vp0, vp0g, rtol=1e-10)
    np.testing.assert_allclose(huge.eps, eg, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(huge.delta, dg, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(huge.gamma, gg, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(huge.qp, 50.0)


def test_backus_running_average_respects_water_and_boundaries():
    m = three_layer().regular(1.0)
    b = m.backus(30.0, respect=[150.0])
    assert np.all(b.vs0[b.is_fluid] == 0.0)
    assert np.all(b.eps[b.is_fluid] == 0.0)
    water = m.layer_index(50.0)
    assert b.vp0[water] == pytest.approx(1500.0)
    # the shale and sand blocks are each homogeneous, so respecting 150 m keeps them
    for z in (110.0, 140.0, 160.0, 290.0):
        i = m.layer_index(z)
        assert b.vp0[i] == pytest.approx(m.vp0[i])
        assert b.gamma[i] == pytest.approx(m.gamma[i])
    # without the respect depth, layers near 150 m mix
    mixed = m.backus(30.0)
    i = m.layer_index(149.0)
    assert mixed.vp0[i] != pytest.approx(m.vp0[i])
