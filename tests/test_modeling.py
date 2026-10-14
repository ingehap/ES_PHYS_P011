import numpy as np
import pytest

from iwt.model import ElasticModel1D, thomsen_to_cij
from iwt.modeling import reflection
from iwt.modeling.convolution import angle_gather, normal_incidence, ricker
from iwt.modeling.elastic import (
    complex_velocity,
    eigenvectors,
    energy_flux,
    fluid_eigenvectors,
    interface_scattering,
    vertical_slowness,
)
from iwt.modeling.reflectivity import (
    Reflectivity,
    spectrum_to_trace,
    time_axis_spectrum,
    wavelet_spectrum,
)
from iwt.qmodel import attenuation_coefficient, phase_velocity

SHALE = (2800.0, 1300.0, 2450.0, 0.12, 0.05)
SAND = (2925.0, 1667.0, 2166.0, 0.02, 0.01)


def solid_D(medium, p):
    vp0, vs0, rho, eps, delta = medium
    c = thomsen_to_cij(vp0, vs0, rho, eps, delta, 0.0)
    p = np.asarray(p, dtype=complex)
    qp, qs = vertical_slowness(c.c11 + 0j, c.c13 + 0j, c.c33 + 0j, c.c44 + 0j, rho, p)
    return eigenvectors(c.c11, c.c13, c.c33, c.c44, rho, p, qp, qs), qp, qs


# -- elastic layer physics --------------------------------------------------------


def test_vertical_slowness_isotropic_and_vti():
    p = np.linspace(0.0, 3e-4, 7)
    c = thomsen_to_cij(3000.0, 1500.0, 2300.0, 0.0, 0.0, 0.0)
    qp, qs = vertical_slowness(c.c11 + 0j, c.c13 + 0j, c.c33 + 0j, c.c44 + 0j, 2300.0, p)
    np.testing.assert_allclose(qp, np.sqrt(1 / 3000.0**2 - p**2), rtol=1e-12)
    np.testing.assert_allclose(qs, np.sqrt(1 / 1500.0**2 - p**2), rtol=1e-12)
    # VTI: the phase velocity along (p, q) equals the exact qP phase velocity
    from iwt.model import phase_velocities

    vp0, vs0, rho, eps, delta = SHALE
    c = thomsen_to_cij(vp0, vs0, rho, eps, delta, 0.0)
    qp, _ = vertical_slowness(c.c11 + 0j, c.c13 + 0j, c.c33 + 0j, c.c44 + 0j, rho, p)
    theta = np.arctan2(p, qp.real)
    v = 1.0 / np.sqrt(p**2 + qp.real**2)
    np.testing.assert_allclose(v, phase_velocities(theta, vp0, vs0, eps, delta, 0.0)[0], rtol=1e-10)


def test_evanescent_branch_decays():
    c = thomsen_to_cij(2000.0, 900.0, 2200.0, 0.1, 0.05, 0.0)
    qp, qs = vertical_slowness(c.c11 + 0j, c.c13 + 0j, c.c33 + 0j, c.c44 + 0j, 2200.0,
                               np.array([6e-4 + 0j]))
    assert qp.imag[0] > 0.0 and abs(qp.real[0]) < 1e-12
    assert qs.real[0] > 0.0


def test_complex_velocity_matches_constant_q_law():
    c0, q, f0 = 2500.0, 40.0, 30.0
    f = np.array([5.0, 30.0, 80.0])
    v = complex_velocity(c0, q, 2 * np.pi * f, 2 * np.pi * f0)
    k = 2 * np.pi * f / v
    np.testing.assert_allclose(2 * np.pi * f / k.real, phase_velocity(c0, q, f, f0), rtol=1e-12)
    np.testing.assert_allclose(k.imag, attenuation_coefficient(c0, q, f, f0), rtol=1e-12)
    np.testing.assert_allclose(complex_velocity(c0, np.inf, 2 * np.pi * f, 1.0), c0)


def test_zoeppritz_explicit_equals_eigenvector_solution():
    theta = np.radians(np.arange(0.0, 61.0, 5.0))
    iso_up, iso_lo = SHALE[:3] + (0.0, 0.0), SAND[:3] + (0.0, 0.0)
    z = reflection.zoeppritz_pp(theta, *SHALE[:3], *SAND[:3])
    e = reflection.exact_pp(theta, iso_up, iso_lo)
    np.testing.assert_allclose(e, z, atol=1e-12)
    r0 = (SAND[0] * SAND[2] - SHALE[0] * SHALE[2]) / (SAND[0] * SAND[2] + SHALE[0] * SHALE[2])
    assert z[0].real == pytest.approx(r0, abs=1e-14)


def test_linear_approximations():
    theta = np.radians(np.arange(0.0, 31.0, 5.0))
    z = reflection.zoeppritz_pp(theta, *SHALE[:3], *SAND[:3]).real
    ar = reflection.aki_richards(theta, *SHALE[:3], *SAND[:3])
    sh = reflection.shuey(theta, *SHALE[:3], *SAND[:3])
    np.testing.assert_allclose(ar, z, atol=0.002)
    np.testing.assert_allclose(sh, ar, atol=1e-3)
    iso = SHALE[:3] + (0.0, 0.0), SAND[:3] + (0.0, 0.0)
    np.testing.assert_allclose(reflection.ruger(theta, *iso), sh, atol=1e-15)
    # Ruger tracks the exact VTI coefficient for weak anisotropy and small angles
    np.testing.assert_allclose(reflection.ruger(theta, SHALE, SAND),
                               reflection.exact_pp(theta, SHALE, SAND).real, atol=0.002)


@pytest.mark.parametrize("upper,lower", [(SHALE, SAND), (SAND, SHALE),
                                         ((2400.0, 950.0, 2280.0, 0.2, 0.08),
                                          (4300.0, 2350.0, 2620.0, 0.0, 0.0))])
def test_solid_interface_conserves_energy_and_is_reciprocal(upper, lower):
    p = np.array([0.0, 5e-5, 1.5e-4, 2.2e-4]) + 0j     # below every critical slowness
    Du, _, _ = solid_D(upper, p)
    Dl, _, _ = solid_D(lower, p)
    RD, TD, RU, TU = interface_scattering(Du, Dl, scale=6e6)
    fu = energy_flux(Du, [0, 1])   # P, S flux in the upper medium (same for up/down)
    fl = energy_flux(Dl, [0, 1])
    for k in range(2):             # P or S incident from above
        out = np.sum(np.abs(RD[:, :, k]) ** 2 * fu, axis=-1) + np.sum(np.abs(TD[:, :, k]) ** 2 * fl, axis=-1)
        np.testing.assert_allclose(out, fu[:, k], rtol=1e-10)
    # reciprocity: the flux-normalized scattering matrix is symmetric up to the
    # SV sign convention, S = J S^T J with J = diag(1, -1, 1, -1)
    S = np.zeros((p.size, 4, 4), dtype=complex)
    S[:, :2, :2], S[:, :2, 2:], S[:, 2:, :2], S[:, 2:, 2:] = RD, TU, TD, RU
    w = np.sqrt(np.concatenate([fu, fl], axis=-1))
    Sn = S * w[:, :, None] / w[:, None, :]
    J = np.array([1.0, -1.0, 1.0, -1.0])
    np.testing.assert_allclose(Sn, J[:, None] * np.swapaxes(Sn, 1, 2) * J[None, :], atol=1e-10)


def test_fluid_solid_interface():
    rho_w, a_w = 1030.0, 1480.0
    p = np.array([0.0, 1e-4, 2.5e-4]) + 0j
    qw = np.sqrt(1 / a_w**2 - p**2)
    Df = fluid_eigenvectors(rho_w * a_w**2, rho_w, p, qw)
    Ds, _, _ = solid_D((1750.0, 450.0, 1950.0, 0.0, 0.0), p)
    RD, TD, RU, TU = interface_scattering(Df, Ds, fluid_upper=True, scale=2e6)
    # normal incidence
    z1, z2 = rho_w * a_w, 1950.0 * 1750.0
    assert RD[0, 0, 0].real == pytest.approx((z2 - z1) / (z2 + z1), abs=1e-12)
    assert abs(TD[0, 1, 0]) < 1e-12           # no P->S conversion at p = 0
    # energy: incident water P = reflected water P + transmitted P and S
    fw = energy_flux(Df, [0])[:, 0]
    fs = energy_flux(Ds, [0, 1])
    out = np.abs(RD[:, 0, 0]) ** 2 * fw + np.sum(np.abs(TD[:, :, 0]) ** 2 * fs, axis=-1)
    np.testing.assert_allclose(out, fw, rtol=1e-10)
    # the dummy S channel of the fluid is never coupled
    assert np.all(RD[:, 1, :] == 0) and np.all(TU[:, 1, :] == 0)


# -- reflectivity and convolution -------------------------------------------------


def layered():
    return ElasticModel1D([0.0, 300.0, 400.0, 450.0, 600.0],
                          vp0=[1480.0, 2000.0, 2600.0, 2300.0],
                          vs0=[0.0, 800.0, 1300.0, 1000.0],
                          rho=[1030.0, 2100.0, 2300.0, 2200.0],
                          eps=[0.0, 0.1, 0.02, 0.15], delta=[0.0, 0.05, 0.01, 0.06],
                          qp=[np.inf, 40.0, 80.0, 50.0], qs=[np.inf, 20.0, 50.0, 30.0])


@pytest.mark.parametrize("attenuation", [False, True])
def test_reflectivity_equals_convolution_at_normal_incidence(attenuation):
    m = layered()
    om, spec = normal_incidence(m, 0.002, 512, None, top=10.0, attenuation=attenuation,
                                transmission_loss=True, f_max=100.0, return_spectrum=True)
    refl = Reflectivity(m, attenuation=attenuation, free_surface=False, internal_multiples=False)
    up = refl.plane_wave(om, [0.0], 10.0, [10.0])["up_p"][0, :, 0]
    np.testing.assert_allclose(up, spec, atol=1e-12)


def test_internal_multiples_and_free_surface_switches():
    m = layered()
    om = 2 * np.pi * np.array([10.0, 25.0, 40.0]) + 0.5j
    base = dict(attenuation=False, converted_waves=True)
    prim = Reflectivity(m, free_surface=False, internal_multiples=False, **base)
    full = Reflectivity(m, free_surface=False, internal_multiples=True, **base)
    a = prim.plane_wave(om, [0.0], 10.0, [10.0])["up_p"][0, :, 0]
    b = full.plane_wave(om, [0.0], 10.0, [10.0])["up_p"][0, :, 0]
    assert np.max(np.abs(b - a)) > 1e-4
    # the first internal multiple: R2 R1' R2 path inside layer 2 has the right order of size
    assert np.max(np.abs(b - a)) < 0.2 * np.max(np.abs(a))
    # free surface: a water-bottom multiple appears; for a pure water half-space the
    # receiver sees exactly the source ghost (-1 times the delayed downgoing wave)
    water = ElasticModel1D([0.0, 500.0, 1000.0], vp0=1480.0, vs0=0.0, rho=1030.0)
    w = 2 * np.pi * np.array([10.0, 30.0]) + 0.3j
    out = Reflectivity(water, free_surface=True).plane_wave(w, [0.0], 20.0, [100.0],
                                                             source="explosion")
    expected_down = np.exp(1j * w * 80.0 / 1480.0) * (1.0 - np.exp(1j * w * 40.0 / 1480.0))
    np.testing.assert_allclose(out["down_p"][0, :, 0], expected_down, rtol=1e-12)
    np.testing.assert_allclose(out["up_p"][0, :, 0], 0.0, atol=1e-14)


def test_converted_waves_switch():
    m = layered()
    om = np.array([2 * np.pi * 30.0 + 0.5j])
    p = [2e-4]
    on = Reflectivity(m, free_surface=False).plane_wave(om, p, 10.0, [500.0])
    off = Reflectivity(m, free_surface=False, converted_waves=False).plane_wave(om, p, 10.0, [500.0])
    assert abs(on["down_s"][0, 0, 0]) > 1e-3
    assert abs(off["down_s"][0, 0, 0]) == 0.0


def test_direct_wave_matches_analytic_point_source():
    alpha, rho = 1480.0, 1030.0
    m = ElasticModel1D([0.0, 2000.0, 4000.0], vp0=alpha, vs0=0.0, rho=rho)
    dt, nt = 0.001, 1024
    wav = ricker(40.0, dt)
    zs, zr = 50.0, 250.0
    offs = np.array([0.0, 100.0])
    rec = Reflectivity(m, free_surface=False).record(offs, zs, [zr], dt, nt, wav, "pressure",
                                                     f_max=120.0)
    _, om, eps, _ = time_axis_spectrum(dt, 2048, 120.0)
    W = wavelet_spectrum(*wav, om)
    for i, r in enumerate(offs):
        R = np.hypot(r, zr - zs)
        an = spectrum_to_trace(rho * om**2 * W * np.exp(1j * om * R / alpha) / (4 * np.pi * R),
                               dt, 2048, eps, om, 0.05)[:nt]
        k = np.argmax(np.abs(an))
        win = slice(k - 40, k + 40)
        assert np.argmax(np.abs(rec.data[0, i])) == k
        err = np.linalg.norm((rec.data[0, i] - an)[win]) / np.linalg.norm(an[win])
        assert err < 1e-3
        assert np.max(np.abs(rec.data[0, i][k + 100:])) < 0.01 * np.max(np.abs(an))


def test_seabed_reflection_matches_image_source():
    # water over a solid half-space: away from the critical angle the reflection is
    # the image-source wave scaled by the plane-wave coefficient at the specular angle
    alpha, rho = 1480.0, 1030.0
    solid = (2200.0, 900.0, 2100.0)
    m = ElasticModel1D([0.0, 400.0, 2000.0], vp0=[alpha, solid[0]], vs0=[0.0, solid[1]],
                       rho=[rho, solid[2]])
    dt, nt = 0.0005, 2048
    wav = ricker(40.0, dt)
    zs = zr = 10.0
    offs = np.array([100.0, 300.0, 500.0])
    rec = Reflectivity(m, free_surface=False, attenuation=False).record(
        offs, zs, [zr], dt, nt, wav, "pressure", f_max=120.0)
    _, om, eps, _ = time_axis_spectrum(dt, 4096, 120.0)
    W = wavelet_spectrum(*wav, om)
    for i, x in enumerate(offs):
        path = np.hypot(x, 2 * 400.0 - zs - zr)
        r_pp = reflection.zoeppritz_pp(np.arctan2(x, 780.0), alpha, 1e-6, rho, *solid).real
        image = spectrum_to_trace(r_pp * rho * om**2 * W * np.exp(1j * om * path / alpha)
                                  / (4 * np.pi * path), dt, 4096, eps, om, 0.05)[:nt]
        k = np.argmax(np.abs(image))
        assert abs(np.argmax(np.abs(rec.data[0, i][k - 30:k + 30])) - 30) <= 1
        assert rec.data[0, i][k] / image[k] == pytest.approx(1.0, abs=0.02)


def test_vertical_attenuation_matches_constant_q():
    # downgoing plane wave through 200 m of Q = 40 rock: amplitude loss exp(-alpha h)
    m = ElasticModel1D([0.0, 100.0, 300.0, 1000.0], vp0=[1480.0, 2500.0, 2500.0],
                       vs0=[0.0, 1100.0, 1100.0], rho=[1030.0, 2200.0, 2200.0],
                       qp=[np.inf, 40.0, 40.0], qs=[np.inf, 25.0, 25.0])
    f = np.array([10.0, 30.0, 60.0])
    out = Reflectivity(m, free_surface=False).plane_wave(2 * np.pi * f, [0.0], 50.0, [100.5, 300.5])
    ratio = np.abs(out["down_p"][1, :, 0] / out["down_p"][0, :, 0])
    expected = np.exp(-attenuation_coefficient(2500.0, 40.0, f, 30.0) * 200.0)
    np.testing.assert_allclose(ratio, expected, rtol=1e-10)


def test_up_down_split_of_vertical_displacement():
    """uz_down + uz_up = uz in the sea and in the solid; nothing goes up below the last interface."""
    m = ElasticModel1D([0.0, 100.0, 300.0, 1000.0], vp0=[1480.0, 2500.0, 3000.0],
                       vs0=[0.0, 1100.0, 1500.0], rho=[1030.0, 2200.0, 2400.0],
                       qp=[np.inf, 40.0, 60.0], qs=[np.inf, 25.0, 40.0])
    f = np.array([10.0, 30.0, 60.0])
    out = Reflectivity(m).plane_wave(2 * np.pi * f + 0.3j, [0.0, 1e-4], 6.0, [50.0, 200.0, 500.0],
                                     source="explosion")
    np.testing.assert_allclose(out["uz_down"] + out["uz_up"], out["uz"], rtol=1e-12, atol=1e-15)
    assert np.all(np.abs(out["uz_up"][:2]) > 0)
    np.testing.assert_allclose(out["uz_up"][2], 0.0, atol=1e-14)


def test_angle_gather_methods_and_timing():
    m = ElasticModel1D([0.0, 500.0, 600.0, 800.0], vp0=[2400.0, 2800.0, 2925.0],
                       vs0=[950.0, 1300.0, 1667.0], rho=[2280.0, 2450.0, 2166.0],
                       eps=[0.2, 0.12, 0.02], delta=[0.08, 0.05, 0.01])
    dt, nt = 0.001, 1024
    wav = ricker(30.0, dt)
    angles = [0.0, 20.0, 40.0]
    t, ang, z = angle_gather(m, angles, dt, nt, wav, "zoeppritz")
    _, _, rg = angle_gather(m, angles, dt, nt, wav, "ruger")
    _, _, ex = angle_gather(m, angles, dt, nt, wav, "exact_vti")
    k = int(round(float(m.twt(500.0)) / dt))
    assert abs(np.argmax(np.abs(z[0])) - k) <= 1
    r0 = (2800 * 2450 - 2400 * 2280) / (2800 * 2450 + 2400 * 2280)
    assert z[0, k] == pytest.approx(r0, rel=0.05)
    np.testing.assert_allclose(rg[:2], ex[:2], atol=0.01 * np.abs(ex).max())
    with pytest.raises(ValueError):
        angle_gather(m, angles, dt, nt, wav, "bogus")


def test_reflectivity_input_checks():
    m = layered()
    r = Reflectivity(m)
    with pytest.raises(NotImplementedError):
        r.plane_wave([1.0], [0.0], 350.0, [400.0])        # source in the solid
    with pytest.raises(ValueError):
        r.plane_wave([1.0], [0.0], 50.0, [20.0])          # receiver above the source
    bad = ElasticModel1D([0.0, 10.0, 20.0, 30.0], vp0=[1500.0, 2000.0, 1500.0],
                         vs0=[0.0, 800.0, 0.0], rho=[1000.0, 2000.0, 1000.0])
    with pytest.raises(NotImplementedError):
        Reflectivity(bad)


def test_closed_form_interface_equals_generic_solve():
    m = ElasticModel1D([0.0, 10.0, 20.0, 30.0], vp0=[2800.0, 2925.0, 4300.0],
                       vs0=[1300.0, 1667.0, 2350.0], rho=[2450.0, 2166.0, 2620.0],
                       eps=[0.12, 0.02, 0.0], delta=[0.05, 0.01, 0.0],
                       qp=[40.0, 90.0, 150.0], qs=[25.0, 60.0, 100.0])
    r = Reflectivity(m)
    om = 2 * np.pi * np.array([5.0, 30.0, 80.0]) + 0.4j
    p = np.linspace(0.0, 8e-4, 9)[None, :] + 0j      # includes evanescent slownesses
    vp, vs = r._velocities(m, om)

    def full(lay):
        D = np.empty(np.broadcast(lay["Ad"][0], om[:, None]).shape + (4, 4), complex)
        for (i, c), b in (((0, 0), "Ad"), ((0, 2), "Au"), ((2, 0), "Cd"), ((2, 2), "Cu")):
            x = lay[b]
            D[..., i, c], D[..., i, c + 1], D[..., i + 1, c], D[..., i + 1, c + 1] = x
        return D

    for j in (1, 2):
        up, lo = r._layer(m, vp, vs, j - 1, p, 6e6), r._layer(m, vp, vs, j, p, 6e6)
        fast = r._interface(up, lo, 6e6)
        ref = interface_scattering(full(up), full(lo))
        for f_, g in zip(fast, ref):
            for k in range(4):
                np.testing.assert_allclose(f_[k], g[..., k // 2, k % 2], atol=1e-12)


def test_parallel_and_chunked_results_are_identical():
    m = layered()
    om = 2 * np.pi * np.linspace(2.0, 60.0, 12) + 0.3j
    p = np.linspace(0.0, 4e-4, 30)
    r = Reflectivity(m)
    a = r.plane_wave(om, p, 10.0, [10.0, 420.0], source="explosion")
    b = r.plane_wave(om, p, 10.0, [10.0, 420.0], source="explosion", chunk=60)
    c = r.plane_wave(om, p, 10.0, [10.0, 420.0], source="explosion", workers=2)
    for k in ("pressure", "uz", "up_p"):
        np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a[k], c[k])


def test_born_primaries_equal_convolution_without_transmission_loss():
    m = layered()
    om, spec = normal_incidence(m, 0.002, 512, None, top=10.0, attenuation=True,
                                transmission_loss=False, f_max=100.0, return_spectrum=True)
    refl = Reflectivity(m, free_surface=False, internal_multiples=False, transmission_loss=False)
    up = refl.plane_wave(om, [0.0], 10.0, [10.0])["up_p"][0, :, 0]
    np.testing.assert_allclose(up, spec, atol=1e-12)
    with pytest.raises(ValueError):
        Reflectivity(m, internal_multiples=True, transmission_loss=False)


def test_short_period_multiples_compensate_transmission_loss():
    # O'Doherty & Anstey (1971): in a finely layered stack, short-period multiples
    # restore much of the amplitude that transmission losses remove
    rng = np.random.default_rng(0)
    n = 400
    vp = 2500.0 * (1.0 + 0.05 * rng.standard_normal(n))
    z = np.concatenate([[0.0, 100.0], 100.0 + np.cumsum(np.full(n, 0.5)), [400.0]])
    m = ElasticModel1D(z, vp0=np.concatenate([[1480.0], vp, [3000.0]]),
                       vs0=np.concatenate([[0.0], vp / 2.0, [1500.0]]),
                       rho=np.concatenate([[1030.0], np.full(n, 2300.0), [2400.0]]))
    dt, nt = 0.002, 512
    wav = ricker(30.0, dt)
    _, om, eps, _ = time_axis_spectrum(dt, nt, 80.0)
    W = wavelet_spectrum(*wav, om)
    t = np.arange(nt) * dt
    traces = {}
    for name, kw in (("full", {}), ("tl", dict(internal_multiples=False)),
                     ("born", dict(internal_multiples=False, transmission_loss=False))):
        up = Reflectivity(m, free_surface=False, attenuation=False, **kw).plane_wave(
            om, [0.0], 10.0, [10.0])["up_p"][0, :, 0]
        traces[name] = spectrum_to_trace(up * W, dt, nt, eps, om, -wav[0][0])
    tb = float(m.twt(300.0) - m.twt(10.0))           # base of the layered stack
    w = (t > tb - 0.02) & (t < tb + 0.04)
    rms = {k: np.sqrt(np.mean(v[w] ** 2)) for k, v in traces.items()}
    assert rms["tl"] < 0.6 * rms["born"]          # two-way transmission loss is large
    assert rms["full"] > 1.4 * rms["tl"]          # multiples restore much of it
    assert rms["full"] < rms["born"]
