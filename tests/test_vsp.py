"""Tests of the ZO-VSP processing (iwt.vsp) on reflectivity synthetics."""

import numpy as np
import pytest

from iwt.model import ElasticModel1D
from iwt.modeling.convolution import normal_incidence, ricker
from iwt.qmodel import phase_velocity
from iwt.synthetic import wisting_like_model
from iwt.vsp import attenuation as A
from iwt.vsp import zovsp as Z

DT = 0.001
V_WATER, V_ROCK = 1480.0, 2500.0


def _half_space(q=np.inf, q2=None, z2=600.0):
    """Sea (0-100 m) over rock with Qp = q (and q2 below z2)."""
    if q2 is None:
        return ElasticModel1D([0.0, 100.0, 1500.0], vp0=[V_WATER, V_ROCK],
                              vs0=[0.0, 1200.0], rho=[1030.0, 2200.0],
                              qp=[np.inf, q], qs=[np.inf, q / 2])
    return ElasticModel1D([0.0, 100.0, z2, 1500.0], vp0=[V_WATER, V_ROCK, V_ROCK],
                          vs0=[0.0, 1200.0, 1200.0], rho=[1030.0, 2200.0, 2200.0],
                          qp=[np.inf, q, q2], qs=[np.inf, q / 2, q2 / 2])


def _direct_only(model, depths, nt=1024):
    """Downgoing source, no free surface: the direct wave only (in a homogeneous rock)."""
    return Z.synthetic_zovsp(model, depths, 6.0, DT, nt, ricker(40.0, DT), f_max=200.0,
                             free_surface=False, source="downgoing")


def _time(depths, f=None, q=np.inf):
    c = V_ROCK if f is None else phase_velocity(V_ROCK, q, f, 30.0)
    return (100.0 - 6.0) / V_WATER + (np.asarray(depths) - 100.0) / c


@pytest.fixture(scope="module")
def layered():
    """p = 0 VSPs of the upper Wisting-like well: with Q, without Q, and the exact downgoing part."""
    m = wisting_like_model().subset(0.0, 1000.0)
    depths = np.arange(450.0, 950.1, 10.0)
    kw = dict(f_max=150.0)
    wav = ricker(40.0, DT)
    full = Z.synthetic_zovsp(m, depths, 6.0, DT, 1024, wav, **kw)
    elastic = Z.synthetic_zovsp(m, depths, 6.0, DT, 1024, wav, attenuation=False, **kw)
    down = Z.synthetic_zovsp(m, depths, 6.0, DT, 1024, wav, component="uz_down", **kw)
    return m, full, elastic, down


# -- trace utilities ----------------------------------------------------------------


def test_shift_traces_is_exact_for_fractional_shifts():
    t = np.arange(512) * DT
    tw, w = ricker(30.0, DT, 0.2)
    x = np.interp(t - 0.2, tw, w)
    shifted = Z.shift_traces(x[None, :], [0.0123], DT)[0]
    a = (np.pi * 30.0 * (t - 0.2123)) ** 2
    assert np.max(np.abs(shifted - (1 - 2 * a) * np.exp(-a))) < 2e-3


def test_add_noise_level_and_reproducibility():
    data = np.zeros((3, 1000))
    data[:, 100] = [1.0, 2.0, 4.0]
    n1 = Z.add_noise(data, 0.1, DT, f_max=100.0, seed=3)
    n2 = Z.add_noise(data, 0.1, DT, f_max=100.0, seed=3)
    assert np.array_equal(n1, n2)
    resid = n1 - data
    assert np.allclose(np.std(resid, axis=1), 0.1 * np.array([1.0, 2.0, 4.0]), rtol=1e-6)


# -- picking and time-depth ---------------------------------------------------------


def test_picks_without_attenuation():
    depths = np.arange(200.0, 1201.0, 50.0)
    v = _direct_only(_half_space(), depths)
    peak = Z.pick_first_breaks(v.t, v.data, "peak")
    onset = Z.pick_first_breaks(v.t, v.data, "onset")
    # a zero-phase source: the peak is the arrival time
    assert np.max(np.abs(peak - _time(depths))) < 5e-5
    # the onset precedes it by a constant (the wavelet's rise time)
    lead = peak - onset
    assert np.all(lead > 0.005) and np.ptp(lead) < 2e-4


def test_vertical_times_cosine_correction():
    tv = Z.vertical_times([0.5], [1006.0], source_depth=6.0, source_offset=100.0)
    assert np.isclose(tv[0], 0.5 * 1000.0 / np.hypot(1000.0, 100.0))
    tv = Z.vertical_times([0.5], [1006.0], source_depth=6.0, datum_velocity=1500.0)
    assert np.isclose(tv[0], 0.5 + 6.0 / 1500.0)


@pytest.mark.parametrize("f", [20.0, 30.0, 60.0])
def test_phase_delay_times_are_constant_q_traveltimes(f):
    q = 50.0
    depths = np.arange(200.0, 1201.0, 50.0)
    v = _direct_only(_half_space(q), depths)
    pk = Z.pick_first_breaks(v.t, v.data)
    tau = Z.phase_delay_times(v.t, v.data, f, pk)
    exact = _time(depths, f, q)
    assert np.max(np.abs((tau - tau[0]) - (exact - exact[0]))) < 5e-5
    # no pick of the pulse shape gives the traveltime at the seismic frequency
    pk_rel = (pk - pk[0]) - (exact - exact[0])
    assert np.max(np.abs(pk_rel)) > 4e-4


def test_peak_picks_follow_one_lobe(layered):
    """Source ghost and interference: the peak picks must not jump between lobes."""
    _, full, elastic, _ = layered
    for v in (full, elastic):
        pk = Z.pick_first_breaks(v.t, v.data)
        assert np.max(np.abs(np.diff(pk, 2))) < 1.5e-3


def test_interval_velocity_regularization_beats_differencing():
    rng = np.random.default_rng(1)
    z = np.arange(500.0, 1500.1, 10.0)
    edges = np.array([500.0, 800.0, 1100.0, 1500.0])
    v_true = np.array([2400.0, 3000.0, 2700.0])
    slow = np.repeat(1.0 / v_true, np.diff(edges).astype(int) // 10)
    t = 0.3 + np.concatenate([[0.0], np.cumsum(slow * 10.0)])
    tn = t + 5e-4 * rng.standard_normal(z.size)

    blocked = Z.interval_velocity(z, t, boundaries=edges[1:-1])
    assert np.allclose(blocked.velocity, v_true, rtol=1e-9)
    noisy = Z.interval_velocity(z, tn, boundaries=edges[1:-1])
    assert np.allclose(noisy.velocity, v_true, rtol=0.015)

    raw = Z.interval_velocity(z, tn, smoothing=0.0)
    smooth = Z.interval_velocity(z, tn)                  # sigma from the data
    truth = 1.0 / slow
    err_raw = np.sqrt(np.mean((raw.velocity - truth) ** 2))
    err_smooth = np.sqrt(np.mean((smooth.velocity - truth) ** 2))
    assert err_smooth < 0.5 * err_raw
    assert 2e-4 < smooth.sigma < 1e-3                    # estimated noise level
    assert np.isclose(np.sqrt(np.mean((smooth.time_fit - tn) ** 2)), smooth.sigma, rtol=1e-3)


# -- separation, deconvolution, corridor stack --------------------------------------


def test_parametric_separation_beats_median(layered):
    _, full, _, down = layered
    t = full.t
    pk = Z.pick_first_breaks(t, full.data)
    up_exact = full.data - down.data
    late = t[None, :] > pk[:, None] + 0.03
    err = {}
    for method, length in (("parametric", 5), ("median", 9)):
        d, u = Z.separate_wavefields(t, full.data, pk, length, method)
        err[method] = np.sqrt(np.sum(((u - up_exact) ** 2 * late)[5:-5])
                              / np.sum((up_exact ** 2 * late)[5:-5]))
        assert np.sqrt(np.sum((d - down.data)[5:-5] ** 2) / np.sum(down.data[5:-5] ** 2)) < 0.1
    assert err["parametric"] < 0.3
    assert err["parametric"] < err["median"]


def test_corridor_stack_matches_reflectivity(layered):
    m, full, _, _ = layered
    t = full.t
    pk = Z.pick_first_breaks(t, full.data)
    down, up = Z.separate_wavefields(t, full.data, pk)
    tv = Z.vertical_times(Z.phase_delay_times(t, down, 30.0, pk), full.depths, 6.0,
                          datum_velocity=1470.0)
    target = ricker(30.0, DT)
    dec = Z.deconvolve(t, up, down, tv, target)
    stack, fold = Z.corridor_stack(t, dec, tv, length=0.1)
    _, born = normal_incidence(m, DT, t.size, target)       # SEG polarity, sea-level datum
    w = (t > 2 * tv.min() + 0.02) & (t < 2 * tv.max())
    r = np.corrcoef(stack[w], born[w])[0, 1]
    assert r > 0.8                                          # acceptance criterion of the plan
    assert fold.max() > 1.0 and fold[t < 2 * tv.min() - 0.02].max() == 0.0
    full_stack, _ = Z.corridor_stack(t, dec, tv, length=None)
    assert np.all(np.isfinite(full_stack))


def test_corridor_stack_fold_and_normalization():
    t = np.arange(400) * DT
    d = np.ones((3, t.size))
    times = np.array([0.05, 0.06, 0.07])
    stack, fold = Z.corridor_stack(t, d, times, length=0.05, taper=0.0)
    assert np.allclose(stack[fold > 0], 1.0)
    assert fold[np.searchsorted(t, 0.145)] == 3.0
    assert fold[np.searchsorted(t, 0.105)] == 1.0


# -- attenuation --------------------------------------------------------------------


def test_q_estimators_homogeneous():
    q = 50.0
    depths = np.arange(200.0, 1201.0, 20.0)
    v = _direct_only(_half_space(q), depths)
    pk = Z.pick_first_breaks(v.t, v.data)
    f, S = A.first_arrival_spectra(v.t, v.data, pk)
    true = (depths - depths[0]) / V_ROCK / q
    ts, a, rms = A.spectral_ratio(f, S, (10.0, 100.0))
    assert np.allclose(ts[1:] / true[1:], 1.0, atol=0.03)
    assert rms.max() < 0.02 and np.abs(a).max() < 0.02
    tc = A.centroid_shift(f, S, (10.0, 100.0))
    assert np.allclose(tc[1:] / true[1:], 1.0, atol=0.03)
    # the Gaussian closed form is biased for a non-Gaussian (Ricker) spectrum
    tg = A.centroid_shift(f, S, (10.0, 100.0), method="gaussian")
    assert np.abs(tg[-1] / true[-1] - 1.0) > np.abs(tc[-1] / true[-1] - 1.0)
    times = _time(depths)
    iq = A.interval_q(depths, times, ts)
    assert np.isclose(iq.q[0], q, rtol=0.03)


def test_interval_q_two_layers_with_noise():
    depths = np.arange(200.0, 1201.0, 20.0)
    v = _direct_only(_half_space(40.0, 100.0), depths)
    times = _time(depths)
    for level, tol in ((0.0, 0.05), (0.01, 0.15)):
        data = Z.add_noise(v.data, level, DT, f_max=200.0, seed=11) if level else v.data
        pk = Z.pick_first_breaks(v.t, data)
        f, S = A.first_arrival_spectra(v.t, data, pk)
        ts = A.spectral_ratio(f, S, (10.0, 90.0))[0]
        iq = A.interval_q(depths, times, ts, boundaries=[600.0])
        assert np.allclose(iq.q, [40.0, 100.0], rtol=tol), (level, iq.q)
        assert np.all(iq.inv_q_std > 0)
    assert np.allclose(iq.at([300.0, 900.0, 2000.0], fill=-1.0), [iq.q[0], iq.q[1], -1.0])


def test_intrinsic_q_from_stratigraphic_correction(layered):
    """1/Q_int = 1/Q_app - 1/Q_strat recovers the model Q; the apparent Q is biased."""
    m, full, elastic, _ = layered
    t, depths = full.t, full.depths
    bounds = [600.0, 740.0]
    edges = np.concatenate([[depths[0]], bounds, [depths[-1]]])

    def tstar(v):
        pk = Z.pick_first_breaks(t, v.data)
        down, _ = Z.separate_wavefields(t, v.data, pk)
        f, S = A.first_arrival_spectra(t, down, pk)
        return Z.phase_delay_times(t, down, 30.0, pk), A.spectral_ratio(f, S, (10.0, 100.0))[0]

    tv, ts_app = tstar(full)
    _, ts_strat = tstar(elastic)
    app = A.interval_q(depths, tv, ts_app, bounds).inv_q
    intr = A.interval_q(depths, tv, ts_app - ts_strat, bounds).inv_q
    true = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = m.split_at([a, b])
        i = (s.mid > a) & (s.mid < b)
        w = s.thickness[i] / s.vp0[i]
        true.append(np.sum(w / s.qp[i]) / np.sum(w))
    true = np.array(true)
    err_int = np.abs(intr / true - 1.0)
    err_app = np.abs(app / true - 1.0)
    assert np.all(err_int < 0.15), err_int
    assert np.all(app > true)                      # layering adds apparent attenuation
    assert np.mean(err_app) > 2.0 * np.mean(err_int)


def test_spectral_ratio_band_check():
    with pytest.raises(ValueError):
        A.spectral_ratio(np.arange(10.0), np.ones((2, 10)), (100.0, 200.0))
