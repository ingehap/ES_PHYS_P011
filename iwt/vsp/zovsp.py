"""Zero-offset VSP processing: picking, time-depth, interval velocity, wavefield
separation, deconvolution and the corridor stack.

The steps follow standard ZO-VSP practice (Hardage, 1985; Hinds et al., 1996):

1. pick the first arrival and convert to vertical one-way time;
2. derive interval velocities from the time-depth pairs (here by regularized
   least squares, because plain differencing amplifies picking noise,
   Lizarralde & Swift, 1999);
3. separate the downgoing and upgoing wavefields, by a least-squares fit of
   up- and downgoing plane waves in a sliding window of traces (Seeman &
   Horowicz, 1983) or by the classic median filter across depth (Kommedal &
   Tjøstheim, 1989);
4. deconvolve the upgoing wavefield by the downgoing one, which removes the
   source signature, the downgoing multiples and the attenuation above each
   receiver, and shift it to two-way time;
5. stack a corridor just after the first arrival, where the upgoing wavefield
   is primaries from just below the receiver.

**The time convention matters.**  In an attenuating earth a pulse broadens as it
travels, so the onset of the first arrival travels at the high-frequency phase
velocity, its peak somewhat slower, and no pick of the pulse shape equals the
traveltime at the seismic frequency (Molyneux & Schmitt, 1999).
:func:`phase_delay_times` therefore also measures the traveltime as the phase
delay at a chosen frequency, which is the time a well tie at that frequency
needs.

References
----------
Hardage, B. A., 1985, Vertical Seismic Profiling, Part A: Principles, 2nd ed.:
Geophysical Press, London.
Hinds, R. C., Anderson, N. L., & Kuzmiski, R. D., 1996, VSP Interpretive
Processing: Theory and Practice: SEG Open File Publications 3.
https://doi.org/10.1190/1.9781560801894
Kommedal, J. H., & Tjøstheim, B. A., 1989, A study of different methods of
wavefield separation for application to VSP data: Geophysical Prospecting,
37(2), 117-142. https://doi.org/10.1111/j.1365-2478.1989.tb01826.x
Lizarralde, D., & Swift, S., 1999, Smooth inversion of VSP traveltime data:
Geophysics, 64(3), 659-661. https://doi.org/10.1190/1.1444574
Molyneux, J. B., & Schmitt, D. R., 1999, First-break timing: Arrival onset times
by direct correlation: Geophysics, 64(5), 1492-1501.
https://doi.org/10.1190/1.1444653
Seeman, B., & Horowicz, L., 1983, Vertical seismic profiling: Separation of
upgoing and downgoing acoustic waves in a stratified medium: Geophysics, 48(5),
555-568. https://doi.org/10.1190/1.1441486
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Sequence

import numpy as np

from ..model import ElasticModel1D

__all__ = [
    "VSP",
    "synthetic_zovsp",
    "add_noise",
    "shift_traces",
    "pick_first_breaks",
    "vertical_times",
    "phase_delay_times",
    "IntervalVelocity",
    "interval_velocity",
    "separate_wavefields",
    "deconvolve",
    "corridor_stack",
]


@dataclass
class VSP:
    """A zero-offset VSP: ``data[receiver, time]`` at receiver ``depths`` (m TVDSS).

    ``source_depth`` (m below sea level) and ``source_offset`` (m from the
    wellhead) describe the source position.
    """

    t: np.ndarray
    depths: np.ndarray
    data: np.ndarray
    source_depth: float = 0.0
    source_offset: float = 0.0

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def with_data(self, data) -> "VSP":
        """A copy with new trace data (same geometry)."""
        return replace(self, data=np.asarray(data, dtype=float))

    @classmethod
    def from_record(cls, rec, source_depth: float, offset_index: int = 0) -> "VSP":
        """The VSP at one source offset of a :class:`~iwt.modeling.reflectivity.Record`."""
        return cls(rec.t, np.asarray(rec.depths, float), rec.data[:, offset_index, :],
                   source_depth, float(rec.offsets[offset_index]))


# -- synthetic data -------------------------------------------------------------


def synthetic_zovsp(model: ElasticModel1D, depths, source_depth: float, dt: float, nt: int,
                    wavelet, f_max: Optional[float] = None, component: str = "uz",
                    source: str = "explosion", attenuation: bool = True,
                    free_surface: bool = True, internal_multiples: bool = True,
                    transmission_loss: bool = True, workers: int = 1) -> VSP:
    """A vertical-incidence (plane-wave, p = 0) ZO-VSP from the reflectivity method.

    A fast stand-in for a point-source ZO-VSP: no geometric spreading and no
    offset, otherwise the full 1D response (free-surface and internal multiples,
    Q, transmission losses).  ``component`` is ``"uz"``, ``"uz_down"`` or
    ``"uz_up"`` (exact wavefield separation).  ``wavelet`` is ``(t, w)``, the
    displacement waveform of the downgoing wave leaving the source (with
    :meth:`~iwt.modeling.reflectivity.Reflectivity.record` it is the source
    *potential*, whose displacement is its time derivative).  With
    ``source="explosion"`` the source ghost is included.
    """
    from ..modeling.reflectivity import (
        Reflectivity,
        spectrum_to_trace,
        time_axis_spectrum,
        wavelet_spectrum,
    )

    depths = np.atleast_1d(np.asarray(depths, dtype=float))
    t_shift = max(0.0, -float(wavelet[0][0]))
    n_pad = int(np.ceil(t_shift / dt))
    _, om, eps, _ = time_axis_spectrum(dt, nt + n_pad, f_max)
    refl = Reflectivity(model, attenuation=attenuation, free_surface=free_surface,
                        internal_multiples=internal_multiples,
                        transmission_loss=transmission_loss)
    spec = refl.plane_wave(om, [0.0], source_depth, depths, source=source,
                           workers=workers)[component][:, :, 0]
    W = wavelet_spectrum(wavelet[0], wavelet[1], om)
    data = spectrum_to_trace(spec * W[None, :], dt, nt + n_pad, eps, om, n_pad * dt)[:, :nt]
    return VSP(np.arange(nt) * dt, depths, data, source_depth, 0.0)


def add_noise(data, level: float, dt: float, f_max: Optional[float] = None,
              seed: Optional[int] = 0) -> np.ndarray:
    """``data`` plus Gaussian noise of standard deviation ``level`` times each trace's peak.

    The noise is white up to ``f_max`` (Hz), with a cosine roll-off over the last
    20 % of the band, so that it occupies the same band as the signal.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    n = data.shape[-1]
    noise = rng.standard_normal(data.shape)
    if f_max is not None:
        f = np.fft.rfftfreq(n, dt)
        x = np.clip((f - 0.8 * f_max) / (0.2 * f_max), 0.0, 1.0)
        noise = np.fft.irfft(np.fft.rfft(noise, axis=-1) * 0.5 * (1 + np.cos(np.pi * x)),
                             n=n, axis=-1)
        noise /= np.std(noise, axis=-1, keepdims=True)
    peak = np.max(np.abs(data), axis=-1, keepdims=True)
    return data + level * peak * noise


# -- trace shifts ---------------------------------------------------------------


def _n_fft(nt: int) -> int:
    return int(2 ** np.ceil(np.log2(2 * nt)))


def shift_traces(data, shifts, dt: float, n_out: Optional[int] = None) -> np.ndarray:
    """Delay each trace by ``shifts[k]`` seconds (negative: advance), by FFT.

    Fractional-sample shifts are exact for band-limited data.  The traces are
    zero-padded to at least twice their length, so shifted energy does not wrap
    around within the returned ``n_out`` samples (default: the input length;
    longer keeps what a delay moves past the end of the input).
    """
    data = np.asarray(data, dtype=float)
    nt = data.shape[-1]
    n_out = nt if n_out is None else int(n_out)
    n = _n_fft(max(nt, n_out))
    f = np.fft.rfftfreq(n, dt)
    X = np.fft.rfft(data, n=n, axis=-1)
    X *= np.exp(-2j * np.pi * f[None, :] * np.asarray(shifts, float)[:, None])
    return np.fft.irfft(X, n=n, axis=-1)[:, :n_out]


# -- picking and time-depth -----------------------------------------------------


def pick_first_breaks(t, data, method: str = "peak", guide=None, threshold: float = 0.1,
                      before: float = 0.01, after: float = 0.03) -> np.ndarray:
    """First-arrival times (s) of each trace.

    ``method``:

    * ``"onset"``: the first time the envelope (Hilbert transform) reaches
      ``threshold`` times the first arrival's envelope peak, with linear
      interpolation.  It sits on the leading edge, which interference with
      later (upgoing) waves barely touches, so it is the most robust pick;
    * ``"peak"``: the main peak of the first arrival (largest absolute
      amplitude), with parabolic interpolation, the usual pick for a zero-phase
      source.  Where upgoing reflections interfere, where attenuation makes a
      side lobe grow, or where the source ghost makes two lobes nearly equal,
      the largest lobe can change from trace to trace.  The lobe is therefore
      chosen once, on the trace whose onset-to-peak lag is the median one, and
      followed from trace to trace along the onset moveout (the pick is the
      local maximum of the absolute amplitude nearest to the previous pick
      plus the onset difference).

    The first arrival is searched for in ``[guide - before, guide + after]``.
    Without ``guide``, the guide is the first time each trace's envelope exceeds
    ``threshold`` times its maximum.
    """
    from scipy.signal import hilbert

    t = np.asarray(t, dtype=float)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    dt = t[1] - t[0]
    n = data.shape[0]
    # zero-padded, so that energy at the end of the trace does not wrap to t = 0
    env = np.abs(hilbert(data, N=_n_fft(t.size), axis=-1))[:, : t.size]
    if guide is None:
        guide = np.array([t[np.argmax(e >= threshold * e.max())] for e in env])
    guide = np.broadcast_to(np.asarray(guide, dtype=float), (n,))

    def index(time):
        return int(np.clip(np.round((time - t[0]) / dt), 1, t.size - 2))

    onset = np.empty(n)
    for k in range(n):
        i0, i1 = index(guide[k] - before), index(guide[k] + after) + 1
        e = env[k]
        i = i0 + int(np.argmax(e[i0:i1]))
        level = threshold * e[i]
        while i > 0 and e[i - 1] >= level:
            i -= 1
        onset[k] = t[0] if i == 0 else t[i - 1] + (level - e[i - 1]) / (e[i] - e[i - 1]) * dt
    if method == "onset":
        return onset
    if method != "peak":
        raise ValueError("method must be 'peak' or 'onset'")
    a = np.abs(data)
    raw = np.array([index(onset[k]) + int(np.argmax(a[k, index(onset[k]):index(onset[k] + after) + 1]))
                    for k in range(n)])
    lag = t[raw] - onset
    k0 = int(np.argmin(np.abs(lag - np.median(lag))))

    def refine(k, g):
        """Local maximum of |trace k| nearest to time g, parabolic interpolation."""
        i0, i1 = index(g - 0.5 * after), index(g + 0.5 * after) + 1
        w = a[k, i0:i1]
        is_max = (w >= a[k, i0 - 1:i1 - 1]) & (w >= a[k, i0 + 1:i1 + 1])
        cand = i0 + np.flatnonzero(is_max)
        i = int(cand[np.argmin(np.abs(t[cand] - g))]) if cand.size else index(g)
        y0, y1, y2 = a[k, i - 1], a[k, i], a[k, i + 1]
        den = y0 - 2.0 * y1 + y2
        return t[i] + (0.5 * (y0 - y2) / den * dt if den < 0 else 0.0)

    picks = np.empty(n)
    picks[k0] = refine(k0, t[raw[k0]])
    for k in range(k0 + 1, n):
        picks[k] = refine(k, picks[k - 1] + onset[k] - onset[k - 1])
    for k in range(k0 - 1, -1, -1):
        picks[k] = refine(k, picks[k + 1] + onset[k] - onset[k + 1])
    return picks


def vertical_times(times, depths, source_depth: float = 0.0, source_offset: float = 0.0,
                   datum_velocity: Optional[float] = None) -> np.ndarray:
    """Straight-ray conversion of first-arrival times to vertical one-way times.

    ``t_v = t (z - z_s) / sqrt((z - z_s)^2 + x^2)`` for a source at depth
    ``z_s`` and offset ``x``.  With ``datum_velocity`` (the sea-water velocity),
    the time from sea level down to the source depth is added, so the result is
    the vertical time below sea level.  The straight-ray assumption is good for
    the small offsets of a ZO-VSP (ray bending changes the time by much less than
    the cosine correction itself).
    """
    times = np.asarray(times, dtype=float)
    dz = np.asarray(depths, dtype=float) - source_depth
    tv = times * dz / np.hypot(dz, source_offset)
    if datum_velocity is not None:
        tv = tv + source_depth / datum_velocity
    return tv


def phase_delay_times(t, data, frequency: float, picks, before: float = 0.05,
                      after: float = 0.08, taper: float = 0.01,
                      reference: int = 0) -> np.ndarray:
    """First-arrival traveltimes measured as phase delays at ``frequency`` (Hz).

    Each trace is windowed around its pick (``picks - before`` to ``picks +
    after``, with cosine tapers of length ``taper``) and its Fourier component
    at ``frequency`` is computed.  The phase difference to the ``reference``
    trace gives the traveltime difference at that frequency,

        tau_k = tau_ref - (phi_k - phi_ref) / (2 pi f),

    in numpy's ``exp(-i 2 pi f t)`` convention, with the 2 pi ambiguity resolved
    by the picks.  The absolute level is that of the reference pick
    (``tau_ref = picks[reference]``), so only differences carry information
    about the earth; the source phase (signature and ghost) cancels.

    In a constant-Q medium the phase delay at f over a path is exactly
    ``Int dz / c(f)``, with ``c(f)`` the phase velocity at f
    (:func:`iwt.qmodel.phase_velocity`).

    Measure it on the separated downgoing wavefield
    (:func:`separate_wavefields`): at seismic frequencies the window also holds
    upgoing reflections, which on the synthetic Wisting-like well scatter the
    phase times on the raw traces by about 0.5 ms from trace to trace and bias
    the reference trace by 1 ms.
    """
    t = np.asarray(t, dtype=float)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    picks = np.asarray(picks, dtype=float)
    w = 2.0 * np.pi * frequency
    resid = np.empty(data.shape[0])
    for k in range(data.shape[0]):
        win = _window(t, picks[k] - before, picks[k] + after, taper)
        X = np.sum(win * data[k] * np.exp(-1j * w * (t - picks[k])))
        resid[k] = np.angle(X)
    dphi = np.angle(np.exp(1j * (resid - resid[reference])))
    return picks - dphi / w


def _window(t, t0: float, t1: float, taper: float) -> np.ndarray:
    """Box window on [t0, t1] with cosine tapers of length ``taper`` inside it."""
    w = np.zeros_like(t)
    inside = (t >= t0) & (t <= t1)
    w[inside] = 1.0
    if taper > 0:
        a = (t - t0) / taper
        b = (t1 - t) / taper
        ramp_a = inside & (a < 1.0)
        ramp_b = inside & (b < 1.0)
        w[ramp_a] *= 0.5 * (1.0 - np.cos(np.pi * a[ramp_a]))
        w[ramp_b] *= 0.5 * (1.0 - np.cos(np.pi * b[ramp_b]))
    return w


# -- interval velocity ----------------------------------------------------------


class IntervalVelocity(NamedTuple):
    """Interval velocities on ``[top, bottom]`` intervals and the fitted times."""

    top: np.ndarray
    bottom: np.ndarray
    velocity: np.ndarray
    time_fit: np.ndarray
    smoothing: float
    sigma: float


def _time_operator(depths, edges):
    """Matrix G with ``G @ s`` the time from depths[0] to each depth for slowness s on edges."""
    depths = np.asarray(depths, float)
    lo, hi = edges[:-1], edges[1:]
    a = np.clip(depths[:, None], lo[None, :], hi[None, :])
    b = np.clip(depths[0], lo, hi)[None, :]
    return a - b


def interval_velocity(depths, times, boundaries: Optional[Sequence[float]] = None,
                      smoothing: Optional[float] = None,
                      sigma: Optional[float] = None) -> IntervalVelocity:
    """Interval velocities from vertical time-depth pairs.

    With ``boundaries`` the velocity is constant between them (blocked least
    squares; the blocks span the receiver range).  The blocks must be nearly
    homogeneous: a velocity gradient inside a thick block cannot be fitted, and
    the least squares then moves the misfit into the neighbouring (thin)
    blocks; split such units.  Otherwise one velocity per
    receiver interval is found by Tikhonov-regularized least squares,

        min |t0 + G s - t|^2 / sigma^2 + smoothing^2 |D (s / s_mean)|^2,

    with D the first difference.  If ``smoothing`` is not given it is chosen by
    the discrepancy principle (residual RMS = ``sigma``), and if ``sigma`` is
    not given either, it is estimated from the second differences of the times
    (robust MAD; valid when the receivers are dense enough that the true times
    are nearly linear over three levels).  Plain differencing is the case
    ``smoothing = 0``.
    """
    z = np.asarray(depths, dtype=float)
    t = np.asarray(times, dtype=float)
    order = np.argsort(z)
    z, t = z[order], t[order]
    if sigma is None:
        d2 = t[2:] - 2.0 * t[1:-1] + t[:-2]
        sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)
        sigma = max(float(sigma), 1e-9)
    if boundaries is not None:
        edges = np.unique(np.concatenate([[z[0]], np.asarray(boundaries, float), [z[-1]]]))
        edges = edges[(edges >= z[0]) & (edges <= z[-1])]
        G = np.column_stack([np.ones(z.size), _time_operator(z, edges)])
        x, *_ = np.linalg.lstsq(G, t, rcond=None)
        return IntervalVelocity(edges[:-1], edges[1:], 1.0 / x[1:], G @ x, 0.0, float(sigma))

    edges = z
    G = np.column_stack([np.ones(z.size), _time_operator(z, edges)])
    n = edges.size - 1
    s_mean = (t[-1] - t[0]) / (z[-1] - z[0])
    D = np.zeros((n - 1, n + 1))
    i = np.arange(n - 1)
    D[i, i + 1], D[i, i + 2] = -1.0 / s_mean, 1.0 / s_mean

    def solve(lam):
        A = np.vstack([G / sigma, lam * D])
        b = np.concatenate([t / sigma, np.zeros(n - 1)])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x

    def misfit(lam):
        x = solve(lam)
        return np.sqrt(np.mean((G @ x - t) ** 2)) - sigma

    if smoothing is None:
        lo, hi = 1e-6, 1e6
        if misfit(hi) < 0:
            smoothing = hi
        elif misfit(lo) > 0:
            smoothing = lo
        else:
            for _ in range(60):     # bisection on log(lambda); the misfit grows with lambda
                mid = np.sqrt(lo * hi)
                if misfit(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            smoothing = np.sqrt(lo * hi)
    x = solve(smoothing)
    return IntervalVelocity(edges[:-1], edges[1:], 1.0 / x[1:], G @ x, float(smoothing),
                            float(sigma))


# -- wavefield separation, deconvolution, corridor stack ------------------------


def separate_wavefields(t, data, picks, length: int = 5, method: str = "parametric",
                        damping: float = 0.01):
    """Downgoing and upgoing wavefields of a ZO-VSP.  Returns ``(down, up)``.

    ``picks`` are the first-arrival times, which give the moveout of the
    downgoing waves (``+picks``) and of the upgoing ones (``-picks``).

    * ``method="parametric"`` (default): in a window of ``length`` neighbouring
      traces, at each frequency, the data are fitted by least squares with one
      downgoing and one upgoing plane wave of known moveout (Seeman & Horowicz,
      1983); ``damping`` (a fraction of ``length``) stabilizes the lowest
      frequencies, where the two moveouts cannot be told apart over the window.
      ``up`` is the fitted upgoing wave, so ``data - down - up`` is the misfit.
    * ``method="median"`` or ``"mean"``: the traces are aligned on the first
      arrivals, filtered across depth with a running median (or mean) of
      ``length`` traces, which keeps the flat downgoing events and rejects the
      upgoing ones (dipping at twice the moveout), and shifted back;
      ``up = data - down``.  The median passes about ``1.25 / sqrt(length)`` of a
      dense, noise-like upgoing wavefield (a finely layered earth), so it needs
      long filters, and the first and last ``length // 2`` traces are poorly
      separated.

    On the synthetic Wisting-like well the parametric method has half the error
    of a 9-trace median filter (see the README).
    """
    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)
    dt = t[1] - t[0]
    picks = np.asarray(picks, dtype=float)
    nt = data.shape[-1]
    if method == "parametric":
        return _parametric_separation(data, picks, dt, length, damping)
    from scipy.ndimage import median_filter, uniform_filter1d

    shifts = picks - picks.min()
    flat = shift_traces(data, -shifts, dt, nt + int(np.ceil(shifts.max() / dt)) + 1)
    if method == "median":
        down_flat = median_filter(flat, size=(length, 1), mode="mirror")
    elif method == "mean":
        down_flat = uniform_filter1d(flat, size=length, axis=0, mode="mirror")
    else:
        raise ValueError("method must be 'parametric', 'median' or 'mean'")
    down = shift_traces(down_flat, shifts, dt)[:, :nt]
    return down, data - down


def _parametric_separation(data, picks, dt: float, length: int, damping: float):
    """Least-squares up/down plane-wave fit per frequency in a sliding window."""
    ntr, nt = data.shape
    length = min(length, ntr)
    n = _n_fft(nt)
    X = np.fft.rfft(data, n=n, axis=-1)
    w = 2.0 * np.pi * np.fft.rfftfreq(n, dt)
    D = np.empty_like(X)
    U = np.empty_like(X)
    a = length * (1.0 + damping)
    for k in range(ntr):
        j0 = min(max(k - length // 2, 0), ntr - length)
        tau = picks[j0:j0 + length] - picks[k]
        # downgoing column exp(-i w tau), upgoing exp(+i w tau); e = conj(downgoing)
        e = np.exp(1j * w[None, :] * tau[:, None])
        b = np.sum(e * e, axis=0)
        r_d = np.sum(e * X[j0:j0 + length], axis=0)
        r_u = np.sum(np.conj(e) * X[j0:j0 + length], axis=0)
        det = a * a - np.abs(b) ** 2
        D[k] = (a * r_d - b * r_u) / det
        U[k] = (a * r_u - np.conj(b) * r_d) / det
    return np.fft.irfft(D, n=n, axis=-1)[:, :nt], np.fft.irfft(U, n=n, axis=-1)[:, :nt]


def deconvolve(t, up, down, times, wavelet, white_noise: float = 0.01,
               design: Optional[float] = None, picks=None,
               polarity: float = -1.0) -> np.ndarray:
    """Upgoing wavefield deconvolved by the downgoing one, in two-way time.

    For each trace, ``U D* / (|D|^2 + white_noise * max|D|^2)`` is convolved with
    the zero-phase target ``wavelet`` ``(t, w)`` (peak 1, so a reflection
    coefficient r gives a peak of about r), which removes the source signature,
    the downgoing multiples and the attenuation above the receiver.  The result
    is shifted by ``2 * times`` (the one-way times of the receivers) into
    two-way time on the same time axis.  With ``design`` (s), the downgoing
    wavefield is used only up to ``picks + design`` (cosine-tapered).

    On vertical displacement (or velocity) the upgoing/downgoing ratio of a
    reflection is minus its pressure reflection coefficient, so the default
    ``polarity=-1`` gives SEG normal polarity (an impedance increase is a
    positive peak).  Use ``polarity=1`` for hydrophone data.

    The two-way times inherit the convention of ``times``: use first-arrival
    traveltimes at the seismic frequency (:func:`phase_delay_times`) for a well
    tie at that frequency.
    """
    t = np.asarray(t, dtype=float)
    up = np.asarray(up, dtype=float)
    down = np.asarray(down, dtype=float)
    dt = t[1] - t[0]
    nt = t.size
    times = np.asarray(times, dtype=float)
    if design is not None:
        if picks is None:
            raise ValueError("design requires picks")
        win = np.array([_window(t, -np.inf, p + design, min(0.02, design / 4)) for p in picks])
        down = down * win
    n = _n_fft(nt)
    U = np.fft.rfft(up, n=n, axis=-1)
    D = np.fft.rfft(down, n=n, axis=-1)
    tw, w = np.asarray(wavelet[0], float), np.asarray(wavelet[1], float)
    wc = np.zeros(n)
    wc[np.round(tw / dt).astype(int) % n] = w
    Wt = np.fft.rfft(wc)
    p2 = np.abs(D) ** 2
    H = np.conj(D) / (p2 + white_noise * p2.max(axis=-1, keepdims=True))
    f = np.fft.rfftfreq(n, dt)
    X = polarity * U * H * Wt[None, :] * np.exp(-2j * np.pi * f[None, :] * 2.0 * times[:, None])
    return np.fft.irfft(X, n=n, axis=-1)[:, :nt]


def corridor_stack(t, decon_twt, times, length: Optional[float] = 0.1, start: float = 0.0,
                   taper: float = 0.01):
    """Corridor stack of the deconvolved upgoing wavefield in two-way time.

    Trace k contributes on ``[2 times_k + start, 2 times_k + start + length]``
    (cosine tapers of length ``taper``), where it holds the primaries from just
    below the receiver and few interbed multiples.  ``length=None`` stacks
    everything after ``2 times_k + start`` (the "full" stack, which includes the
    interbed multiples generated below each receiver).  Returns ``(stack,
    fold)``: the stack is normalized by the (taper-weighted) fold.
    """
    t = np.asarray(t, dtype=float)
    d = np.asarray(decon_twt, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = np.empty_like(d)
    for k, tk in enumerate(times):
        t0 = 2.0 * tk + start
        t1 = np.inf if length is None else t0 + length
        mask[k] = _window(t, t0, t1, taper)
    fold = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        stack = np.where(fold > 1e-6, (mask * d).sum(axis=0) / np.maximum(fold, 1e-12), 0.0)
    return stack, fold
