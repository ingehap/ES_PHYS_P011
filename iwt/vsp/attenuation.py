"""Attenuation from the ZO-VSP first arrival: spectral ratio, centroid shift, interval Q.

The amplitude spectrum of the direct downgoing wave at receiver k is modelled as

    A_k(f) = G_k S(f) exp(-pi f t*_k),      t*_k = Int dt / Q,

with ``G_k`` frequency independent (spreading, transmission), ``S`` the source
spectrum and ``t*`` the attenuated traveltime.  Relative to a reference receiver,

* the **spectral ratio** ``ln(A_k / A_ref)`` is a straight line in f of slope
  ``-pi (t*_k - t*_ref)`` (Hauge, 1981);
* the **centroid frequency** decreases with t*.  Quan & Harris (1997) give the
  closed form ``Delta t* = (f_ref - f_k) / (pi sigma_ref^2)`` for a Gaussian
  spectrum; :func:`centroid_shift` also inverts the centroid exactly for the
  measured reference spectrum, which removes the Gaussian assumption.

Interval Q follows from the t* differences and the first-arrival times by
(non-negative) least squares over depth blocks (:func:`interval_q`).

**Apparent, stratigraphic and intrinsic Q.**  In a finely layered earth the
transmitted pulse also loses high frequencies to short-period multiples
(O'Doherty & Anstey, 1971; Richards & Menke, 1983), so the VSP measures an
apparent attenuation 1/Q_app = 1/Q_int + 1/Q_strat.  The stratigraphic part can
be computed from the log-based elastic model: a synthetic VSP *without* Q,
processed with the same windows and estimator, gives t*_strat, and
t*_int = t*_app - t*_strat.  Only Q_int is a material property that also
predicts velocity dispersion (Kjartansson, 1979); the velocity effect of the
layering is the Backus (long-wavelength) average instead
(:mod:`iwt.calibration`).

References
----------
Hauge, P. S., 1981, Measurements of attenuation from vertical seismic profiles:
Geophysics, 46(11), 1548-1558. https://doi.org/10.1190/1.1441161
Quan, Y., & Harris, J. M., 1997, Seismic attenuation tomography using the
frequency shift method: Geophysics, 62(3), 895-905.
https://doi.org/10.1190/1.1444197
O'Doherty, R. F., & Anstey, N. A., 1971, Reflections on amplitudes: Geophysical
Prospecting, 19(3), 430-458. https://doi.org/10.1111/j.1365-2478.1971.tb00610.x
Richards, P. G., & Menke, W., 1983, The apparent attenuation of a scattering
medium: BSSA, 73(4), 1005-1021. https://doi.org/10.1785/BSSA0730041005
Kjartansson, E., 1979, Constant Q-wave propagation and attenuation: JGR, 84(B9),
4737-4748. https://doi.org/10.1029/JB084iB09p04737
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .zovsp import _window

__all__ = [
    "first_arrival_spectra",
    "spectral_ratio",
    "centroid_frequency",
    "centroid_shift",
    "IntervalQ",
    "interval_q",
]


def first_arrival_spectra(t, data, picks, before: float = 0.05, after: float = 0.1,
                          taper: float = 0.01, n_fft: Optional[int] = None):
    """Amplitude spectra of the first arrival, windowed on ``[picks - before, picks + after]``.

    Returns ``(f, A)`` with ``A[receiver, frequency]``.  The same window length
    is used at every receiver.  Apply it to the downgoing wavefield
    (:func:`~iwt.vsp.zovsp.separate_wavefields`) so that upgoing reflections
    crossing the window do not bias the spectra.  The window must hold the whole
    pulse, side lobes included, at every receiver: a taper that cuts into the
    pulse leaks energy into the weak high frequencies and biases t* (with a
    40 Hz Ricker, starting the taper 20 ms before the peak biases t* by up to
    6 %, starting it 50 ms before by less than 1 %).
    """
    t = np.asarray(t, dtype=float)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    dt = t[1] - t[0]
    n = n_fft or int(2 ** np.ceil(np.log2(4 * t.size)))
    win = np.array([_window(t, p - before, p + after, taper) for p in picks])
    return np.fft.rfftfreq(n, dt), np.abs(np.fft.rfft(win * data, n=n, axis=-1)) * dt


def _band(f, band: Tuple[float, float]):
    sel = (f >= band[0]) & (f <= band[1])
    if sel.sum() < 3:
        raise ValueError("fewer than three frequencies in the band")
    return sel


def spectral_ratio(f, spectra, band: Tuple[float, float], reference: int = 0):
    """t* relative to the ``reference`` receiver by the spectral-ratio method.

    Fits ``ln(A_k / A_ref) = a_k - pi f dt*_k`` by least squares over ``band``
    (Hz).  Returns ``(dt_star, intercept, rms)``: the t* differences (s), the
    intercepts ``a_k`` and the RMS misfit of each fit (natural-log units).
    """
    f = np.asarray(f, dtype=float)
    A = np.atleast_2d(np.asarray(spectra, dtype=float))
    sel = _band(f, band)
    with np.errstate(divide="ignore"):
        y = np.log(A[:, sel]) - np.log(A[reference, sel])[None, :]
    G = np.column_stack([np.ones(sel.sum()), -np.pi * f[sel]])
    x, *_ = np.linalg.lstsq(G, y.T, rcond=None)
    rms = np.sqrt(np.mean((G @ x - y.T) ** 2, axis=0))
    return x[1], x[0], rms


def centroid_frequency(f, spectra, band: Tuple[float, float]):
    """Centroid ``f_c`` and variance ``sigma^2`` of each amplitude spectrum within ``band``."""
    f = np.asarray(f, dtype=float)
    A = np.atleast_2d(np.asarray(spectra, dtype=float))
    sel = _band(f, band)
    fs, a = f[sel], A[:, sel]
    w = a.sum(axis=1)
    fc = (a * fs).sum(axis=1) / w
    var = (a * (fs[None, :] - fc[:, None]) ** 2).sum(axis=1) / w
    return fc, var


def centroid_shift(f, spectra, band: Tuple[float, float], reference: int = 0,
                   method: str = "exact"):
    """t* relative to the ``reference`` receiver from the centroid-frequency shift.

    ``method="gaussian"``: ``dt* = (f_ref - f_k) / (pi sigma_ref^2)`` (Quan &
    Harris, 1997), exact only for a Gaussian reference spectrum.
    ``method="exact"``: the dt* for which ``A_ref(f) exp(-pi f dt*)`` has the
    observed centroid within ``band`` (one-dimensional root finding; the
    centroid decreases monotonically with dt*).  NaN where no such dt* exists.
    """
    from scipy.optimize import brentq

    f = np.asarray(f, dtype=float)
    A = np.atleast_2d(np.asarray(spectra, dtype=float))
    fc, var = centroid_frequency(f, A, band)
    if method == "gaussian":
        return (fc[reference] - fc) / (np.pi * var[reference])
    if method != "exact":
        raise ValueError("method must be 'exact' or 'gaussian'")
    sel = _band(f, band)
    fs, a_ref = f[sel], A[reference, sel]
    x = fs - fs.mean()

    def centroid(ts):
        e = a_ref * np.exp(-np.pi * x * ts)
        return (e * fs).sum() / e.sum()

    bound = 50.0 / (np.pi * (fs[-1] - fs[0]))       # exp(+-50) across the band
    lo_c, hi_c = centroid(bound), centroid(-bound)
    out = np.full(A.shape[0], np.nan)
    for k, target in enumerate(fc):
        if lo_c < target < hi_c:
            out[k] = brentq(lambda ts: centroid(ts) - target, -bound, bound, xtol=1e-9)
    return out


class IntervalQ(NamedTuple):
    """Interval attenuation on ``[top, bottom]`` blocks."""

    top: np.ndarray
    bottom: np.ndarray
    inv_q: np.ndarray
    inv_q_std: np.ndarray
    residual_rms: float

    @property
    def q(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(self.inv_q > 0, 1.0 / np.where(self.inv_q > 0, self.inv_q, 1.0),
                            np.inf)

    def at(self, depth, fill: float = np.inf) -> np.ndarray:
        """Q at ``depth`` (``fill`` outside the blocks)."""
        z = np.asarray(depth, dtype=float)
        out = np.full(z.shape, fill, dtype=float)
        q = self.q
        for j in range(self.top.size):
            last = j == self.top.size - 1
            sel = (z >= self.top[j]) & ((z <= self.bottom[j]) if last else (z < self.bottom[j]))
            out[sel] = q[j]
        return out


def interval_q(depths, times, dt_star, boundaries: Optional[Sequence[float]] = None,
               reference: int = 0, nonnegative: bool = True,
               intercept: bool = True) -> IntervalQ:
    """Interval 1/Q from t* differences and one-way times at the receivers.

    Model: ``dt*_k = c + Sum_j dT_jk / Q_j``, with ``dT_jk`` the one-way time
    spent in block j between the ``reference`` receiver and receiver k (from the
    piecewise-linear time-depth curve through ``(depths, times)``).  Blocks are
    bounded by ``boundaries`` and the receiver range.  The constant ``c``
    (``intercept=True``) absorbs the error of the reference spectrum, which
    otherwise biases the first block: the reference receiver is at the end of
    the array, where wavefield separation and noise are worst.  ``nonnegative``
    enforces 1/Q >= 0 (bounded least squares); the standard deviations are those
    of the unconstrained solution, from the fit residual.
    """
    from scipy.optimize import lsq_linear

    z = np.asarray(depths, dtype=float)
    tt = np.asarray(times, dtype=float)
    y = np.asarray(dt_star, dtype=float)
    ok = np.isfinite(y)
    order = np.argsort(z)
    zs, ts = z[order], tt[order]
    edges = np.unique(np.concatenate([[zs[0]], np.asarray(boundaries if boundaries is not None
                                                          else [], float), [zs[-1]]]))
    edges = edges[(edges >= zs[0]) & (edges <= zs[-1])]

    def T(d):
        return np.interp(d, zs, ts)

    lo, hi = edges[:-1], edges[1:]
    zk = np.clip(z[:, None], lo[None, :], hi[None, :])
    zr = np.clip(z[reference], lo, hi)[None, :]
    G = T(zk) - T(zr)
    if intercept:
        G = np.column_stack([G, np.ones(z.size)])
    Gk, yk = G[ok], y[ok]
    if nonnegative:
        lower = np.zeros(G.shape[1])
        if intercept:
            lower[-1] = -np.inf
        x = lsq_linear(Gk, yk, bounds=(lower, np.inf)).x
    else:
        x, *_ = np.linalg.lstsq(Gk, yk, rcond=None)
    r = Gk @ x - yk
    dof = max(yk.size - x.size, 1)
    s2 = float(r @ r) / dof
    std = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(Gk.T @ Gk)), 0.0, None))
    nb = lo.size
    return IntervalQ(lo, hi, x[:nb], std[:nb], float(np.sqrt(np.mean(r ** 2))))
