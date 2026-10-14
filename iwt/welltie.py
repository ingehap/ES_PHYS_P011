"""Well tie: wavelet estimation and tie metrics.

:func:`estimate_wavelet` finds the wavelet that, convolved with a reflectivity
series (or with a synthetic made with a unit spike), best predicts the seismic
trace(s), by damped least squares over a centred window of lags.  It is the
time-domain form of the matched-filter (Wiener) estimate; Walden & White (1998)
give the frequency-domain treatment of the same noisy input-output problem.
Several traces (angles, offsets, nearby traces) can be used together.

:func:`tie_metrics` reports what a tie is judged by (White & Simm, 2003): the
correlation at zero lag and its maximum over lags (the time shift, refined to a
fraction of a sample), the constant phase rotation that maximizes the
correlation, the least-squares amplitude scale, and the proportion of the trace
energy the scaled synthetic predicts (PEP).

References
----------
Walden, A. T., & White, R. E., 1998, Seismic wavelet estimation: A frequency
domain solution to a geophysical noisy input-output problem: IEEE Transactions on
Geoscience and Remote Sensing, 36(1), 287-297. https://doi.org/10.1109/36.655337
White, R. E., & Simm, R., 2003, Tutorial: Good practice in well ties: First Break,
21(10). https://doi.org/10.3997/1365-2397.21.10.25640
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["estimate_wavelet", "TieMetrics", "tie_metrics", "rotate_phase"]


def estimate_wavelet(trace, reflectivity, dt: float, half_length: float,
                     damping: float = 1e-3):
    """Least-squares wavelet ``(t, w)`` on lags ``-half_length .. +half_length`` (s).

    ``trace`` and ``reflectivity`` are ``[n_traces, nt]`` (or 1D) on the same
    time axis; the wavelet minimizes ``|trace - reflectivity * w|^2 +
    damping * |R|^2 |w|^2`` summed over traces, with ``|R|^2`` the reflectivity
    energy (so the damping is relative).
    """
    d = np.atleast_2d(np.asarray(trace, dtype=float))
    r = np.atleast_2d(np.asarray(reflectivity, dtype=float))
    L = int(round(half_length / dt))
    lags = np.arange(-L, L + 1)
    nt = d.shape[1]
    rows = []
    for k in lags:                           # column k: reflectivity shifted by k samples
        sh = np.zeros_like(r)
        if k >= 0:
            sh[:, k:] = r[:, :nt - k]
        else:
            sh[:, :k] = r[:, -k:]
        rows.append(sh.ravel())
    G = np.array(rows).T
    A = G.T @ G
    A = A + damping * np.trace(A) / A.shape[0] * np.eye(A.shape[0])
    w = np.linalg.solve(A, G.T @ d.ravel())
    return lags * dt, w


def rotate_phase(trace, phi: float):
    """Constant phase rotation by ``phi`` (rad) via the analytic signal."""
    from scipy.signal import hilbert

    a = hilbert(np.asarray(trace, dtype=float))
    return np.real(a * np.exp(-1j * phi))


class TieMetrics(NamedTuple):
    """Correlation at zero lag, envelope-peak correlation, its lag (s), phase (rad), scale, PEP."""

    correlation: float
    max_correlation: float
    shift: float
    phase: float
    scale: float
    pep: float


def tie_metrics(observed, synthetic, dt: float, max_shift: float = 0.02) -> TieMetrics:
    """Tie metrics of ``synthetic`` against ``observed`` (same time axis and window).

    For a band-limited wavelet a constant phase rotation looks much like a time
    shift, so both come from the complex cross-correlation of the analytic
    signals: its envelope peaks at the time shift (positive when the synthetic
    must be delayed), and its phase there is the constant rotation that turns
    the shifted synthetic into the observed trace (:func:`rotate_phase`).
    ``max_correlation`` is the envelope value there (the correlation after the
    best shift and rotation).  ``scale`` and ``pep`` refer to the unshifted,
    unrotated synthetic.
    """
    from scipy.signal import hilbert

    o = np.asarray(observed, dtype=float)
    s = np.asarray(synthetic, dtype=float)
    no, ns = np.linalg.norm(o), np.linalg.norm(s)
    oa, sa = hilbert(o), hilbert(s)
    L = int(round(max_shift / dt))
    lags = np.arange(-L, L + 1)
    # |analytic| norms are twice the real ones (equal energy in the quadrature part)
    cc = np.array([np.vdot(np.roll(sa, k), oa) for k in lags]) / (2.0 * no * ns)
    env = np.abs(cc)
    i = int(np.argmax(env))
    off = 0.0
    if 0 < i < lags.size - 1:
        a, b, c = env[i - 1], env[i], env[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    phi = float(-np.angle(cc[i]))
    scale = float(o @ s / (ns * ns))
    pep = float(1.0 - np.sum((o - scale * s) ** 2) / np.sum(o * o))
    return TieMetrics(float(o @ s / (no * ns)), float(env[i]), float((lags[i] + off) * dt),
                      phi, scale, pep)
