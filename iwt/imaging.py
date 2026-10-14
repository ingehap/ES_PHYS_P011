"""1D prestack depth migration of CMP gathers with ray-based Q compensation.

A CMP gather over a laterally homogeneous earth is migrated offset by offset:
every output depth z takes the sample of each trace at the two-way PP time
T(x, z) of the migration model (:func:`iwt.raytrace.reflection_table`),
multiplied by the true-amplitude spreading factor, so that a flat event carries
the plane-wave reflection coefficient R(theta) at the reflection angle
theta(x, z).  This is the 1D form of a Kirchhoff depth migration with
stationary-phase weights.

**Q compensation** follows the ray-based approach of Cavalca et al. (2013): the
attenuation accumulated along each source-reflector-receiver ray,
``t* = Sum t_j / Q_j``, is removed inside the migration operator, so every
output point gets its own inverse filter.  For constant-Q (Kjartansson, 1979)
propagation the operator of a path with attenuation time t* is, to first order
in 1/Q, in the ``exp(-i omega t)`` inverse-transform convention of numpy,

    H(f) = exp(-pi f t*) exp(2 i f t* ln(f / f_ref)),

an amplitude decay and a phase that makes frequencies above ``f_ref`` arrive
earlier (``f_ref`` is the frequency at which the model velocities apply).  The
inverse amplitude ``1 / exp(-pi f t*)`` grows without bound, so it is
stabilized as in Wang (2002),

    Lambda(f) = (beta + s) / (beta^2 + s),    beta = exp(-pi f t*),

with ``s`` chosen so that the largest gain equals ``gain_limit``.  The phase is
always compensated in full.  ``q_compensation="phase"`` compensates the phase
only.

The migrated gather keeps the source wavelet as recorded by the receivers; for
the point-source records of :mod:`iwt.modeling.reflectivity` (explosion in the
sea, hydrophones) that is the time derivative of the source-potential wavelet,
the same at every offset.  Source and receiver ghosts are not corrected.

References
----------
Cavalca, M., Fletcher, R., & Riedel, M., 2013, Q-compensation in complex media -
Ray-based and wavefield extrapolation approaches: SEG Technical Program Expanded
Abstracts 2013, 3831-3835. https://doi.org/10.1190/segam2013-1130.1
Kjartansson, E., 1979, Constant Q-wave propagation and attenuation: JGR, 84(B9),
4737-4748. https://doi.org/10.1029/JB084iB09p04737
Wang, Y., 2002, A stable and efficient approach of inverse Q filtering:
Geophysics, 67(2), 657-663. https://doi.org/10.1190/1.1468627
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .model import ElasticModel1D
from .raytrace import ReflectionTable, reflection_table

__all__ = [
    "attenuation_operator",
    "stabilization",
    "attenuate",
    "compensate",
    "high_cut_taper",
    "Image",
    "migrate_cmp",
]


def attenuation_operator(f, tstar, f_ref: float):
    """Constant-Q propagation operator ``H(f)`` for attenuation time ``tstar`` (numpy convention)."""
    f = np.abs(np.asarray(f, dtype=float))
    tstar = np.asarray(tstar, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnf = np.where(f > 0, np.log(np.where(f > 0, f, 1.0) / f_ref), 0.0)
    return np.exp(-np.pi * f * tstar) * np.exp(2j * f * tstar * lnf)


def stabilization(gain_limit: float) -> float:
    """Wang's (2002) parameter ``s`` for which ``max (beta + s) / (beta^2 + s) = gain_limit``."""
    from scipy.optimize import brentq

    if not gain_limit > 1.0:
        raise ValueError("gain_limit must exceed 1")
    if not np.isfinite(gain_limit):
        return 0.0

    def peak(log_s):
        s = np.exp(log_s)
        r = np.sqrt(s * s + s)                     # beta* + s at the maximum
        return r / (2.0 * s * (s + 1.0 - r)) - gain_limit

    return float(np.exp(brentq(peak, -60.0, 10.0)))


def _inverse(f, tstar, f_ref, mode: str, s: float):
    """Inverse of :func:`attenuation_operator`, amplitude stabilized with ``s``."""
    if mode not in ("amplitude", "phase"):
        raise ValueError("q_compensation must be None, 'amplitude' or 'phase'")
    H = attenuation_operator(f, tstar, f_ref)
    beta = np.abs(H)
    phase = np.conj(H) / np.where(beta > 0, beta, 1.0)
    if mode == "phase":
        return phase
    return phase * (beta + s) / (beta * beta + s)


def attenuate(trace, dt: float, tstar: float, f_ref: float = 30.0):
    """Apply a constant-Q path with attenuation time ``tstar`` (s) to a trace."""
    trace = np.asarray(trace, dtype=float)
    n = trace.shape[-1]
    f = np.fft.rfftfreq(n, dt)
    return np.fft.irfft(np.fft.rfft(trace) * attenuation_operator(f, tstar, f_ref), n=n)


def compensate(trace, dt: float, tstar: float, f_ref: float = 30.0, mode: str = "amplitude",
               gain_limit: float = 10.0):
    """Remove a constant-Q path with attenuation time ``tstar`` (stationary inverse Q filter)."""
    trace = np.asarray(trace, dtype=float)
    n = trace.shape[-1]
    f = np.fft.rfftfreq(n, dt)
    inv = _inverse(f, tstar, f_ref, mode, stabilization(gain_limit))
    return np.fft.irfft(np.fft.rfft(trace) * inv, n=n)


class Image(NamedTuple):
    """A migrated CMP gather: ``data[depth, offset]`` and the reflection angle (rad)."""

    depth: np.ndarray
    offsets: np.ndarray
    data: np.ndarray
    theta: np.ndarray
    table: ReflectionTable


def high_cut_taper(f, high_cut: Optional[float]):
    """1 up to 0.8 ``high_cut``, a cosine taper to 0 at ``high_cut`` (all ones if None)."""
    f = np.asarray(f, dtype=float)
    if high_cut is None:
        return np.ones_like(f)
    lo = 0.8 * high_cut
    x = np.clip((f - lo) / (high_cut - lo), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _sample(trace, dt: float, t0: float, times, tstar, f_ref, mode, s, high_cut=None,
            chunk: int = 256):
    """Band-limited value of ``trace`` at ``times``, with per-sample inverse Q filters."""
    n = trace.size
    f = np.fft.rfftfreq(n, dt)
    U = np.fft.rfft(trace) * high_cut_taper(f, high_cut)
    wgt = np.full(f.size, 2.0)
    wgt[0] = 1.0
    if n % 2 == 0:
        wgt[-1] = 1.0
    U = U * wgt / n
    out = np.full(times.size, np.nan)
    ok = np.flatnonzero(np.isfinite(times) & (times >= t0) & (times <= t0 + (n - 1) * dt))
    for c in range(0, ok.size, chunk):
        k = ok[c:c + chunk]
        E = np.exp(2j * np.pi * f[None, :] * (times[k, None] - t0))
        if mode is not None:
            E = E * _inverse(f[None, :], tstar[k, None], f_ref, mode, s)
        out[k] = (E @ U).real
    return out


def migrate_cmp(data, dt: float, offsets, model: ElasticModel1D, depths,
                z_source: float = 0.0, z_receiver: Optional[float] = None,
                t0: float = 0.0, q_compensation: Optional[str] = None,
                gain_limit: float = 10.0, f_ref: Optional[float] = None,
                high_cut: Optional[float] = None, true_amplitude: bool = True,
                table: Optional[ReflectionTable] = None) -> Image:
    """Depth-migrate a CMP gather ``data[offset, time]`` with the layered ``model``.

    ``depths`` is the output grid (m).  ``q_compensation`` is None, ``"amplitude"``
    (amplitude and phase, gain-limited) or ``"phase"``; the Q comes from the
    model's ``qp`` and ``f_ref`` defaults to the model's.  With
    ``true_amplitude`` each sample is multiplied by the spreading factor, so a
    reflection reads as its plane-wave coefficient (times the wavelet).
    ``high_cut`` (Hz) tapers the spectrum of every variant the same way, which
    keeps the compensated band comparable with the uncompensated one.  A
    precomputed ``table`` may be passed to migrate several gathers.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    x = np.abs(np.atleast_1d(np.asarray(offsets, dtype=float)))
    if table is None:
        table = reflection_table(model, depths, x, z_source, z_receiver)
    f_ref = model.f_ref if f_ref is None else float(f_ref)
    s = stabilization(gain_limit) if q_compensation == "amplitude" else 0.0
    img = np.full(table.time.shape, np.nan)
    for i in range(x.size):
        img[:, i] = _sample(data[i], dt, t0, table.time[:, i], table.tstar[:, i], f_ref,
                            q_compensation, s, high_cut)
    if true_amplitude:
        img = img * table.spreading
    return Image(table.depth, x, img, table.theta, table)
