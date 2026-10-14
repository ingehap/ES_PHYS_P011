"""AVA extraction from migrated gathers and intercept-gradient fits.

:func:`pick_event` follows one reflection across a migrated gather (depth by
offset), taking the extreme of a fixed polarity in a window that moves with the
event, so residual moveout from a wrong migration model is tracked rather than
cut.  :func:`intercept_gradient` fits the two-term form ``R = A + B sin^2(theta)``
(Shuey, 1985) by least squares over an angle range.  :func:`linear_resolution`
says which model changes such data can tell apart (the identifiability study of
the plan): Gaussian linear inversion of the sensitivities with prior widths.

References
----------
Shuey, R. T., 1985, A simplification of the Zoeppritz equations: Geophysics,
50(4), 609-614. https://doi.org/10.1190/1.1441936
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

__all__ = ["Pick", "pick_event", "AVAFit", "intercept_gradient", "Resolution",
           "linear_resolution"]


class Pick(NamedTuple):
    """One event across a gather: depth (m), amplitude and angle (rad) per offset."""

    depth: np.ndarray
    amplitude: np.ndarray
    theta: np.ndarray


def _parabolic(y, i):
    """Sub-sample position offset and value of an extreme at index ``i``."""
    if i == 0 or i == y.size - 1:
        return 0.0, y[i]
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2.0 * b + c
    if den == 0.0:
        return 0.0, b
    d = 0.5 * (a - c) / den
    return d, b - 0.25 * (a - c) * d


def pick_event(depth, data, theta, z0: float, half_window: float = 6.0,
               polarity: Optional[int] = None, start: int = 0) -> Pick:
    """Follow the event nearest ``z0`` across ``data[depth, offset]``.

    The first trace (index ``start``) is searched in ``z0 +- half_window``; each
    next trace in a window of the same size centred on the previous pick.  The
    polarity (+1 peak, -1 trough) is that of the largest sample on the first
    trace unless given.  Amplitudes and depths are refined by a parabola;
    ``theta`` is the angle at the picked depth (interpolated).  The angle of a
    migration table is that of the layer just above each depth, so in a model
    with a sharp interface at the reflector read it just above the interface:
    a pick a fraction of a metre below would give the transmitted angle.
    """
    z = np.asarray(depth, dtype=float)
    d = np.asarray(data, dtype=float)
    th = np.asarray(theta, dtype=float)
    n = d.shape[1]
    out = Pick(np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    dz = z[1] - z[0]
    centre = float(z0)
    order = list(range(start, n)) + list(range(start - 1, -1, -1))
    for j in order:
        if j == start - 1:
            centre = out.depth[start]
        w = np.flatnonzero((z >= centre - half_window) & (z <= centre + half_window))
        col = d[w, j]
        if w.size < 3 or not np.all(np.isfinite(col)):
            continue
        if polarity is None:
            polarity = 1 if col[np.argmax(np.abs(col))] > 0 else -1
        i = int(np.argmax(polarity * col))
        off, val = _parabolic(col, i)
        zp = z[w[i]] + off * dz
        out.depth[j], out.amplitude[j] = zp, val
        out.theta[j] = np.interp(zp, z, th[:, j])
        centre = zp
    return out


class AVAFit(NamedTuple):
    """``A + B sin^2(theta)`` fit: intercept, gradient, their covariance, RMS residual, points used."""

    intercept: float
    gradient: float
    cov: np.ndarray
    rms: float
    n: int


def intercept_gradient(theta, amplitude, theta_range: Sequence[float] = (0.0, 90.0),
                       sigma=None) -> AVAFit:
    """Least-squares intercept and gradient over ``theta_range`` (degrees).

    ``sigma`` (per point) weights the fit and scales the covariance; without it
    the covariance uses the residual variance.
    """
    th = np.asarray(theta, dtype=float)
    a = np.asarray(amplitude, dtype=float)
    lo, hi = np.radians(theta_range[0]), np.radians(theta_range[1])
    ok = np.isfinite(th) & np.isfinite(a) & (th >= lo) & (th <= hi)
    G = np.column_stack([np.ones(ok.sum()), np.sin(th[ok]) ** 2])
    s = np.ones(ok.sum()) if sigma is None else np.broadcast_to(np.asarray(sigma, float), th.shape)[ok]
    coef, *_ = np.linalg.lstsq(G / s[:, None], a[ok] / s, rcond=None)
    r = a[ok] - G @ coef
    n = int(ok.sum())
    scale = 1.0 if sigma is not None else (r @ r) / max(n - 2, 1)
    cov = np.linalg.inv((G / s[:, None]).T @ (G / s[:, None])) * scale
    return AVAFit(float(coef[0]), float(coef[1]), cov, float(np.sqrt(np.mean(r * r))), n)


class Resolution(NamedTuple):
    """Linear (Gaussian) resolution of parameters by data: posterior std, data weight, correlation."""

    posterior_std: np.ndarray
    data_weight: np.ndarray
    correlation: np.ndarray


def linear_resolution(J, prior_std, data_cov) -> Resolution:
    """What linearized data ``d = J m`` with covariance ``data_cov`` resolve about ``m``.

    ``prior_std`` are the prior standard deviations of the parameters.  The
    data weight ``1 - posterior variance / prior variance`` is near 0 for a
    parameter the data cannot see; a correlation near +-1 between two
    parameters means the data can explain a change of one by the other.
    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    s = np.asarray(prior_std, dtype=float)
    Cd = np.atleast_2d(np.asarray(data_cov, dtype=float))
    H = J.T @ np.linalg.solve(Cd, J) + np.diag(1.0 / s ** 2)
    C = np.linalg.inv(H)
    sd = np.sqrt(np.diag(C))
    return Resolution(sd, 1.0 - sd ** 2 / s ** 2, C / np.outer(sd, sd))
