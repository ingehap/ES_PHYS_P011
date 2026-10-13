"""Convolutional synthetics: wavelets, normal-incidence traces and angle gathers.

These are the fast, idealized companions of the reflectivity method:

* :func:`normal_incidence` is the exact primaries-only plane-wave response at
  normal incidence.  It can include transmission losses and constant-Q
  attenuation and dispersion, layer by layer and frequency by frequency.  With
  both switched on it equals the reflectivity method at p = 0 with internal
  multiples and the free surface off; the tests check this.
* :func:`angle_gather` places one reflection coefficient per interface at the
  vertical two-way time, for a constant incidence angle at every interface (an
  ideal, NMO-corrected angle gather).  It has no transmission loss, no
  attenuation and no multiples.  This is the usual "elastic synthetic" for AVO
  calibration.

Spectra use the ``exp(-i omega t)`` convention of :mod:`.reflectivity`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..model import ElasticModel1D
from . import reflection
from .elastic import complex_velocity
from .reflectivity import spectrum_to_trace, time_axis_spectrum, wavelet_spectrum

__all__ = ["ricker", "normal_incidence", "angle_gather"]


def ricker(f_peak: float, dt: float, half_length: Optional[float] = None):
    """Zero-phase Ricker wavelet ``(t, w)`` centred on t = 0 with peak value 1."""
    half_length = 1.5 / f_peak if half_length is None else half_length
    t = dt * np.arange(-int(round(half_length / dt)), int(round(half_length / dt)) + 1)
    a = (np.pi * f_peak * t) ** 2
    return t, (1.0 - 2.0 * a) * np.exp(-a)


def _top_index(model: ElasticModel1D, top: Optional[float]):
    if top is None:
        return model, 0
    m = model.split_at([top])
    return m, int(np.searchsorted(m.interfaces, top))


def normal_incidence(model: ElasticModel1D, dt: float, nt: int, wavelet,
                     top: Optional[float] = None, attenuation: bool = False,
                     transmission_loss: bool = False, f_max: Optional[float] = None,
                     damping: Optional[float] = None, return_spectrum: bool = False):
    """Primaries-only normal-incidence reflection response recorded at depth ``top``.

    The source is a unit downgoing plane wave at ``top`` (default: the model top),
    and the trace is the upgoing displacement amplitude there.  Returns
    ``(t, trace)``, or ``(omega, spectrum)`` with ``return_spectrum`` (without
    wavelet).
    """
    m, k0 = _top_index(model, top)
    _, omega, eps, _ = time_axis_spectrum(dt, nt, f_max, damping)
    w_ref = 2.0 * np.pi * m.f_ref
    if attenuation:
        v = complex_velocity(m.vp0[:, None], m.qp[:, None], omega[None, :], w_ref)
    else:
        v = np.broadcast_to(m.vp0[:, None] + 0j, (m.n_layers, omega.size))
    z = m.rho[:, None] * v                                     # impedance (nl, nf)
    r = (z[k0 + 1:] - z[k0:-1]) / (z[k0 + 1:] + z[k0:-1])     # interfaces below top
    phase = np.cumsum(2.0 * m.thickness[k0:-1, None] / v[k0:-1], axis=0)
    spec_terms = r * np.exp(1j * omega[None, :] * phase)
    if transmission_loss:
        tl = np.cumprod(1.0 - r**2, axis=0)
        tl = np.vstack([np.ones((1, omega.size)), tl[:-1]])
        spec_terms = spec_terms * tl
    spec = spec_terms.sum(axis=0)
    if return_spectrum:
        return omega, spec
    W = wavelet_spectrum(wavelet[0], wavelet[1], omega)
    shift = max(0.0, -float(wavelet[0][0]))
    return np.arange(nt) * dt, spectrum_to_trace(spec * W, dt, nt, eps, omega, shift)


_METHODS = ("zoeppritz", "exact_vti", "aki_richards", "shuey", "ruger")


def angle_gather(model: ElasticModel1D, angles_deg, dt: float, nt: int, wavelet,
                 method: str = "zoeppritz", top: Optional[float] = None,
                 f_max: Optional[float] = None):
    """Ideal angle gather: R(theta) of every interface below ``top`` at its vertical TWT.

    ``method`` is one of ``"zoeppritz"`` (exact, isotropic, using vp0, vs0, rho),
    ``"exact_vti"``, ``"aki_richards"``, ``"shuey"`` or ``"ruger"``.  Interfaces
    with a fluid on either side use the exact isotropic coefficient.  Returns
    ``(t, angles_deg, traces[n_angles, nt])``.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    m, k0 = _top_index(model, top)
    theta = np.radians(np.atleast_1d(np.asarray(angles_deg, dtype=float)))
    up = slice(k0, m.n_layers - 1)
    lo = slice(k0 + 1, m.n_layers)
    keep = np.flatnonzero(np.any(np.vstack([getattr(m, n)[lo] != getattr(m, n)[up]
                                            for n in ("vp0", "vs0", "rho", "eps", "delta")]),
                                 axis=0)) + k0
    t_int = m.twt(m.interfaces[keep + 1]) - m.twt(m.interfaces[k0])
    coeffs = np.zeros((theta.size, keep.size))
    for col, i in enumerate(keep):
        u = (m.vp0[i], m.vs0[i], m.rho[i], m.eps[i], m.delta[i])
        lw = (m.vp0[i + 1], m.vs0[i + 1], m.rho[i + 1], m.eps[i + 1], m.delta[i + 1])
        if method == "zoeppritz" or m.is_fluid[i] or m.is_fluid[i + 1]:
            vs1 = max(u[1], 1e-6)
            coeffs[:, col] = reflection.zoeppritz_pp(theta, u[0], vs1, u[2], lw[0], lw[1], lw[2]).real
        elif method == "exact_vti":
            coeffs[:, col] = reflection.exact_pp(theta, u, lw).real
        elif method == "ruger":
            coeffs[:, col] = reflection.ruger(theta, u, lw)
        else:
            fn = getattr(reflection, method)
            coeffs[:, col] = fn(theta, u[0], u[1], u[2], lw[0], lw[1], lw[2])
    _, omega, eps, _ = time_axis_spectrum(dt, nt, f_max, 0.0)
    W = wavelet_spectrum(wavelet[0], wavelet[1], omega)
    phase = np.exp(1j * omega[:, None] * t_int[None, :])           # (nf, n_int)
    spec = coeffs @ phase.T * W[None, :]                           # (n_angles, nf)
    shift = max(0.0, -float(wavelet[0][0]))
    return (np.arange(nt) * dt, np.degrees(theta),
            spectrum_to_trace(spec, dt, nt, 0.0, omega, shift))
