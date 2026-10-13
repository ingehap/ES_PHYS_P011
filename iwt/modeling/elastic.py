"""Plane-wave P-SV quantities for VTI layers.

Conventions
-----------
* Time dependence ``exp(-i omega t)``; a plane wave is
  ``exp(i omega (p x + q z - t))`` with z positive down, so a downgoing wave has
  vertical slowness q with Im(q) >= 0 (it decays downward) and an upgoing wave -q.
* Attenuation follows Kjartansson's (1979) constant-Q law.  The complex velocity
  is ``V(omega) = c0 cos(pi g / 2) (-i omega / omega_ref)^g`` with
  ``g = arctan(1/Q) / pi``, which makes ``c0`` the phase velocity at
  ``omega_ref`` (as in :mod:`iwt.qmodel`).  The P-wave Q acts on C33 and C11, the
  S-wave Q on C44.  This is the simplest form of anisotropic attenuation (a
  single Q per wave type); it is an assumption.
* A wave's amplitude multiplies a polarization vector ``n`` with ``n . n = 1``.
  For P, n points along the propagation direction; for SV it is turned by +90
  degrees from it (a convention; the R/T magnitudes do not depend on it).
* The stress-displacement vector is ``b = [u_x, u_z, t_xz, t_zz]`` with tractions
  divided by ``i omega``: ``t_xz = C44 (q u_x + p u_z)``,
  ``t_zz = C13 p u_x + C33 q u_z``.

The eigenvector matrix ``D`` of a layer has columns
``[P down, S down, P up, S up]``.  Interface scattering follows from continuity
of ``b`` (solid-solid) or of ``u_z`` and ``t_zz`` with ``t_xz = 0`` (fluid-solid).

References
----------
Fryer, G. J., & Frazer, L. N., 1984, Seismic waves in stratified anisotropic media:
GJI, 78(3), 691-710. https://doi.org/10.1111/j.1365-246X.1984.tb05065.x
Kennett, B. L. N., & Kerry, N. J., 1979, Seismic waves in a stratified half space:
GJI, 57(3), 557-583. https://doi.org/10.1111/j.1365-246X.1979.tb06779.x
Kjartansson, E., 1979, Constant Q-wave propagation and attenuation: JGR, 84(B9),
4737-4748. https://doi.org/10.1029/JB084iB09p04737
"""

from __future__ import annotations

import numpy as np

from ..qmodel import dispersion_exponent

__all__ = [
    "complex_velocity",
    "layer_moduli",
    "vertical_slowness",
    "eigenvectors",
    "fluid_eigenvectors",
    "interface_scattering",
    "energy_flux",
]


def complex_velocity(c0, q, omega, omega_ref):
    """Kjartansson complex velocity; ``omega`` may be complex (damped frequency).

    ``c0`` and ``q`` broadcast against ``omega``.  Q = inf gives ``c0``.
    """
    g = dispersion_exponent(q)
    w = np.asarray(omega, dtype=complex)
    return np.asarray(c0) * np.cos(0.5 * np.pi * g) * np.power(-1j * w / omega_ref, g)


def layer_moduli(vp, vs, rho, eps, delta):
    """``(c11, c13, c33, c44)`` from (possibly complex) vertical velocities."""
    c33 = rho * vp**2
    c44 = rho * vs**2
    c11 = c33 * (1.0 + 2.0 * eps)
    c13 = np.sqrt(2.0 * delta * c33 * (c33 - c44) + (c33 - c44) ** 2 + 0j) - c44
    return c11, c13, c33, c44


def _branch(q):
    """Choose the root with Im(q) >= 0 (and Re(q) >= 0 when real)."""
    flip = (q.imag < 0.0) | ((q.imag == 0.0) & (q.real < 0.0))
    return np.where(flip, -q, q)


def vertical_slowness(c11, c13, c33, c44, rho, p):
    """Downgoing vertical slownesses ``(qP, qS)`` of a VTI layer.

    Solves ``A q^4 + B q^2 + C = 0`` from the Christoffel equation.  For a fluid
    (C44 = 0) ``qS`` is returned as NaN.
    """
    a11, a13, a33, a55 = c11 / rho, c13 / rho, c33 / rho, c44 / rho
    p2 = np.asarray(p, dtype=complex) ** 2
    if np.all(a55 == 0):
        qp = _branch(np.sqrt((1.0 - a11 * p2) / a33))
        return qp, np.full_like(qp, np.nan)
    A = a33 * a55
    B = a33 * (a11 * p2 - 1.0) + a55 * (a55 * p2 - 1.0) - (a13 + a55) ** 2 * p2
    C = (a11 * p2 - 1.0) * (a55 * p2 - 1.0)
    s = np.sqrt(B * B - 4.0 * A * C)
    # stable quadratic roots: the root of larger magnitude first
    s = np.where((np.conj(B) * s).real < 0.0, -s, s)
    big = (-B - s) / (2.0 * A)
    small = C / (A * big)
    # P is the faster wave: smaller real part of q^2
    qp2 = np.where(small.real <= big.real, small, big)
    qs2 = np.where(small.real <= big.real, big, small)
    return _branch(np.sqrt(qp2)), _branch(np.sqrt(qs2))


def _normalize(ux, uz):
    n = np.sqrt(ux * ux + uz * uz)
    return ux / n, uz / n


def _polarization(a11, a13, a33, a55, p, q, wave):
    """Unit polarization of a qP or qSV wave with slownesses (p, q)."""
    f1x, f1z = (a13 + a55) * p * q, 1.0 - a11 * p * p - a55 * q * q
    f2x, f2z = 1.0 - a55 * p * p - a33 * q * q, (a13 + a55) * p * q
    use1 = (np.abs(f1x) ** 2 + np.abs(f1z) ** 2) >= (np.abs(f2x) ** 2 + np.abs(f2z) ** 2)
    ux, uz = _normalize(np.where(use1, f1x, f2x), np.where(use1, f1z, f2z))
    # sign: P along (p, q); SV along (q, -p)
    if wave == "P":
        s = (ux * p + uz * q).real
    else:
        s = (ux * q - uz * p).real
    sign = np.where(s < 0.0, -1.0, 1.0)
    return sign * ux, sign * uz


def eigenvectors(c11, c13, c33, c44, rho, p, qp, qs):
    """Eigenvector matrix ``D`` (..., 4, 4) of a solid VTI layer.

    Columns ``[P down, S down, P up, S up]``; rows ``[u_x, u_z, t_xz, t_zz]``.
    """
    a11, a13, a33, a55 = c11 / rho, c13 / rho, c33 / rho, c44 / rho
    p = np.asarray(p, dtype=complex)
    shape = np.broadcast(p, qp).shape
    D = np.empty(shape + (4, 4), dtype=complex)
    for col, (q, wave) in enumerate(((qp, "P"), (qs, "S"), (-qp, "P"), (-qs, "S"))):
        ux, uz = _polarization(a11, a13, a33, a55, p, q, wave)
        D[..., 0, col] = ux
        D[..., 1, col] = uz
        D[..., 2, col] = c44 * (q * ux + p * uz)
        D[..., 3, col] = c13 * p * ux + c33 * q * uz
    return D


def fluid_eigenvectors(c33, rho, p, qp):
    """Fluid eigenvectors (..., 2, 2): rows ``[u_z, t_zz]``, columns ``[P down, P up]``.

    The polarization of P is ``alpha (p, +-q)`` with ``alpha^2 = c33 / rho``.
    """
    alpha = np.sqrt(c33 / rho + 0j)
    shape = np.broadcast(p, qp).shape
    D = np.empty(shape + (2, 2), dtype=complex)
    D[..., 0, 0] = alpha * qp
    D[..., 0, 1] = -alpha * qp
    D[..., 1, 0] = rho * alpha
    D[..., 1, 1] = rho * alpha
    return D


def interface_scattering(D_upper, D_lower, fluid_upper: bool = False, scale: float = 1.0):
    """Scattering blocks ``(RD, TD, RU, TU)`` (..., 2, 2) of one interface.

    * RD: downgoing incident from above -> upgoing above;
    * TD: downgoing incident from above -> downgoing below;
    * RU: upgoing incident from below -> downgoing below;
    * TU: upgoing incident from below -> upgoing above.

    Rows and columns are ``[P, S]``.  With ``fluid_upper`` the upper medium is a
    fluid (``D_upper`` from :func:`fluid_eigenvectors`); its S channel is a
    dummy and all coefficients to and from it are zero.  ``scale`` divides the
    traction rows to balance the linear system (any positive number).
    """
    if not fluid_upper:
        Du = D_upper.copy()
        Dl = D_lower.copy()
        Du[..., 2:, :] /= scale
        Dl[..., 2:, :] /= scale
        M = np.concatenate([Du[..., :, 2:], -Dl[..., :, :2]], axis=-1)
        N = np.concatenate([-Du[..., :, :2], Dl[..., :, 2:]], axis=-1)
        S = np.linalg.solve(M, N)
        return S[..., :2, :2], S[..., 2:, :2], S[..., 2:, 2:], S[..., :2, 2:]

    # fluid over solid: unknowns [u1 (fluid up P), d2P, d2S];
    # inputs [d1 (fluid down P), u2P, u2S]; equations: u_z, t_zz continuous, t_xz = 0
    Df = D_upper.copy()
    Dl = D_lower.copy()
    Df[..., 1, :] /= scale
    Dl[..., 2:, :] /= scale
    shape = Df.shape[:-2]
    M = np.zeros(shape + (3, 3), dtype=complex)
    N = np.zeros(shape + (3, 3), dtype=complex)
    for r, (fr, sr) in enumerate(((0, 1), (1, 3))):     # u_z, t_zz
        M[..., r, 0] = Df[..., fr, 1]
        M[..., r, 1:] = -Dl[..., sr, :2]
        N[..., r, 0] = -Df[..., fr, 0]
        N[..., r, 1:] = Dl[..., sr, 2:]
    M[..., 2, 1:] = -Dl[..., 2, :2]                      # t_xz = 0 in the solid
    N[..., 2, 1:] = Dl[..., 2, 2:]
    S = np.linalg.solve(M, N)
    z = np.zeros(shape + (2, 2), dtype=complex)
    RD, TD, RU, TU = z.copy(), z.copy(), z.copy(), z.copy()
    RD[..., 0, 0] = S[..., 0, 0]
    TD[..., :, 0] = S[..., 1:, 0]
    RU[...] = S[..., 1:, 1:]
    TU[..., 0, :] = S[..., 0, 1:]
    return RD, TD, RU, TU


def energy_flux(D, columns):
    """Vertical energy flux (up to a constant) of the waves in ``columns`` of ``D``.

    ``|Re(conj(u) . t)|`` for a solid ``D`` (4 rows) or ``|Re(conj(u_z) t_zz)|``
    for a fluid ``D`` (2 rows).  Use only for propagating waves at real
    frequency.
    """
    out = []
    for c in columns:
        if D.shape[-2] == 2:          # fluid: rows [u_z, t_zz]
            f = (np.conj(D[..., 0, c]) * D[..., 1, c]).real
        else:
            f = np.sum(np.conj(D[..., :2, c]) * D[..., 2:, c], axis=-1).real
        out.append(np.abs(f))
    return np.stack(out, axis=-1)
