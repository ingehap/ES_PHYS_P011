"""Exact qP kinematics in a 1D VTI model: slowness, polarization and two-point rays.

For a horizontal slowness ``p`` the vertical slowness ``q`` of the qP wave in a VTI
layer is the smaller root of the Christoffel equation (Tsvankin, 2012),

    C33 C44 q^4 + b q^2 + c = 0,
    b = C33 (C11 p^2 - rho) + C44 (C44 p^2 - rho) - (C13 + C44)^2 p^2,
    c = (C11 p^2 - rho)(C44 p^2 - rho),

and the ray (group) direction follows from ``dx/dz = -dq/dp``.  In a laterally
homogeneous layered medium p is constant along the ray, so for a source at depth
z_s and a receiver at z_r

    X(p) = -Sum_j h_j dq_j/dp,        T(p) = p X(p) + Sum_j h_j q_j,

over the layers crossed.  :func:`two_point` evaluates X and T on a slowness grid
for every receiver at once and finds the ray to each source offset by
interpolation and Newton refinement.  The qP polarization is the Christoffel
eigenvector at (p, q).  Fluid layers (C44 = 0) are isotropic.

Only direct (transmitted, downgoing) qP rays are traced.  Rays steeper than
``max_angle`` (phase angle from the vertical) in any layer crossed are excluded:
near the critical slowness of a fast layer the ray grazes along it, and the first
arrival is then a refracted (head) wave, which is not modelled.

References
----------
Tsvankin, I., 2012, Seismic Signatures and Analysis of Reflection Data in
Anisotropic Media, 3rd ed.: SEG. https://doi.org/10.1190/1.9781560803003
Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954-1966.
https://doi.org/10.1190/1.1442051
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .model import ElasticModel1D

__all__ = [
    "qp_phase_velocity",
    "qp_vertical_slowness",
    "qp_polarization",
    "velocity_stiffness",
    "Rays",
    "two_point",
]


def velocity_stiffness(vp0, vs0, eps, delta):
    """Density-normalized ``(A11, A13, A33, A44)`` (m^2/s^2) from Thomsen parameters."""
    a33, a44 = np.asarray(vp0, float) ** 2, np.asarray(vs0, float) ** 2
    a11 = a33 * (1.0 + 2.0 * np.asarray(eps, float))
    a13 = np.sqrt(2.0 * a33 * (a33 - a44) * np.asarray(delta, float) + (a33 - a44) ** 2) - a44
    return a11, a13, a33, a44


def qp_phase_velocity(theta, a11, a13, a33, a44):
    """Exact qP phase velocity (m/s) at phase angle ``theta`` (rad from the vertical)."""
    s2 = np.sin(theta) ** 2
    c2 = 1.0 - s2
    b = (a11 + a44) * s2 + (a33 + a44) * c2
    d = np.sqrt(((a11 - a44) * s2 - (a33 - a44) * c2) ** 2 + 4.0 * (a13 + a44) ** 2 * s2 * c2)
    return np.sqrt(0.5 * (b + d))


def qp_vertical_slowness(p, a11, a13, a33, a44, derivative: bool = False):
    """qP vertical slowness q(p) (s/m) for density-normalized stiffnesses (rho = 1).

    ``a44 = 0`` is a fluid.  NaN where the qP wave is evanescent (p beyond the
    horizontal slowness).  With ``derivative`` also returns dq/dp.
    """
    p = np.asarray(p, dtype=float)
    a11, a13, a33, a44 = np.broadcast_arrays(*(np.asarray(x, float) for x in (a11, a13, a33, a44)))
    P = p * p
    fluid = a44 == 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        # solid: quadratic in Q = q^2
        a = a33 * a44
        b = a33 * (a11 * P - 1.0) + a44 * (a44 * P - 1.0) - (a13 + a44) ** 2 * P
        c = (a11 * P - 1.0) * (a44 * P - 1.0)
        disc = np.sqrt(b * b - 4.0 * a * c)
        # the qP root is the smaller Q; written to avoid cancellation
        Q_s = np.where(b < 0, 2.0 * c / (-b + disc), (-b - disc) / (2.0 * a))
        Q_f = 1.0 / a33 - P
        Q = np.where(fluid, Q_f, Q_s)
        q = np.where(Q > 0, np.sqrt(np.where(Q > 0, Q, 1.0)), np.nan)
        if not derivative:
            return q
        # implicit derivative: F(P, Q) = a Q^2 + b(P) Q + c(P) = 0
        db = a33 * a11 + a44 * a44 - (a13 + a44) ** 2
        dc = a11 * (a44 * P - 1.0) + a44 * (a11 * P - 1.0)
        dQdP_s = -(db * Q + dc) / (2.0 * a * Q + b)
        dQdP = np.where(fluid, -1.0, dQdP_s)
        dqdp = p / q * dQdP
    return q, dqdp


def qp_polarization(p, q, a11, a13, a33, a44):
    """qP polarization angle from the vertical (rad), in the direction of propagation."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    ux = (a13 + a44) * p * q
    uz = 1.0 - a11 * p * p - a44 * q * q
    fluid = np.asarray(a44) == 0.0
    # in a fluid the motion is along the slowness vector
    ux = np.where(fluid, p, ux)
    uz = np.where(fluid, q, uz)
    sgn = np.where(ux * p + uz * q < 0, -1.0, 1.0)      # point along the propagation
    return np.arctan2(sgn * ux, sgn * uz)


class Rays(NamedTuple):
    """Direct qP rays: ``[receiver, offset]`` arrays.

    ``time`` (s), ray parameter ``p`` (s/m), vertical slowness ``q`` and
    polarization angle ``psi`` (rad from vertical) at the receiver, and the phase
    angle ``theta`` (rad) there.
    """

    time: np.ndarray
    p: np.ndarray
    q: np.ndarray
    psi: np.ndarray
    theta: np.ndarray


def two_point(model: ElasticModel1D, z_source: float, z_receivers, offsets,
              n_p: int = 4000, refine: int = 3, max_angle: float = 85.0) -> Rays:
    """Direct qP rays from a source at ``z_source`` to receivers at ``z_receivers``.

    ``offsets`` (m, horizontal source-receiver distances, >= 0).  Rays are found
    on a grid of ``n_p`` ray parameters and refined by ``refine`` Newton steps.
    NaN where no direct ray exists with a phase angle below ``max_angle``
    (degrees) in every layer crossed.
    """
    z_rec = np.atleast_1d(np.asarray(z_receivers, dtype=float))
    x = np.abs(np.atleast_1d(np.asarray(offsets, dtype=float)))
    m = model.split_at(np.concatenate([[z_source], z_rec]))
    c = m.cij()
    rho = m.rho
    A = [c.c11 / rho, c.c13 / rho, c.c33 / rho, c.c44 / rho]
    s = int(np.searchsorted(m.interfaces, z_source))
    out = {k: np.full((z_rec.size, x.size), np.nan) for k in Rays._fields}
    for k, zr in enumerate(z_rec):
        r = int(np.searchsorted(m.interfaces, zr))
        sl = slice(s, r)
        h = m.thickness[sl]
        a = [ai[sl] for ai in A]
        # largest p: the ray reaches max_angle in the fastest layer crossed
        th_max = np.radians(max_angle)
        p_top = np.min(np.sin(th_max) / qp_phase_velocity(th_max, *a))
        pg = p_top * np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, n_p + 1)[:-1])

        def xt(pp):
            q, dq = qp_vertical_slowness(pp[:, None], *[ai[None, :] for ai in a], derivative=True)
            X = -(dq * h).sum(axis=1)
            return X, pp * X + (q * h).sum(axis=1)

        X, _ = xt(pg)
        ok = np.isfinite(X) & np.concatenate([[True], np.diff(X) > 0])
        pg, X = pg[ok], X[ok]
        inside = (x >= X[0]) & (x <= X[-1])
        p = np.interp(x[inside], X, pg)
        for _ in range(refine):          # Newton on X(p) = x
            dp = 1e-7 * p_top
            p1 = np.minimum(p + dp, p_top)
            Xp, Xq = xt(p)[0], xt(p1)[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                step = (Xp - x[inside]) * (p1 - p) / (Xq - Xp)
            p = np.clip(p - np.nan_to_num(step), 0.0, p_top)
        Xf, Tf = xt(p)
        Tf = Tf + p * (x[inside] - Xf)    # dT/dX = p: first-order correction of the offset
        lay = [ai[r - 1] for ai in A]     # the layer just above the receiver
        q_r = qp_vertical_slowness(p, *lay)
        out["time"][k, inside] = Tf
        out["p"][k, inside] = p
        out["q"][k, inside] = q_r
        out["psi"][k, inside] = qp_polarization(p, q_r, *lay)
        out["theta"][k, inside] = np.arctan2(p, q_r)
    return Rays(*(out[f] for f in Rays._fields))
