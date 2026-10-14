"""Exact qP kinematics in a 1D VTI model: slowness, polarization, direct and reflected rays.

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

:func:`reflection_table` traces the PP reflection from every depth of an output
grid to every offset of a surface gather, with its reflection angle, the
attenuation ``t* = Sum t_j / Q_j`` along the ray and the geometrical spreading.
For a point source in a layered medium the stationary-phase evaluation of the
Sommerfeld integral gives the reflected amplitude

    A = R(p) sqrt(p / (x |dX/dp|)) / q_s

(times frequency-dependent source and receiver constants), with ``q_s`` the
vertical slowness at the source; in a homogeneous medium this is ``R / distance``.
The spreading factor ``q_s sqrt(x |dX/dp| / p)`` therefore turns a reflection
amplitude into the plane-wave coefficient R(p) (true-amplitude correction).

References
----------
Tsvankin, I., 2012, Seismic Signatures and Analysis of Reflection Data in
Anisotropic Media, 3rd ed.: SEG. https://doi.org/10.1190/1.9781560803003
Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954-1966.
https://doi.org/10.1190/1.1442051
Aki, K., & Richards, P. G., 1980, Quantitative Seismology: W. H. Freeman (the
Sommerfeld integral and its stationary-phase evaluation).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .model import ElasticModel1D

__all__ = [
    "qp_phase_velocity",
    "qp_vertical_slowness",
    "qp_group_velocity",
    "qp_polarization",
    "velocity_stiffness",
    "Rays",
    "two_point",
    "ReflectionTable",
    "reflection_table",
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


def qp_vertical_slowness(p, a11, a13, a33, a44, derivative: int = 0):
    """qP vertical slowness q(p) (s/m) for density-normalized stiffnesses (rho = 1).

    ``a44 = 0`` is a fluid.  NaN where the qP wave is evanescent (p beyond the
    horizontal slowness).  ``derivative=1`` (or True) also returns dq/dp,
    ``derivative=2`` dq/dp and d2q/dp2.
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
        if derivative < 2:
            return q, dqdp
        # second implicit derivative: F_PP + 2 F_PQ Q' + F_QQ Q'^2 + F_Q Q'' = 0
        d2Q_s = -(2.0 * a11 * a44 + 2.0 * db * dQdP + 2.0 * a * dQdP ** 2) / (2.0 * a * Q + b)
        d2Q = np.where(fluid, 0.0, d2Q_s)
        d2qdp2 = dQdP / q + 2.0 * P * d2Q / q - P * dQdP ** 2 / q ** 3
    return q, dqdp, d2qdp2


def qp_group_velocity(theta, a11, a13, a33, a44):
    """qP group velocity (m/s) and group (ray) angle (rad) at phase angle ``theta``.

    The ray direction is normal to the slowness surface, ``tan(psi) = -dq/dp``,
    and the group velocity satisfies ``p sin(psi) + q cos(psi) = 1 / v_g``.
    """
    v = qp_phase_velocity(theta, a11, a13, a33, a44)
    p = np.sin(theta) / v
    q, dq = qp_vertical_slowness(p, a11, a13, a33, a44, derivative=1)
    psi = np.arctan(-dq)
    return 1.0 / (p * np.sin(psi) + q * np.cos(psi)), psi


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


class ReflectionTable(NamedTuple):
    """PP reflections from ``depth`` to ``offsets``: ``[depth, offset]`` arrays.

    ``time`` two-way traveltime (s); ``p`` ray parameter (s/m); ``theta`` phase
    angle of incidence at the reflector (rad, in the layer just above it);
    ``tstar`` attenuation time ``Sum t_j / Q_j`` (s); ``spreading`` the factor
    that turns a point-source amplitude into the plane-wave coefficient (see the
    module docstring).  NaN where no ray exists.
    """

    depth: np.ndarray
    offsets: np.ndarray
    time: np.ndarray
    p: np.ndarray
    theta: np.ndarray
    tstar: np.ndarray
    spreading: np.ndarray


def reflection_table(model: ElasticModel1D, depths, offsets, z_source: float = 0.0,
                     z_receiver: float = None, n_p: int = 2000, refine: int = 3,
                     max_angle: float = 85.0) -> ReflectionTable:
    """PP reflection rays from every depth in ``depths`` to every offset.

    Source and receiver sit at ``z_source`` and ``z_receiver`` (default: the
    source depth); the reflectors must lie below both.  The ray of each
    (depth, offset) pair is found on a grid of ``n_p`` ray parameters and refined
    by ``refine`` Newton steps on the exact layer sums.  Q is the P-wave Q of the
    model (``inf`` for no attenuation), the same in every direction.
    """
    zs = float(z_source)
    zr = zs if z_receiver is None else float(z_receiver)
    z_out = np.atleast_1d(np.asarray(depths, dtype=float))
    x = np.abs(np.atleast_1d(np.asarray(offsets, dtype=float)))
    if np.any(z_out <= max(zs, zr)):
        raise ValueError("reflectors must lie below the source and the receiver")
    m = model.split_at(np.concatenate([[zs, zr], z_out]))
    c = m.cij()
    A = np.array([c.c11, c.c13, c.c33, c.c44]) / m.rho
    h = m.thickness
    top = m.interfaces[:-1]
    # number of legs (down from the source, up to the receiver) crossing each layer
    legs = (top >= zs - 1e-9).astype(float) + (top >= zr - 1e-9)
    with np.errstate(divide="ignore"):
        qinv = np.where(np.isfinite(m.qp), 1.0 / m.qp, 0.0)
    th_max = np.radians(max_angle)
    p_lim = np.where(legs > 0, np.sin(th_max) / qp_phase_velocity(th_max, *A), np.inf)
    k_out = np.searchsorted(m.interfaces, z_out)          # layers [0, k) lie above z_out
    n_lay = int(k_out.max())
    p_cap = np.minimum.accumulate(p_lim[:n_lay])          # smallest limit above each depth
    i_src = int(np.searchsorted(m.interfaces, zs, side="right")) - 1

    # coarse tables on a common slowness grid: X(p) and its depth accumulation
    pg = p_cap[legs[:n_lay] > 0][0] * np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, n_p + 1)[:-1])
    w = (legs * h)[None, :n_lay]
    qg, dqg = qp_vertical_slowness(pg[:, None], *A[:, None, :n_lay], derivative=1)
    bad = pg[:, None] > p_lim[None, :n_lay]
    Xg = np.cumsum(np.where(legs[None, :n_lay] > 0, np.where(bad, np.nan, -w * dqg), 0.0), axis=1)

    out = {k: np.full((z_out.size, x.size), np.nan) for k in ReflectionTable._fields[2:]}
    for i, k in enumerate(k_out):
        fin = np.isfinite(Xg[:, k - 1])
        Xk, pk = Xg[fin, k - 1], pg[fin]
        keep = np.concatenate([[True], np.diff(Xk) > 0])
        Xk, pk = Xk[keep], pk[keep]
        if Xk.size < 2:
            continue
        inside = x <= Xk[-1]
        if not inside.any():
            continue
        xi = x[inside]
        p = np.interp(xi, Xk, pk)
        a = A[:, None, :k]
        wk = w[:, :k]
        for _ in range(refine):                    # Newton on X(p) = x with exact sums
            q, dq, d2q = qp_vertical_slowness(p[:, None], *a, derivative=2)
            X = -(wk * dq).sum(axis=1)
            dX = -(wk * d2q).sum(axis=1)
            p = np.clip(p - (X - xi) / dX, 0.0, p_cap[k - 1] * (1.0 - 1e-12))
        q, dq, d2q = qp_vertical_slowness(p[:, None], *a, derivative=2)
        X = -(wk * dq).sum(axis=1)
        dX = -(wk * d2q).sum(axis=1)
        good = np.isfinite(X) & (np.abs(X - xi) <= 1e-6 * np.maximum(xi, 1.0) + 1e-3)
        tau = (wk * q).sum(axis=1)
        t_layer = wk * (q - p[:, None] * dq)       # traveltime in each layer
        q_s = qp_vertical_slowness(p, *A[:, i_src])
        ratio = np.where(p > 1e-12, xi / np.where(p > 1e-12, p, 1.0), dX)
        cols = np.flatnonzero(inside)[good]
        out["time"][i, cols] = (tau + p * xi)[good]
        out["p"][i, cols] = p[good]
        out["theta"][i, cols] = np.arctan2(p, q[:, -1])[good]
        out["tstar"][i, cols] = (t_layer * qinv[None, :k]).sum(axis=1)[good]
        out["spreading"][i, cols] = (q_s * np.sqrt(ratio * dX))[good]
    return ReflectionTable(z_out, x, *(out[f] for f in ReflectionTable._fields[2:]))
