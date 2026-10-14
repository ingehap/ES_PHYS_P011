"""Walkaway VSP anisotropy by slowness-polarization inversion.

At each receiver level of a walkaway VSP the direct qP wave gives, for every
source position (Gaiser, 1990; Miller & Spencer, 1994; Leaney & Hornby, 2007):

* the horizontal slowness ``p = dt/dx_source`` (equal to the horizontal slowness
  at the receiver only in a laterally homogeneous medium);
* the vertical slowness ``q = dt/dz`` from the moveout across the receivers;
* the polarization angle from the 3C hodogram.

The pairs (p, q) sample the qP slowness surface of the medium around the
receivers: a phase velocity ``1/|(p, q)|`` at a phase angle ``atan(p/q)``.  The
polarization adds the direction of particle motion.  :func:`invert_level` fits
(Vp0, Vs0, eps, delta) with the *exact* VTI Christoffel equation
(:mod:`iwt.raytrace`), with Gaussian priors, and returns the Laplace
(Gauss-Newton) posterior.  ``method="weak"`` fits Thomsen's (1986)
weak-anisotropy phase velocity instead, for comparison.

Error model.  p, q and the polarization are linear combinations of picked times
and single-trace polarizations, so :func:`measure_slowness` propagates
independent pick errors into their joint covariance (the moving source windows
make neighbouring p estimates correlated).  The inversion maps it through the
linearized residuals (effective variance, which includes the error of the
observed angle) and holds it fixed within each iteration.  Two properties of the
estimator found in the tests:

* slowness-only fits with priors have calibrated errors and no measurable bias;
  with polarizations the estimator has an errors-in-variables bias of about half
  a standard deviation at 0.2 ms / 1.5 degree noise, growing with the noise
  variance (Vp0 low, eps and delta high);
* P-wave slowness-polarization data are governed by Vp0 and two combinations of
  eps and delta that absorb Vs0 (Grechka & Mateeva, 2007).  With polarizations an
  error in the Vs0 prior therefore maps into eps and delta; the phase velocity
  alone hardly depends on Vs0 but determines delta much less well.

Two consistency checks guard the assumptions of the method:

* :func:`azimuthal_consistency`: estimates from several azimuths must agree
  within their errors, or the medium is not VTI (Miller & Spencer, 1994);
* :func:`polarization_check`: the polarizations predicted from a slowness-only
  fit must match the measured ones.  A lateral velocity variation biases the
  source-side p but not q or the polarization, and shows up here (cf. Grechka &
  Mateeva, 2007, who fit the slowness-polarization dependence).

References
----------
Gaiser, J. E., 1990, Transversely isotropic phase velocity analysis from
slowness estimates: JGR, 95(B7), 11241-11254.
https://doi.org/10.1029/JB095iB07p11241
Grechka, V., & Mateeva, A., 2007, Inversion of P-wave VSP data for local
anisotropy: Theory and case study: Geophysics, 72(4), D69-D79.
https://doi.org/10.1190/1.2742970
Leaney, W. S., & Hornby, B. E., 2007, Depth-dependent anisotropy from sub-salt
walkaway VSP data: 69th EAGE Conference & Exhibition.
https://doi.org/10.3997/2214-4609.201401677
Miller, D. E., & Spencer, C., 1994, An exact inversion for anisotropic moduli
from phase slowness data: JGR, 99(B11), 21651-21657.
https://doi.org/10.1029/94JB01848
Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954-1966.
https://doi.org/10.1190/1.1442051
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..model import ElasticModel1D
from ..raytrace import qp_phase_velocity, qp_polarization, two_point, velocity_stiffness

__all__ = [
    "Walkaway",
    "synthetic_walkaway",
    "hodogram_polarization",
    "SlownessData",
    "measure_slowness",
    "PARAMETERS",
    "Posterior",
    "invert_level",
    "azimuthal_consistency",
    "polarization_check",
]

PARAMETERS = ("vp0", "vs0", "eps", "delta")


@dataclass
class Walkaway:
    """First-arrival times and polarizations ``[azimuth, source, receiver]``.

    ``offsets`` are signed source positions along each line (m from the well),
    ``psi`` the polarization angles from the vertical (rad), positive towards
    positive offsets.
    """

    offsets: np.ndarray
    depths: np.ndarray
    azimuths: np.ndarray
    time: np.ndarray
    psi: np.ndarray
    source_depth: float


def synthetic_walkaway(model: ElasticModel1D, offsets, depths, azimuths=(0.0,),
                       source_depth: float = 6.0, time_noise: float = 0.0,
                       psi_noise: float = 0.0,
                       statics: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       azimuth_models: Optional[Sequence[ElasticModel1D]] = None,
                       seed: Optional[int] = 0) -> Walkaway:
    """Direct qP times and polarizations of a walkaway VSP by exact ray tracing.

    ``model`` should be the seismic-scale (e.g. Backus-averaged) VTI model.
    ``azimuth_models`` (one per azimuth) replace it to simulate azimuthal
    anisotropy, each azimuth seeing its own vertical symmetry plane.
    ``statics(offset)`` adds a source-side delay (s), a proxy for lateral
    velocity variations near the source.  Gaussian noise of ``time_noise`` (s)
    and ``psi_noise`` (rad) is added.
    Ray theory is used as is: in a model layered at the scale of the receiver
    sub-array, the slowness (an average over the sub-array) and the polarization
    (local at each receiver) see different media.
    """
    x = np.asarray(offsets, dtype=float)
    z = np.asarray(depths, dtype=float)
    az = np.atleast_1d(np.asarray(azimuths, dtype=float))
    rng = np.random.default_rng(seed)
    t = np.empty((az.size, x.size, z.size))
    psi = np.empty_like(t)
    for i in range(az.size):
        m = model if azimuth_models is None else azimuth_models[i]
        rays = two_point(m, source_depth, z, np.abs(x))
        t[i] = rays.time.T
        psi[i] = (np.sign(x)[None, :] * rays.psi).T
    if statics is not None:
        t = t + np.asarray(statics(x), float)[None, :, None]
    if time_noise:
        t = t + time_noise * rng.standard_normal(t.shape)
    if psi_noise:
        psi = psi + psi_noise * rng.standard_normal(psi.shape)
    return Walkaway(x, z, az, t, psi, source_depth)


def hodogram_polarization(ux, uz) -> float:
    """Polarization axis angle from the vertical (rad) of a 2C window, by principal components.

    The axis is returned with a non-negative vertical component, angle in
    (-pi/2, pi/2], so the result does not depend on the wavelet polarity.
    """
    c = np.cov(np.vstack([np.asarray(ux, float), np.asarray(uz, float)]))
    w, v = np.linalg.eigh(c)
    ax, az_ = v[:, np.argmax(w)]
    if az_ < 0 or (az_ == 0 and ax < 0):
        ax, az_ = -ax, -az_
    return float(np.arctan2(ax, az_))


class SlownessData(NamedTuple):
    """Slowness-polarization measurements at one level (one entry per source).

    ``cov`` is the joint covariance of the stacked vector ``(p, q, psi)``
    (3n x 3n).  The estimates share traces (moving windows of sources for p,
    the middle receiver for both p and q), so their errors are correlated;
    ``sigma_*`` are the marginal standard deviations.
    """

    p: np.ndarray
    q: np.ndarray
    psi: np.ndarray
    sigma_p: np.ndarray
    sigma_q: np.ndarray
    sigma_psi: np.ndarray
    depth: float
    top: float
    bottom: float
    cov: Optional[np.ndarray] = None

    def select(self, mask) -> "SlownessData":
        m = np.flatnonzero(np.asarray(mask, dtype=bool))
        cov = None
        if self.cov is not None:
            n = self.p.size
            k = np.concatenate([m, m + n, m + 2 * n])
            cov = self.cov[np.ix_(k, k)]
        return SlownessData(*(f[m] for f in self[:6]), self.depth, self.top, self.bottom, cov)

    def covariance(self) -> np.ndarray:
        """Joint covariance of ``(p, q, psi)`` (diagonal if none was given)."""
        if self.cov is not None:
            return self.cov
        return np.diag(np.concatenate([self.sigma_p, self.sigma_q, self.sigma_psi]) ** 2)

    @property
    def theta(self) -> np.ndarray:
        """Phase angle (rad) of each measurement."""
        return np.arctan2(self.p, self.q)

    @property
    def velocity(self) -> np.ndarray:
        """Phase velocity ``1 / |(p, q)|`` (m/s) of each measurement."""
        return 1.0 / np.hypot(self.p, self.q)


def _slope_weights(x):
    """Least-squares slope operator: ``slope = w @ y``."""
    d = np.asarray(x, float) - np.mean(x)
    return d / (d @ d)


def measure_slowness(survey: Walkaway, level: float, n_receivers: int = 5,
                     n_sources: int = 5, azimuth: int = 0,
                     time_noise: Optional[float] = None,
                     psi_noise: Optional[float] = None) -> SlownessData:
    """p, q and polarization at the receiver sub-array centred on ``level``.

    q is the least-squares slope of the times across the ``n_receivers``
    receivers nearest to ``level`` for each source; p the slope of the times at
    the middle receiver across ``n_sources`` neighbouring sources on the same
    side of the well; the polarization is the sub-array mean.  Both sides of the
    line are kept, with ``|p|`` and ``|psi|``.

    The covariance of all estimates follows from the linear operators that
    produce them, for independent pick errors of standard deviation
    ``time_noise`` (s) and single-trace polarization errors ``psi_noise`` (rad).
    If not given, they are estimated from the residuals of the q fits and from
    the polarization scatter across the sub-array (pooled over all sources).
    """
    z = survey.depths
    idx = np.sort(np.argsort(np.abs(z - level))[:n_receivers])
    zr = z[idx]
    nr = idx.size
    im = nr // 2
    x = survey.offsets
    ns = x.size
    t = survey.time[azimuth][:, idx]
    psi = survey.psi[azimuth][:, idx]
    sgn = np.sign(x)
    # q: one slope across the sub-array per source
    bq = _slope_weights(zr)
    q = t @ bq
    # p: slope across neighbouring sources at the middle receiver; P is the operator
    k = n_sources // 2
    order = np.argsort(x)
    P = np.zeros((ns, ns))
    for jj in range(k, ns - k):
        w = order[jj - k: jj + k + 1]
        if np.all(sgn[w] == sgn[order[jj]]) and np.all(np.diff(x[w]) > 0):
            P[order[jj], w] = _slope_weights(x[w])
    has_p = np.any(P != 0.0, axis=1)
    tm = t[:, im]
    p = P @ np.where(np.isfinite(tm), tm, 0.0)
    # a missing pick (no direct ray, e.g. beyond a critical offset) voids the windows using it
    p[(P != 0.0).astype(float) @ (~np.isfinite(tm)).astype(float) > 0] = np.nan
    full = np.all(np.isfinite(t), axis=1) & np.all(np.isfinite(psi), axis=1)
    if time_noise is None:
        dz = zr - zr.mean()
        res = (t - t.mean(axis=1, keepdims=True) - q[:, None] * dz)[full]
        time_noise = float(np.sqrt((res * res).sum() / (full.sum() * max(nr - 2, 1))))
    psi_s = psi * sgn[:, None]
    if psi_noise is None:
        dpsi = (psi_s - psi_s.mean(axis=1, keepdims=True))[full]
        psi_noise = float(np.sqrt((dpsi * dpsi).sum() / (full.sum() * max(nr - 1, 1))))
    # covariance of (|p|, q, |psi|): |p| = sgn p, and p_j shares trace (l, middle) with q_l
    S = sgn[:, None] * P
    s2 = time_noise ** 2
    C = np.zeros((3 * ns, 3 * ns))
    C[:ns, :ns] = s2 * S @ S.T
    C[:ns, ns:2 * ns] = s2 * S * bq[im]
    C[ns:2 * ns, :ns] = C[:ns, ns:2 * ns].T
    C[ns:2 * ns, ns:2 * ns] = s2 * (bq @ bq) * np.eye(ns)
    C[2 * ns:, 2 * ns:] = psi_noise ** 2 / nr * np.eye(ns)
    sd = np.sqrt(np.diag(C)).reshape(3, ns)
    data = SlownessData(np.abs(p), q, np.abs(psi_s.mean(axis=1)), *sd, float(zr[im]),
                        float(zr[0]), float(zr[-1]), C)
    return data.select(has_p & np.isfinite(p) & full)


class Posterior(NamedTuple):
    """Gaussian (Laplace) posterior of :data:`PARAMETERS`."""

    mean: np.ndarray
    cov: np.ndarray
    chi2: float
    n_data: int
    prior_std: np.ndarray
    method: str

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def corr(self) -> np.ndarray:
        s = self.std
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.cov / np.outer(s, s)

    @property
    def data_weight(self) -> np.ndarray:
        """Share of the information on each parameter that comes from the data.

        ``1 - posterior variance / prior variance``: 0 means the estimate is the
        prior, 1 that the data alone determine it (also for parameters without a
        prior).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(self.prior_std),
                            1.0 - np.diag(self.cov) / self.prior_std ** 2, 1.0)

    def __getitem__(self, name):          # posterior["eps"] -> (mean, std)
        if isinstance(name, str):
            i = PARAMETERS.index(name)
            return self.mean[i], self.std[i]
        return tuple.__getitem__(self, name)


def _forward(x, theta, method: str = "exact"):
    """Phase velocity and polarization at phase angles ``theta`` for x = (vp0, vs0, eps, delta)."""
    if method == "weak":
        vp0, _, eps, delta = x
        s2 = np.sin(theta) ** 2
        return vp0 * (1.0 + delta * s2 * (1.0 - s2) + eps * s2 * s2), None
    a = velocity_stiffness(*x)
    v = qp_phase_velocity(theta, *a)
    return v, qp_polarization(np.sin(theta) / v, np.cos(theta) / v, *a)


def _residuals(x, data: SlownessData, use_polarization: bool, method: str, whiten=None):
    """Data residuals and the lower Cholesky factor of their covariance.

    Residuals are ``V_obs - V(theta_obs)`` and ``psi_obs - psi(theta_obs)``
    with ``V_obs = 1/|(p, q)|``, ``theta_obs = atan(p/q)``.  Their covariance
    is the measurement covariance mapped through the linearization in
    (p, q, psi), which includes the error of the observed angle (effective
    variance).  ``whiten`` reuses a given factor.
    """
    p, q = data.p, data.q
    pol = use_polarization and method == "exact"

    def f(pp, qq):
        v, psi = _forward(x, np.arctan2(pp, qq), method)
        out = [1.0 / np.hypot(pp, qq) - v]
        if pol:
            out.append(data.psi - psi)
        return np.stack(out)

    r = f(p, q)
    if whiten is None:
        h = 1e-4 * np.hypot(p, q)
        dp = (f(p + h, q) - f(p - h, q)) / (2 * h)
        dq = (f(p, q + h) - f(p, q - h)) / (2 * h)
        n = p.size
        C = data.covariance()
        blocks = [[C[a * n:(a + 1) * n, c * n:(c + 1) * n] for c in range(3)] for a in range(3)]
        # sensitivity of each residual to (p, q, psi); psi_obs enters only its own residual
        rows = [(dp[k], dq[k], np.full(n, 1.0 if k else 0.0)) for k in range(r.shape[0])]
        R = np.block([[sum(np.outer(rk[a], rl[c]) * blocks[a][c]
                           for a in range(3) for c in range(3)) for rl in rows] for rk in rows])
        whiten = np.linalg.cholesky(R)
    return r.ravel(), whiten


def invert_level(data: SlownessData, prior: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_polarization: bool = True, method: str = "exact",
                 x0=None, max_iter: int = 60) -> Posterior:
    """MAP estimate and Laplace posterior of (Vp0, Vs0, eps, delta) at one level.

    The measured (p, q) give a phase velocity ``V = 1/|(p, q)|`` at a phase angle
    ``theta = atan(p/q)`` (Gaiser, 1990; Miller & Spencer, 1994).
    ``method="exact"`` fits them (and, with ``use_polarization``, the
    polarizations) with the exact VTI Christoffel equation; ``method="weak"``
    with Thomsen's (1986) weak-anisotropy phase velocity (slowness only; Vs0
    then only enters through its prior).  The residual covariance is the full
    measurement covariance (:func:`measure_slowness`) mapped through the
    linearization, re-evaluated at each iteration.  ``prior`` maps parameter
    names to ``(mean, std)``.

    The parameters are kept physical (0 < Vs0 < Vp0, |eps|, |delta| < 1, real
    C13).  Without near angles Vp0 trades off with delta, and with
    polarizations eps and delta depend on Vs0: give priors on both (ZO-VSP Vp0,
    dipole Vs0) and check how much the data move them
    (:attr:`Posterior.data_weight`).
    """
    from scipy.linalg import solve_triangular

    if method not in ("exact", "weak"):
        raise ValueError("method must be 'exact' or 'weak'")
    prior = prior or {}
    mu = np.array([prior.get(k, (np.nan, np.inf))[0] for k in PARAMETERS])
    sd = np.array([prior.get(k, (np.nan, np.inf))[1] for k in PARAMETERS], dtype=float)
    has = np.isfinite(sd)
    if x0 is None:
        v, th = data.velocity, data.theta
        vp0 = mu[0] if np.isfinite(mu[0]) else float(np.median(v[th <= np.median(th)]))
        vs0 = mu[1] if np.isfinite(mu[1]) else 0.5 * vp0
        x0 = [vp0, vs0, mu[2] if np.isfinite(mu[2]) else 0.05,
              mu[3] if np.isfinite(mu[3]) else 0.02]
    x = np.array(x0, dtype=float)
    scale = np.array([x[0], x[1], 1.0, 1.0])
    n_res = data.p.size * (2 if (method == "exact" and use_polarization) else 1) + int(has.sum())

    def residuals(xx, whiten=None):
        """Whitened residuals (data and prior) and the whitening factor; inf outside bounds."""
        if not (0.0 < xx[1] < xx[0]) or abs(xx[2]) > 1.0 or abs(xx[3]) > 1.0:
            return np.full(n_res, np.inf), whiten
        with np.errstate(invalid="ignore"):
            r, W = _residuals(xx, data, use_polarization, method, whiten)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(W)):
            return np.full(n_res, np.inf), whiten
        rw = solve_triangular(W, r, lower=True)
        return np.concatenate([rw, ((xx - mu) / sd)[has]]), W

    def cost_of(r):
        return float(r @ r) if np.all(np.isfinite(r)) else np.inf

    def jac(xx):
        r0, L = residuals(xx)
        J = np.empty((r0.size, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = 1e-6 * scale[j]
            rp, rm = residuals(xx + e, L)[0], residuals(xx - e, L)[0]
            if np.all(np.isfinite(rp)) and np.all(np.isfinite(rm)):
                J[:, j] = (rp - rm) / (2 * e[j])
            else:                       # at a bound: one-sided difference
                J[:, j] = (rp - r0) / e[j] if np.all(np.isfinite(rp)) else (r0 - rm) / e[j]
        return r0, J, L

    # Levenberg-Marquardt with the residual covariance frozen at the current
    # iterate (iterated effective variance).  Letting it vary inside the misfit
    # would reward models that inflate the errors, i.e. biases towards stronger
    # anisotropy.
    lam = 1e-3
    r, J, L = jac(x)
    cost = cost_of(r)
    for _ in range(max_iter):
        A = J.T @ J
        if not np.all(np.isfinite(A)):
            break
        g = J.T @ r
        D = np.diag(np.diag(A)) + 1e-30
        try:
            step = -np.linalg.solve(A + lam * D, g)
        except np.linalg.LinAlgError:
            break
        xn = x + step
        cn = cost_of(residuals(xn, L)[0])
        if cn < cost:
            converged = (cost - cn < 1e-10 * max(cost, 1.0)
                         or np.all(np.abs(step) < 1e-9 * scale))
            x = xn
            r, J, L = jac(x)
            cost = cost_of(r)
            lam = max(lam / 5.0, 1e-9)
            if converged:
                break
        else:
            lam *= 10.0
            if lam > 1e10:
                break
    # parameters the data and priors do not constrain (Vs0 in the weak fit without a
    # prior) get an infinite variance instead of making the system singular
    A = J.T @ J
    if not np.all(np.isfinite(A)):
        raise RuntimeError("walkaway inversion ended outside the physical bounds")
    free = np.diag(A) > 1e-12 * np.max(np.diag(A))
    cov = np.zeros((4, 4))
    cov[np.ix_(free, free)] = np.linalg.inv(A[np.ix_(free, free)])
    cov[~free, ~free] = np.inf
    n_data = int(r.size - has.sum())
    return Posterior(x, cov, cost, n_data, sd, method)


def azimuthal_consistency(posteriors: Sequence[Posterior],
                          names: Sequence[str] = ("eps", "delta")):
    """Chi-square test that estimates from several azimuths agree (VTI hypothesis).

    Returns ``(chi2, dof, p_value)`` for the deviations of each azimuth's
    (eps, delta) from their precision-weighted mean.  A small p-value flags
    azimuthal anisotropy (or another inconsistency) that a VTI fit would hide.
    """
    from scipy.stats import chi2 as chi2_dist

    idx = [PARAMETERS.index(n) for n in names]
    if not all(np.all(np.isfinite(post.cov[np.ix_(idx, idx)])) for post in posteriors):
        raise ValueError("every posterior needs a finite covariance of " + ", ".join(names))
    xs = [post.mean[idx] for post in posteriors]
    Ws = [np.linalg.inv(post.cov[np.ix_(idx, idx)]) for post in posteriors]
    W = sum(Ws)
    xbar = np.linalg.solve(W, sum(w @ x for w, x in zip(Ws, xs)))
    stat = float(sum((x - xbar) @ w @ (x - xbar) for w, x in zip(Ws, xs)))
    dof = len(idx) * (len(posteriors) - 1)
    return stat, dof, float(chi2_dist.sf(stat, dof))


def polarization_check(data: SlownessData, posterior: Posterior):
    """Polarization misfit of a slowness-only fit: ``(chi2 / n, p_value)``.

    The polarizations predicted by the fitted model at the measured phase
    angles are compared with the measured ones, with the full covariance of
    the measurements and of the prediction (posterior uncertainty of the
    model).  A source-side bias in p (lateral heterogeneity) moves the (p, q)
    pairs off the true slowness surface but leaves q and the polarization
    untouched, so the polarizations disagree with the slowness fit.
    """
    from scipy.linalg import solve_triangular
    from scipy.stats import chi2 as chi2_dist

    n = data.p.size
    x = posterior.mean
    r, L = _residuals(x, data, True, "exact")
    R = (L @ L.T)[n:, n:]                  # measurement part of the polarization residuals
    ok = np.all(np.isfinite(posterior.cov), axis=0)
    G = np.zeros((n, 4))
    for j in np.flatnonzero(ok):           # d psi_model / d parameter
        e = np.zeros(4)
        e[j] = 1e-6 * max(abs(x[j]), 1.0)
        G[:, j] = (_forward(x + e, data.theta)[1] - _forward(x - e, data.theta)[1]) / (2 * e[j])
    cov = np.where(np.outer(ok, ok), posterior.cov, 0.0)
    R = R + G @ cov @ G.T
    z = solve_triangular(np.linalg.cholesky(R), r[n:], lower=True)
    stat = float(z @ z)
    return stat / n, float(chi2_dist.sf(stat, n))
