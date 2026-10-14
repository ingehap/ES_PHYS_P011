"""Sonic-to-seismic calibration: the drift curve and two ways to remove it.

The sonic log measures velocity at kHz frequencies over a ~0.6 m aperture; the
VSP measures traveltime at tens of Hz over wavelengths of tens of metres.  The
difference of the integrated sonic time and the VSP time is the *drift*
(positive when the sonic time is shorter).  It has three physical causes and
any number of measurement ones:

1. **intrinsic dispersion**: with constant Q the phase velocity rises with
   frequency as ``c(f) = c0 (f / f0)^g``, ``g = arctan(1/Q) / pi``
   (Kjartansson, 1979; Stewart et al., 1984);
2. **layering (scale)**: a seismic wave much longer than the layering travels at
   the Backus (1962) average velocity, a sonic wave at the time average, and
   the Backus velocity is the lower one (Marion et al., 1994; Rio et al., 1996).
   This is the velocity side of the stratigraphic filtering that also gives an
   apparent Q (O'Doherty & Anstey, 1971);
3. **anisotropy**: in a deviated well the sonic travels along the borehole, at
   an angle to the VTI symmetry axis (zero in the vertical well modelled here);
4. picking convention, borehole damage, log editing, noise.

Two calibrations are provided:

* :func:`knee_calibration`, the classic one: a continuous piecewise-linear
  drift with knees at chosen depths, removed by a slowness shift (or scaling)
  in each segment (Goetz et al., 1979; White & Simm, 2003).  It removes all
  causes at once and absorbs the attenuation into the velocity model, so a
  later Q compensation counts it a second time;
* :func:`physical_calibration`: the sonic velocities dispersed to the seismic
  frequency with the *intrinsic* VSP Q, Backus averaged over a seismic
  wavelength, and (for deviated wells) rotated to the vertical with the Thomsen
  parameters.  What remains is the drift that physics does not explain, which is
  the quantity to inspect (:func:`drift`).

References
----------
Backus, G. E., 1962, Long-wave elastic anisotropy produced by horizontal
layering: JGR, 67(11), 4427-4440. https://doi.org/10.1029/JZ067i011p04427
Goetz, J. F., Dupal, L., & Bowler, J., 1979, An investigation into discrepancies
between sonic log and seismic check shot velocities: APEA Journal, 19(1),
131-141. https://doi.org/10.1071/AJ78014
Kjartansson, E., 1979, Constant Q-wave propagation and attenuation: JGR, 84(B9),
4737-4748. https://doi.org/10.1029/JB084iB09p04737
Marion, D., Mukerji, T., & Mavko, G., 1994, Scale effects on velocity dispersion:
From ray to effective medium theories in stratified media: Geophysics, 59(10),
1613-1619. https://doi.org/10.1190/1.1443550
Rio, P., Mukerji, T., Mavko, G., & Marion, D., 1996, Velocity dispersion and
upscaling in a laboratory-simulated VSP: Geophysics, 61(2), 584-593.
https://doi.org/10.1190/1.1443984
Stewart, R. R., Huddleston, P. D., & Kan, T. K., 1984, Seismic versus sonic
velocities: A vertical seismic profiling study: Geophysics, 49(8), 1153-1168.
https://doi.org/10.1190/1.1441745
Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954-1966.
https://doi.org/10.1190/1.1442051
White, R. E., & Simm, R., 2003, Tutorial: Good practice in well ties: First
Break, 21(10). https://doi.org/10.3997/1365-2397.21.10.25640
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

from .qmodel import phase_velocity

__all__ = [
    "sonic_time",
    "drift",
    "DriftCalibration",
    "knee_calibration",
    "backus_vertical_velocity",
    "physical_calibration",
]


def sonic_time(depth, velocity, z=None) -> np.ndarray:
    """One-way vertical time (s) from the top of the log to ``z`` (default: every log depth).

    The slowness is integrated with the trapezoidal rule on the log samples;
    depths outside the log give NaN.
    """
    d = np.asarray(depth, dtype=float)
    s = 1.0 / np.asarray(velocity, dtype=float)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(d))])
    if z is None:
        return t
    z = np.asarray(z, dtype=float)
    return np.where((z >= d[0]) & (z <= d[-1]), np.interp(z, d, t), np.nan)


def drift(depth, velocity, z_vsp, t_vsp, z_tie: Optional[float] = None) -> np.ndarray:
    """Drift ``(t_vsp - t_vsp(z_tie)) - (t_sonic - t_sonic(z_tie))`` (s) at the VSP depths.

    ``t_vsp`` are vertical one-way times; ``z_tie`` defaults to the shallowest VSP
    depth inside the log.  Positive drift: the sonic is faster than the VSP.
    NaN outside the log.
    """
    zv = np.asarray(z_vsp, dtype=float)
    tv = np.asarray(t_vsp, dtype=float)
    d = np.asarray(depth, dtype=float)
    if z_tie is None:
        inside = zv[(zv >= d[0]) & (zv <= d[-1])]
        if inside.size == 0:
            raise ValueError("no VSP depth inside the log")
        z_tie = float(inside.min())
    order = np.argsort(zv)
    t_tie = np.interp(z_tie, zv[order], tv[order])
    ts = sonic_time(d, velocity, zv)
    return (tv - t_tie) - (ts - sonic_time(d, velocity, z_tie))


class DriftCalibration(NamedTuple):
    """A calibrated log and the drift it was calibrated to."""

    depth: np.ndarray
    velocity: np.ndarray
    nodes: np.ndarray
    node_drift: np.ndarray
    drift_fit: np.ndarray
    residual: np.ndarray


def knee_calibration(depth, velocity, z_vsp, t_vsp, knees: Sequence[float] = (),
                     mode: str = "shift", z_tie: Optional[float] = None) -> DriftCalibration:
    """Classic drift correction: piecewise-linear drift with knees, removed per segment.

    The drift at the VSP depths is fitted by least squares with a continuous
    piecewise-linear function that is zero at ``z_tie`` and has nodes there, at
    the ``knees`` and at the deepest VSP level in the log.  In each segment the
    drift gradient ``g = d(drift)/dz`` (s/m) is removed from the log:
    ``mode="shift"`` adds g to the slowness (the "block shift"), ``mode="scale"``
    multiplies the slowness by ``1 + g / mean slowness`` of the segment.  Below
    the last node and above the tie the gradient of the nearest segment is
    used.  Returns the calibrated log, the nodes, the fitted drift at the VSP
    depths and the residual drift after calibration.
    """
    d = np.asarray(depth, dtype=float)
    v = np.asarray(velocity, dtype=float)
    zv = np.asarray(z_vsp, dtype=float)
    dr = drift(d, v, zv, t_vsp, z_tie)
    ok = np.isfinite(dr)
    if z_tie is None:
        z_tie = float(zv[ok].min())
    nodes = np.unique(np.concatenate([[z_tie], np.asarray(knees, float), [zv[ok].max()]]))
    nodes = nodes[(nodes >= z_tie) & (nodes <= zv[ok].max())]
    # hat-function basis, without the first node (drift fixed to 0 at the tie)
    B = np.column_stack([np.interp(zv[ok], nodes, np.eye(nodes.size)[i])
                         for i in range(1, nodes.size)])
    c, *_ = np.linalg.lstsq(B, dr[ok], rcond=None)
    node_drift = np.concatenate([[0.0], c])
    grad = np.diff(node_drift) / np.diff(nodes)
    seg = np.clip(np.searchsorted(nodes, d, side="right") - 1, 0, grad.size - 1)
    s = 1.0 / v
    if mode == "shift":
        s_new = s + grad[seg]
    elif mode == "scale":
        s_mean = np.array([s[(d >= a) & (d <= b)].mean() for a, b in zip(nodes[:-1], nodes[1:])])
        s_new = s * (1.0 + grad[seg] / s_mean[seg])
    else:
        raise ValueError("mode must be 'shift' or 'scale'")
    fit = np.full(zv.shape, np.nan)
    fit[ok] = np.interp(zv[ok], nodes, node_drift)
    resid = drift(d, 1.0 / s_new, zv, t_vsp, z_tie)
    return DriftCalibration(d, 1.0 / s_new, nodes, node_drift, fit, resid)


def _running_mean(x, n: int) -> np.ndarray:
    """Centred running mean over ``n`` samples, shortened at the ends."""
    if n <= 1:
        return np.asarray(x, dtype=float)
    k = np.ones(n)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x), k, mode="same")
    return num / den


def backus_vertical_velocity(depth, vp, rho, length: float) -> np.ndarray:
    """Long-wavelength vertical P velocity of the layering (Backus, 1962).

    ``sqrt(<1 / (rho vp^2)>^-1 / <rho>)`` with running averages over ``length``
    (m) on the regular log grid.  The effective vertical modulus of a layered
    stack is the harmonic average of C33 whatever the anisotropy of the layers,
    so the shear velocity is not needed.
    """
    d = np.asarray(depth, dtype=float)
    vp = np.asarray(vp, dtype=float)
    rho = np.asarray(rho, dtype=float)
    dz = float(np.median(np.diff(d)))
    n = max(1, int(round(length / dz)) | 1)
    m33 = 1.0 / _running_mean(1.0 / (rho * vp ** 2), n)
    return np.sqrt(m33 / _running_mean(rho, n))


def physical_calibration(depth, vp, rho, q, f_sonic: float, f_seismic: float,
                         backus_length: Optional[float] = None,
                         inclination_deg: float = 0.0, eps=0.0, delta=0.0) -> np.ndarray:
    """Vertical P velocity at ``f_seismic`` predicted from the sonic ``vp`` (at ``f_sonic``).

    1. Anisotropy (deviated wells): the sonic phase velocity at
       ``inclination_deg`` from vertical is converted to the vertical velocity with
       Thomsen's (1986) weak-anisotropy relation.
    2. Dispersion: each sample is moved from ``f_sonic`` to ``f_seismic`` with the
       Kjartansson law for the given ``q`` (array or scalar; use the *intrinsic*
       Q, see :mod:`iwt.vsp.attenuation`).
    3. Scale: with ``backus_length`` (m, about a quarter of the seismic
       wavelength or more), the Backus average replaces the time average.

    The VSP time predicted by the result is :func:`sonic_time` of it.
    """
    vp = np.asarray(vp, dtype=float)
    th = np.radians(inclination_deg)
    s2, c2 = np.sin(th) ** 2, np.cos(th) ** 2
    v0 = vp / (1.0 + np.asarray(delta) * s2 * c2 + np.asarray(eps) * s2 * s2)
    v = phase_velocity(v0, q, f_seismic, f_sonic)
    if backus_length:
        v = backus_vertical_velocity(depth, v, rho, backus_length)
    return v
