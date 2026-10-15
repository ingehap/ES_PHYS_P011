"""P-wave reflection coefficients: exact (isotropic and VTI) and approximations.

All functions take the incidence angle ``theta`` (radians) in the upper medium and
the properties of the upper (1) and lower (2) media, and return the PP
displacement reflection coefficient.  The sign convention is that of Aki &
Richards (1980): at normal incidence ``R = (Z2 - Z1) / (Z2 + Z1)``.

* :func:`zoeppritz_pp`: exact, isotropic (Aki & Richards, 1980, explicit form);
* :func:`exact_pp`: exact, VTI, from the layer eigenvectors (:mod:`.elastic`);
* :func:`aki_richards`: linearized, isotropic (Aki & Richards, 1980);
* :func:`shuey`: three-term form (Shuey, 1985);
* :func:`ruger`: VTI linearization (Rüger, 1997): Shuey's terms plus
  ``(Δδ/2) sin^2θ + (Δε/2) sin^2θ tan^2θ``.

The linearized forms use the incidence angle in place of the average of the
incidence and transmission angles, as is usual for AVO work; this is accurate for
small contrasts.  Vp, Vs refer to the vertical (symmetry-axis) velocities in the
VTI forms.

References
----------
Aki, K., & Richards, P. G., 1980, Quantitative Seismology: W. H. Freeman.
Rüger, A., 1997, P-wave reflection coefficients for transversely isotropic models
with vertical and horizontal axis of symmetry: Geophysics, 62(3), 713-722.
https://doi.org/10.1190/1.1444181
Shuey, R. T., 1985, A simplification of the Zoeppritz equations: Geophysics, 50(4),
609-614. https://doi.org/10.1190/1.1441936
"""

from __future__ import annotations

import numpy as np

from ..model import phase_velocities, thomsen_to_cij
from .elastic import eigenvectors, interface_scattering, vertical_slowness

__all__ = ["zoeppritz_pp", "exact_pp", "exact_pp_slowness", "aki_richards", "shuey", "ruger",
           "slowness_from_angle"]


def zoeppritz_pp(theta, vp1, vs1, rho1, vp2, vs2, rho2):
    """Exact isotropic PP reflection coefficient (complex beyond critical angle)."""
    theta = np.asarray(theta, dtype=float)
    p = np.sin(theta) / vp1
    ci1 = np.cos(theta) + 0j
    ci2 = np.sqrt(1.0 - (p * vp2) ** 2 + 0j)
    cj1 = np.sqrt(1.0 - (p * vs1) ** 2 + 0j)
    cj2 = np.sqrt(1.0 - (p * vs2) ** 2 + 0j)
    a = rho2 * (1.0 - 2.0 * vs2**2 * p**2) - rho1 * (1.0 - 2.0 * vs1**2 * p**2)
    b = rho2 * (1.0 - 2.0 * vs2**2 * p**2) + 2.0 * rho1 * vs1**2 * p**2
    c = rho1 * (1.0 - 2.0 * vs1**2 * p**2) + 2.0 * rho2 * vs2**2 * p**2
    d = 2.0 * (rho2 * vs2**2 - rho1 * vs1**2)
    E = b * ci1 / vp1 + c * ci2 / vp2
    F = b * cj1 / vs1 + c * cj2 / vs2
    G = a - d * ci1 / vp1 * cj2 / vs2
    H = a - d * ci2 / vp2 * cj1 / vs1
    D = E * F + G * H * p**2
    return ((b * ci1 / vp1 - c * ci2 / vp2) * F - (a + d * ci1 / vp1 * cj2 / vs2) * H * p**2) / D


def slowness_from_angle(theta, vp0, vs0, eps, delta):
    """Horizontal slowness of a qP wave with phase angle ``theta`` in a VTI medium."""
    vp, _, _ = phase_velocities(theta, vp0, vs0, eps, delta, 0.0)
    return np.sin(theta) / vp


def exact_pp(theta, upper, lower):
    """Exact PP coefficient for two VTI half-spaces.

    ``upper`` and ``lower`` are tuples ``(vp0, vs0, rho, eps, delta)``; ``theta`` is
    the phase angle of the incident qP wave.  Both media must be solids.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return exact_pp_slowness(slowness_from_angle(theta, upper[0], upper[1], upper[3], upper[4]),
                             upper, lower)


def exact_pp_slowness(p, upper, lower):
    """:func:`exact_pp` as a function of the horizontal slowness ``p`` (s/m).

    Beyond the critical slowness of either medium the coefficient is complex
    (``exp(-i omega t)`` convention); beyond the upper medium's own horizontal
    slowness the incident wave is evanescent and the value is formal.
    """
    p = np.atleast_1d(np.asarray(p, dtype=float)) + 0j
    Ds = []
    for vp0, vs0, rho, eps, delta in (upper, lower):
        c = thomsen_to_cij(vp0, vs0, rho, eps, delta, 0.0)
        qp, qs = vertical_slowness(c.c11 + 0j, c.c13 + 0j, c.c33 + 0j, c.c44 + 0j, rho, p)
        Ds.append(eigenvectors(c.c11, c.c13, c.c33, c.c44, rho, p, qp, qs))
    scale = 0.5 * (upper[2] * upper[0] + lower[2] * lower[0])
    RD, _, _, _ = interface_scattering(Ds[0], Ds[1], scale=scale)
    return RD[..., 0, 0]


def aki_richards(theta, vp1, vs1, rho1, vp2, vs2, rho2):
    """Linearized isotropic PP coefficient (Aki & Richards, 1980)."""
    theta = np.asarray(theta, dtype=float)
    vp, vs, rho = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1
    p = np.sin(theta) / vp
    return (0.5 * (1.0 - 4.0 * vs**2 * p**2) * drho / rho
            + dvp / (2.0 * np.cos(theta) ** 2 * vp)
            - 4.0 * vs**2 * p**2 * dvs / vs)


def shuey(theta, vp1, vs1, rho1, vp2, vs2, rho2):
    """Three-term Shuey (1985) form: ``A + B sin^2θ + C (tan^2θ - sin^2θ)``."""
    theta = np.asarray(theta, dtype=float)
    vp, vs, rho = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1
    A = 0.5 * (dvp / vp + drho / rho)
    B = 0.5 * dvp / vp - 2.0 * (vs / vp) ** 2 * (drho / rho + 2.0 * dvs / vs)
    C = 0.5 * dvp / vp
    s2 = np.sin(theta) ** 2
    return A + B * s2 + C * (np.tan(theta) ** 2 - s2)


def ruger(theta, upper, lower):
    """Rüger (1997) VTI linearization; media as ``(vp0, vs0, rho, eps, delta)``."""
    theta = np.asarray(theta, dtype=float)
    iso = shuey(theta, upper[0], upper[1], upper[2], lower[0], lower[1], lower[2])
    s2 = np.sin(theta) ** 2
    d_delta = lower[4] - upper[4]
    d_eps = lower[3] - upper[3]
    return iso + 0.5 * d_delta * s2 + 0.5 * d_eps * s2 * np.tan(theta) ** 2
