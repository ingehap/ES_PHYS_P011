"""Minimal rock physics: Gassmann (1951) fluid substitution and fluid mixing.

Used by :mod:`iwt.synthetic` to derive the oil leg of the Stø Formation from its
water leg, so that the two share one rock frame.  Gassmann's relations hold for an
isotropic, homogeneous, fully connected pore space at low frequency; the shear
modulus does not depend on the pore fluid, so Vs changes only through density.

References
----------
Gassmann, F., 1951, Über die Elastizität poröser Medien: Vierteljahrsschrift der
Naturforschenden Gesellschaft in Zürich, 96, 1-23.
Mavko, G., Mukerji, T., & Dvorkin, J., 2009, The Rock Physics Handbook (2nd ed.):
Cambridge University Press.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gassmann_dry", "gassmann_saturated", "fluid_substitution", "reuss_fluid"]


def gassmann_dry(k_sat, k_min, k_fl, phi):
    """Dry-frame bulk modulus from the saturated one (inverse Gassmann)."""
    k_sat, k_min, k_fl, phi = (np.asarray(a, dtype=float) for a in (k_sat, k_min, k_fl, phi))
    a = phi * k_min / k_fl
    return (k_sat * (a + 1.0 - phi) - k_min) / (a + k_sat / k_min - 1.0 - phi)


def gassmann_saturated(k_dry, k_min, k_fl, phi):
    """Saturated bulk modulus from the dry-frame one (Gassmann, 1951)."""
    k_dry, k_min, k_fl, phi = (np.asarray(a, dtype=float) for a in (k_dry, k_min, k_fl, phi))
    b = 1.0 - k_dry / k_min
    return k_dry + b**2 / (phi / k_fl + (1.0 - phi) / k_min - k_dry / k_min**2)


def reuss_fluid(moduli, densities, saturations):
    """Bulk modulus (Reuss/Wood average) and density of a mixed pore fluid."""
    k = np.asarray(moduli, dtype=float)
    r = np.asarray(densities, dtype=float)
    s = np.asarray(saturations, dtype=float)
    if not np.isclose(s.sum(), 1.0):
        raise ValueError("saturations must sum to 1")
    return 1.0 / np.sum(s / k), float(np.sum(s * r))


def fluid_substitution(vp, vs, rho, phi, k_min, k_fl1, rho_fl1, k_fl2, rho_fl2):
    """Replace pore fluid 1 by fluid 2.  Returns ``(vp2, vs2, rho2)`` in SI units."""
    vp, vs, rho = (np.asarray(a, dtype=float) for a in (vp, vs, rho))
    mu = rho * vs**2
    k1 = rho * vp**2 - 4.0 / 3.0 * mu
    k_dry = gassmann_dry(k1, k_min, k_fl1, phi)
    if np.any(k_dry <= 0.0) or np.any(k_dry >= k_min):
        raise ValueError("implied dry-frame modulus outside (0, k_min)")
    k2 = gassmann_saturated(k_dry, k_min, k_fl2, phi)
    rho2 = rho + phi * (rho_fl2 - rho_fl1)
    return np.sqrt((k2 + 4.0 / 3.0 * mu) / rho2), np.sqrt(mu / rho2), rho2
