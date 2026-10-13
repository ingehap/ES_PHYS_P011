"""Constant-Q attenuation and velocity dispersion (Kjartansson, 1979).

In Kjartansson's model Q is exactly independent of frequency.  The phase
velocity then follows a power law,

    c(f) = c0 (f / f0)^g,      g = arctan(1/Q) / pi,

so a velocity measured at sonic frequencies (kHz) is higher than the same
formation's velocity at seismic frequencies (tens of Hz).  This dispersion is one
cause of the sonic-to-VSP traveltime "drift" (Stewart et al., 1984).

The amplitude attenuation coefficient is exactly ``alpha = omega tan(pi g / 2) / c``,
which for large Q tends to ``pi f / (Q c)``.

References
----------
Kjartansson, E., 1979, Constant Q-wave propagation and attenuation: JGR, 84(B9),
4737-4748. https://doi.org/10.1029/JB084iB09p04737
Stewart, R. R., Huddleston, P. D., & Kan, T. K., 1984, Seismic versus sonic
velocities: A vertical seismic profiling study: Geophysics, 49(8), 1153-1168.
https://doi.org/10.1190/1.1441745
"""

from __future__ import annotations

import numpy as np

__all__ = ["dispersion_exponent", "phase_velocity", "attenuation_coefficient", "t_star"]


def dispersion_exponent(q):
    """Kjartansson's exponent g = arctan(1/Q)/pi (0 for Q = inf)."""
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore"):
        return np.arctan(1.0 / q) / np.pi


def phase_velocity(c0, q, f, f0):
    """Phase velocity at frequency ``f`` given ``c0`` at reference frequency ``f0``."""
    f = np.asarray(f, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    if np.any(f <= 0.0) or np.any(f0 <= 0.0):
        raise ValueError("frequencies must be positive")
    return np.asarray(c0, dtype=float) * (f / f0) ** dispersion_exponent(q)


def attenuation_coefficient(c0, q, f, f0):
    """Amplitude attenuation coefficient alpha(f) (1/m), exact for constant Q."""
    g = dispersion_exponent(q)
    c = phase_velocity(c0, q, f, f0)
    return 2.0 * np.pi * np.asarray(f, dtype=float) * np.tan(0.5 * np.pi * g) / c


def t_star(thickness, velocity, q):
    """Attenuated traveltime t* = sum(h / (v Q)) (s) along a vertical path."""
    h = np.asarray(thickness, dtype=float)
    v = np.asarray(velocity, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore"):
        return float(np.sum(h / v * (1.0 / q)))
