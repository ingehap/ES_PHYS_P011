"""Sonic anisotropy in a vertical well: C33, C44, C66 and Thomsen's gamma.

In a vertical well through a VTI formation (Walsh et al., 2007):

* the monopole P wave travels along the symmetry axis, ``C33 = rho Vp^2``;
* the dipole shear waves travel along the axis, ``C44 = rho Vs^2``;
* the low-frequency Stoneley (tube) wave senses an effective shear modulus
  (Norris & Sinha, 1993), which for a borehole along the axis of a VTI
  formation is ``C66``: the borehole wall expands in plane strain in the
  isotropic plane;
* ``gamma = (C66 - C44) / (2 C44)``.

**The tube-wave relation.**  At zero frequency the Stoneley wave is a tube wave
whose slowness follows from the compliance of the fluid annulus.  For a
borehole of radius b with a concentric tool of radius a and area modulus K_t
(``lambda + mu`` of a solid rod in plane strain; inf for a rigid tool),

    S_T^2 = S_f^2 + rho_f (b^2 / C66 + a^2 / K_t) / (b^2 - a^2),

with S_f and rho_f the mud slowness and density.  Without a tool this is the
classic ``S_f^2 + rho_f / mu`` (White, 1983).  A rigid tool of 46 mm radius in
a 108 mm borehole raises the C66 term by 22 %, so the tool term is not small.
Norris (1990) treats the tool, casing and eccentricity; an eccentred tool or
borehole lowers the tube-wave speed further, which is not modelled here.

**Dispersion.**  Sonic tools measure the Stoneley wave at 0.5-3 kHz, where it is
no longer a tube wave.  :func:`stoneley_slowness` solves the exact modal
equation of a fluid-filled borehole, with an optional concentric tool, along the
axis of a TI formation (Biot, 1952, and Cheng & Toksoz, 1981, for the isotropic
case; White & Tongtaow, 1981, for the TI fields).  The dispersion depends on
C11 and C13 as well, which a vertical well does not measure, so
:func:`sonic_anisotropy` assumes Thomsen eps and delta (default 0).  In the slow
Wisting-like shales the 1 kHz slowness is about 3 % above the tube-wave value,
which biases C66 by about -10 % if ignored.  Where the tube wave (set by C66)
is faster than the shear wave along the borehole (set by C44), the Stoneley
wave radiates shear energy into the formation and is leaky; this happens in the
slow, strongly anisotropic Wisting-like overburden shales.  An "equivalent
isotropic" formation with shear modulus C66, a common shortcut (cf. Tang,
2003), misses this and gets the dispersion wrong there.

**Uncertainty.**  C66 depends on the *difference* ``S_T^2 - S_f^2``, so errors
in the mud slowness are amplified, most strongly in fast formations.  The mud
slowness is not measured in the hole at sonic frequencies (Valero et al.,
2009); :func:`calibrate_mud_slowness` fits it in an interval of assumed gamma.
Permeability raises the Stoneley slowness in sands (Winkler et al., 1989), which
is not modelled here and would bias both C66 and such a calibration.
:func:`gamma_jacobian` gives the exact derivatives of gamma with respect to
every input, for linear error propagation.

References
----------
Biot, M. A., 1952, Propagation of elastic waves in a cylindrical bore containing
a fluid: Journal of Applied Physics, 23(9), 997-1005.
https://doi.org/10.1063/1.1702365
Cheng, C. H., & Toksoz, M. N., 1981, Elastic wave propagation in a fluid-filled
borehole and synthetic acoustic logs: Geophysics, 46(7), 1042-1053.
https://doi.org/10.1190/1.1441242
Norris, A. N., 1990, The speed of a tube wave: JASA, 87(1), 414-417.
https://doi.org/10.1121/1.399262
Norris, A. N., & Sinha, B. K., 1993, Weak elastic anisotropy and the tube wave:
Geophysics, 58(8), 1091-1098. https://doi.org/10.1190/1.1443493
Tang, X. M., 2003, Determining formation shear-wave transverse isotropy from
borehole Stoneley-wave measurements: Geophysics, 68(1), 118-126.
https://doi.org/10.1190/1.1543199
Valero, H.-P., Djikpesse, H. A., & Sinha, B. K., 2009, Estimation of borehole
fluid slowness using sonic array waveforms: SEG Expanded Abstracts, 361-365.
https://doi.org/10.1190/1.3255617
Walsh, J., Sinha, B., Plona, T., & Ammerman, M., 2007, Derivation of anisotropy
parameters in a shale using borehole sonic data: SEG Expanded Abstracts,
323-327. https://doi.org/10.1190/1.2792435
White, J. E., 1983, Underground Sound: Application of Seismic Waves: Elsevier.
White, J. E., & Tongtaow, C., 1981, Cylindrical waves in transversely isotropic
media: JASA, 70(4), 1147-1155. https://doi.org/10.1121/1.386946
Winkler, K. W., Liu, H.-L., & Johnson, D. L., 1989, Permeability and borehole
Stoneley waves: Comparison between experiment and theory: Geophysics, 54(1),
66-75. https://doi.org/10.1190/1.1442578
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

__all__ = [
    "Mud",
    "Borehole",
    "tube_wave_slowness",
    "c66_from_tube_wave",
    "stoneley_slowness",
    "ti_stiffness",
    "gamma_from_moduli",
    "running_average",
    "SonicAnisotropy",
    "sonic_anisotropy",
    "GAMMA_INPUTS",
    "gamma_jacobian",
    "gamma_uncertainty",
    "calibrate_mud_slowness",
]


@dataclass(frozen=True)
class Mud:
    """Borehole fluid: velocity (m/s) and density (kg/m^3)."""

    velocity: float = 1500.0
    density: float = 1200.0

    @property
    def slowness(self) -> float:
        return 1.0 / self.velocity


@dataclass(frozen=True)
class Borehole:
    """Borehole radius and a concentric tool (radius 0: no tool; modulus inf: rigid)."""

    radius: float = 0.108
    tool_radius: float = 0.0
    tool_modulus: float = np.inf


# -- the tube wave (zero frequency) -----------------------------------------------


def _tool_term(tool_radius, tool_modulus):
    a = np.asarray(tool_radius, dtype=float)
    with np.errstate(divide="ignore"):
        kt_inv = np.where(np.isinf(tool_modulus), 0.0, 1.0 / np.asarray(tool_modulus, float))
    return a * a * kt_inv


def tube_wave_slowness(c66, mud_slowness, mud_density, radius, tool_radius=0.0,
                       tool_modulus=np.inf):
    """Zero-frequency Stoneley (tube-wave) slowness (s/m), Norris (1990)."""
    b2 = np.asarray(radius, float) ** 2
    a2 = np.asarray(tool_radius, float) ** 2
    return np.sqrt(np.asarray(mud_slowness, float) ** 2
                   + mud_density * (b2 / np.asarray(c66, float)
                                    + _tool_term(tool_radius, tool_modulus)) / (b2 - a2))


def c66_from_tube_wave(s_tube, mud_slowness, mud_density, radius, tool_radius=0.0,
                       tool_modulus=np.inf):
    """C66 (Pa) from the tube-wave slowness: the inverse of :func:`tube_wave_slowness`.

    NaN where the slowness is not larger than the tool and mud terms allow.
    """
    b2 = np.asarray(radius, float) ** 2
    a2 = np.asarray(tool_radius, float) ** 2
    den = ((np.asarray(s_tube, float) ** 2 - np.asarray(mud_slowness, float) ** 2)
           * (b2 - a2) / mud_density - _tool_term(tool_radius, tool_modulus))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, b2 / den, np.nan)


def gamma_from_moduli(c66, c44):
    """Thomsen's gamma = (C66 - C44) / (2 C44)."""
    return 0.5 * (np.asarray(c66, float) / np.asarray(c44, float) - 1.0)


# -- the exact Stoneley mode --------------------------------------------------------


def _radial(S, c11, c13, c33, c44, rho):
    """Squared radial slownesses Y (two roots) and polarizations w of the formation waves.

    For (complex) axial slowness S the axisymmetric fields of a TI formation
    (axis along the borehole) are ``u_r = U K1(kappa r)``,
    ``u_z = i w U K0(kappa r)``, with ``kappa^2 = omega^2 Y`` and Y a root of the
    Christoffel equation with an imaginary horizontal slowness.  Returns
    ``(Y, w)``, each of shape (2, len(S)).
    """
    a2 = c11 * c44
    a1 = c11 * (rho - c33 * S ** 2) + c44 * (rho - c44 * S ** 2) + S ** 2 * (c13 + c44) ** 2
    a0 = (rho - c44 * S ** 2) * (rho - c33 * S ** 2)
    sq = np.sqrt(a1 * a1 - 4.0 * a2 * a0 + 0j)
    Y = np.array([(-a1 - sq) / (2.0 * a2), (-a1 + sq) / (2.0 * a2)])
    kap = np.sqrt(Y)
    c = c13 + c44
    # two equivalent forms of the polarization; use the better-conditioned one
    d1 = rho + c44 * Y - c33 * S ** 2
    d2 = S * kap * c
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = S * kap * c / d1
        w2 = (rho + c11 * Y - c44 * S ** 2) / d2
    w = np.where(np.abs(d1) >= 1e-6 * np.abs(c11 * Y), w1, w2)
    return Y, w


def _kappa(Y):
    """Radial wavenumber / omega: evanescent (Re > 0) or outgoing (Hankel-1) radiation."""
    ev = np.sqrt(Y)
    out = -1j * np.sqrt(-Y)
    return np.where(Y.real >= 0.0, ev, out)


def _period(S, w, cij, rho, sf, rhof, b, a, kt):
    """Scaled period determinant of the axisymmetric borehole modes at (complex) slownesses S.

    A formation wave with Re(Y) < 0 radiates; it takes the outgoing branch, so
    the root is a leaky mode with complex slowness (Im S > 0: decaying along the
    borehole).
    """
    from scipy.special import ive, kve

    c11, c13, c33, c44, c66 = cij
    S = np.atleast_1d(np.asarray(S, dtype=complex))
    k = w * S
    Y, w_y = _radial(S, c11, c13, c33, c44, rho)
    q = _kappa(Y)
    pol = np.where(Y.real >= 0.0, w_y, w_y * (q / np.sqrt(Y)))
    kap = w * q
    f = w * np.sqrt(S ** 2 - sf ** 2)
    K0, K1 = kve(0, kap * b), kve(1, kap * b)
    ur = -K1                                                     # (fluid - solid) u_r
    srr = (c11 * kap + c13 * k * pol) * K0 + 2.0 * c66 * K1 / b  # -(solid sigma_rr)
    srz = c44 * (k - kap * pol) * K1                             # solid sigma_rz / i
    pf = rhof * w ** 2
    sc = c44 * w                                                 # row scale of stresses
    zero = np.zeros_like(S)
    col = [[ur[j], srr[j] / sc, srz[j] / sc] for j in range(2)]
    if a <= 0.0:
        M = np.array([
            [f * ive(1, f * b), col[0][0], col[1][0]],
            [-pf * ive(0, f * b) / sc, col[0][1], col[1][1]],
            [zero, col[0][2], col[1][2]],
        ])
    else:
        ea = np.exp(-f * (b - a))
        c = 0.0 if np.isinf(kt) else pf * a / (2.0 * kt)
        M = np.array([
            [f * ive(1, f * b), -f * kve(1, f * b) * ea, col[0][0], col[1][0]],
            [-pf * ive(0, f * b) / sc, -pf * kve(0, f * b) * ea / sc, col[0][1], col[1][1]],
            [zero, zero, col[0][2], col[1][2]],
            [(f * ive(1, f * a) + c * ive(0, f * a)) * ea,
             -f * kve(1, f * a) + c * kve(0, f * a), zero, zero],
        ])
    return np.linalg.det(np.moveaxis(M, -1, 0))


def _track(frequency: float, cij, rho, sf, rhof, b, a, kt, start: float = 20.0,
           ratio: float = 1.15, max_jump: float = 0.03):
    """Stoneley root at one frequency for many formations at once (complex, NaN on failure).

    The root is followed from the tube-wave slowness at ``frequency / start`` in
    geometric frequency steps of ``ratio``, with a few secant iterations per
    step.  A formation is dropped (NaN) when its root jumps by more than
    ``max_jump`` in one step, or when it leaves the physical region.
    """
    c11, c13, c33, c44, c66, rho, b = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(c, float)) for c in (*cij, rho, b)))
    cij = (c11, c13, c33, c44, c66)
    n_steps = max(1, int(np.ceil(np.log(start) / np.log(ratio))))
    freqs = frequency / start * start ** (np.arange(n_steps + 1) / n_steps)
    S = tube_wave_slowness(c66, sf, rhof, b, a, kt).astype(complex)
    S = np.where(np.isfinite(S), S, np.nan)
    for fr in freqs:
        w = 2.0 * np.pi * fr
        live = np.isfinite(S)
        if not live.any():
            break
        idx = np.flatnonzero(live)
        sub = tuple(c[idx] for c in cij)

        def g(x):
            with np.errstate(all="ignore"):
                return _period(x, w, sub, rho[idx], sf, rhof, b[idx], a, kt)

        x0 = S[idx]
        x1 = x0 * (1.0 + 1e-6)
        f0, f1 = g(x0), g(x1)
        done = np.zeros(idx.size, dtype=bool)
        for _ in range(15):
            act = ~done
            with np.errstate(all="ignore"):
                x2 = x1[act] - f1[act] * (x1[act] - x0[act]) / (f1[act] - f0[act])
            x2 = np.where(np.isfinite(x2), x2, x1[act])
            x0[act], f0[act] = x1[act], f1[act]
            x1[act] = x2
            sub_act = tuple(c[act] for c in sub)
            with np.errstate(all="ignore"):
                f1[act] = _period(x2, w, sub_act, rho[idx][act], sf, rhof, b[idx][act], a, kt)
            # converged samples are frozen: further secant steps on a zero residual are noise
            done |= np.abs(x1 - x0) <= 1e-12 * np.abs(x1)
            if done.all():
                break
        ok = (done & np.isfinite(x1) & (x1.real > sf) & (x1.imag > -1e-9 * np.abs(x1))
              & (np.abs(x1 - S[idx]) < max_jump * np.abs(S[idx])))
        S[idx] = np.where(ok, x1, np.nan)
    return S


def stoneley_slowness(frequency, cij, rho, mud_slowness, mud_density, radius,
                      tool_radius=0.0, tool_modulus=np.inf, full=False):
    """Stoneley phase slowness (s/m) of a fluid-filled borehole along the axis of a TI formation.

    Root of the exact modal (period) equation of a borehole of ``radius`` in a
    transversely isotropic formation with stiffnesses
    ``cij = (C11, C13, C33, C44, C66)`` (Pa) and density ``rho``, with an
    optional concentric tool of ``tool_radius`` whose wall follows the pressure
    with area modulus ``tool_modulus`` (inf: rigid).  For an isotropic formation
    this is the equation of Biot (1952) and Cheng & Toksoz (1981); the TI fields
    follow White & Tongtaow (1981).

    Either ``frequency`` (Hz) is an array and the formation is one set of
    scalars (a dispersion curve), or ``frequency`` is a scalar and the
    stiffnesses, ``rho`` and ``radius`` are arrays (a log; solved for all
    depths at once).

    The root is followed in frequency from the tube-wave limit (Norris, 1990).
    Where the Stoneley wave is faster than a formation wave travelling along the
    borehole it radiates into the formation and is leaky, with a complex
    slowness.  ``full=True`` returns the complex slowness (Im S > 0; the
    Stoneley attenuation from radiation is ``1/Q = 2 Im S / Re S``), otherwise
    its real part.  NaN where the root cannot be followed, which happens where a
    leaky mode turns into a trapped one.  With no tool and an isotropic
    formation the slowness tends to the Scholte wave of a flat fluid-solid
    interface at high frequency (Biot, 1952).
    """
    args = (rho, mud_slowness, mud_density, radius, tool_radius, tool_modulus)
    if np.ndim(frequency):
        freq = np.asarray(frequency, dtype=float)
        out = np.array([_track(fr, cij, *args)[0] for fr in freq.ravel()]).reshape(freq.shape)
    else:
        out = _track(float(frequency), cij, *args)
        if not any(np.ndim(x) for x in (*cij, rho, radius)):
            out = out.reshape(())[()]
    return out if full else np.real(out)


def ti_stiffness(vp0, vs0, rho, c66, eps=0.0, delta=0.0):
    """``(C11, C13, C33, C44, C66)`` from vertical velocities, C66 and Thomsen eps, delta."""
    c33, c44 = rho * vp0 ** 2, rho * vs0 ** 2
    c11 = c33 * (1.0 + 2.0 * eps)
    c13 = np.sqrt(2.0 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2) - c44
    return c11, c13, c33, c44, c66


# -- the anisotropy -----------------------------------------------------------------


def running_average(depth, values, length: float, harmonic: bool = False) -> np.ndarray:
    """Centred running (arithmetic or harmonic) average over ``length`` (m) on a regular grid.

    Use it to bring two logs to a common resolution before combining them: the
    dipole C44 (harmonic, as Backus) and the density to the Stoneley array
    aperture, for example.  The window is shortened at the ends.
    """
    d = np.asarray(depth, dtype=float)
    x = np.asarray(values, dtype=float)
    dz = float(np.median(np.diff(d)))
    n = max(1, int(round(length / dz)) | 1)
    if n == 1:
        return x.copy()
    v = 1.0 / x if harmonic else x
    k = np.ones(n)
    m = np.convolve(v, k, mode="same") / np.convolve(np.ones_like(v), k, mode="same")
    return 1.0 / m if harmonic else m



class SonicAnisotropy(NamedTuple):
    """Stiffnesses (Pa), gamma, and the dispersion ratio S(f) / S(0) used for C66."""

    c33: np.ndarray
    c44: np.ndarray
    c66: np.ndarray
    gamma: np.ndarray
    dispersion: np.ndarray


def sonic_anisotropy(vp, vs, rho, s_stoneley, mud: Mud, borehole: Borehole,
                     frequency: Optional[float] = None, radius=None, eps=0.0, delta=0.0,
                     equivalent_isotropic: bool = False, shear_frequency: Optional[float] = None,
                     qs=None, max_iter: int = 30, tol: float = 1e-9) -> SonicAnisotropy:
    """C33, C44, C66 and gamma from monopole P, dipole shear and Stoneley logs.

    ``vp`` and ``vs`` (m/s) and ``rho`` (kg/m^3) are the logs, ``s_stoneley``
    (s/m) the Stoneley slowness measured at ``frequency`` (Hz).  ``radius``
    (array, e.g. the caliper) overrides ``borehole.radius``.

    With ``frequency=None`` the Stoneley slowness is taken as the tube-wave
    slowness.  Otherwise C66 is found by a secant iteration (in log C66, for all
    depths at once) such that the exact borehole mode (:func:`stoneley_slowness`)
    reproduces the measured slowness.  The formation is TI with C33 and C44 from
    the logs and C11, C13 from the assumed Thomsen ``eps`` and ``delta``; with
    ``equivalent_isotropic`` it is isotropic with shear modulus C66 instead (the
    usual shortcut, which ignores the measured C44).  ``dispersion`` is
    ``S(f) / S(0)`` at the solution; NaN where no solution was found.

    The dipole measures C44 at its own frequency (about 2 kHz), above the
    Stoneley frequency.  With ``shear_frequency`` and ``qs``, C44 is moved to
    the Stoneley ``frequency`` with the constant-Q law before gamma is formed;
    otherwise gamma is biased low by about ``(1 + 2 gamma) g ln(f_dipole / f)``
    with ``g = arctan(1/Qs) / pi`` (0.02-0.03 for Qs = 25 and a factor 4).
    """
    vp, vs, rho = (np.asarray(x, dtype=float) for x in (vp, vs, rho))
    s_st = np.asarray(s_stoneley, dtype=float)
    shape = np.broadcast(vp, vs, rho, s_st).shape
    b = np.broadcast_to(borehole.radius if radius is None else np.asarray(radius, float), shape)
    vp, vs, rho, s_st = (np.broadcast_to(x, shape).astype(float) for x in (vp, vs, rho, s_st))
    eps, delta = (np.broadcast_to(np.asarray(x, float), shape) for x in (eps, delta))
    sf, rf = mud.slowness, mud.density
    a, kt = borehole.tool_radius, borehole.tool_modulus
    c33, c44 = rho * vp ** 2, rho * vs ** 2
    if frequency is not None and shear_frequency is not None and qs is not None:
        from .qmodel import dispersion_exponent

        c44 = c44 * (frequency / shear_frequency) ** (2.0 * dispersion_exponent(qs))
        vs = np.sqrt(c44 / rho)
    c66 = c66_from_tube_wave(s_st, sf, rf, b, a, kt)
    ratio = np.ones(shape)
    if frequency is not None:

        def model(c):
            if equivalent_isotropic:
                cij = ti_stiffness(vp, np.sqrt(c / rho), rho, c)
            else:
                cij = ti_stiffness(vp, vs, rho, c, eps, delta)
            return stoneley_slowness(frequency, cij, rho, sf, rf, b, a, kt)

        x0 = np.log(c66)
        f0 = model(c66) - s_st
        # where the model is undefined (too soft: past the leaky-trapped transition), stiffen
        for _ in range(10):
            bad = ~np.isfinite(f0) & np.isfinite(x0)
            if not bad.any():
                break
            x0 = np.where(bad, x0 + 0.1, x0)
            f0 = np.where(bad, model(np.exp(x0)) - s_st, f0)
        x1 = x0 + 0.02
        f1 = model(np.exp(x1)) - s_st
        for _ in range(max_iter):
            with np.errstate(all="ignore"):
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x2 = np.where(np.isfinite(x2), np.clip(x2, x1 - 0.3, x1 + 0.3), x1)
            x0, f0, x1 = x1, f1, x2
            f1 = model(np.exp(x1)) - s_st
            if np.all(~np.isfinite(f1) | (np.abs(x1 - x0) < tol)):
                break
        c66 = np.where(np.isfinite(f1) & (np.abs(f1) < 1e-6 * s_st), np.exp(x1), np.nan)
        ratio = s_st / tube_wave_slowness(c66, sf, rf, b, a, kt)
    return SonicAnisotropy(c33, c44, c66, gamma_from_moduli(c66, c44), ratio)


#: Inputs of :func:`gamma_jacobian`, in order.
GAMMA_INPUTS = ("s_stoneley", "s_shear", "rho", "mud_slowness", "mud_density", "radius",
                "tool_radius", "tool_compliance")


def gamma_jacobian(s_stoneley, s_shear, rho, mud_slowness, mud_density, radius,
                   tool_radius=0.0, tool_compliance=0.0):
    """Exact derivatives of gamma (tube-wave relation) with respect to :data:`GAMMA_INPUTS`.

    ``s_stoneley`` is the (dispersion-corrected) tube-wave slowness and
    ``tool_compliance`` = 1 / K_t.  Returns an array ``[..., 8]`` of partial
    derivatives (per s/m, per kg/m^3, per m, per 1/Pa).
    """
    S, Ss, rho, Sf, rf, b, a, ct = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (s_stoneley, s_shear, rho, mud_slowness,
                                                mud_density, radius, tool_radius,
                                                tool_compliance)))
    e = S ** 2 - Sf ** 2
    ann = b ** 2 - a ** 2
    den = e * ann / rf - a ** 2 * ct
    c66 = b ** 2 / den
    c44 = rho / Ss ** 2
    dg_dc66 = 0.5 / c44
    dg_dc44 = -0.5 * c66 / c44 ** 2
    dc66 = {
        "s_stoneley": -b ** 2 * (2.0 * S * ann / rf) / den ** 2,
        "mud_slowness": b ** 2 * (2.0 * Sf * ann / rf) / den ** 2,
        "mud_density": b ** 2 * (e * ann / rf ** 2) / den ** 2,
        "radius": 2.0 * b / den - b ** 2 * (2.0 * b * e / rf) / den ** 2,
        "tool_radius": -b ** 2 * (-2.0 * a * e / rf - 2.0 * a * ct) / den ** 2,
        "tool_compliance": b ** 2 * a ** 2 / den ** 2,
    }
    dc44 = {"s_shear": -2.0 * rho / Ss ** 3, "rho": 1.0 / Ss ** 2}
    return np.stack([dg_dc66 * dc66.get(k, 0.0) + dg_dc44 * dc44.get(k, 0.0)
                     for k in GAMMA_INPUTS], axis=-1)


def gamma_uncertainty(jacobian, sigmas: dict):
    """Linear propagation: ``(std, contributions)`` of gamma for independent input errors.

    ``sigmas`` maps names in :data:`GAMMA_INPUTS` to standard deviations (scalars
    or arrays); ``contributions`` maps each name to its share ``|dg/dx| sigma_x``.
    """
    contrib = {k: np.abs(jacobian[..., GAMMA_INPUTS.index(k)]) * np.asarray(v, float)
               for k, v in sigmas.items()}
    total = np.sqrt(sum(c ** 2 for c in contrib.values()))
    return total, contrib


def calibrate_mud_slowness(vp, vs, rho, s_stoneley, mud_density: float, borehole: Borehole,
                           gamma_ref: float = 0.0, frequency: Optional[float] = None,
                           radius=None, bounds=(1.0 / 1800.0, 1.0 / 1300.0), **kwargs) -> float:
    """Mud slowness (s/m) for which the mean gamma over the given samples equals ``gamma_ref``.

    The samples should come from an interval believed to have a known gamma
    (usually assumed isotropic).  Any error in that belief maps one to one into
    every gamma of the well, and permeability raises the Stoneley slowness in
    sands, which this calibration would wrongly absorb into the mud slowness.
    Further keyword arguments (``eps``, ``delta``, ``shear_frequency``, ``qs``,
    ...) go to :func:`sonic_anisotropy`.
    """
    from scipy.optimize import brentq

    def misfit(sf):
        g = sonic_anisotropy(vp, vs, rho, s_stoneley, Mud(1.0 / sf, mud_density), borehole,
                             frequency, radius, **kwargs).gamma
        return float(np.nanmean(g)) - gamma_ref

    return brentq(misfit, bounds[0], bounds[1], xtol=1e-12)
