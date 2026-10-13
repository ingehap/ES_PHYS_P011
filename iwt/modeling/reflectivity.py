"""Reflectivity method for a 1D layered VTI earth with constant-Q attenuation.

The response is computed in the frequency-slowness domain with Kennett's
recursive reflection and transmission matrices, which are stable for any layer
thickness (only decaying exponentials appear).  The model may start with one
fluid layer (the sea), under a free surface; every other layer is a solid.

Physics switches (all on by default):

* ``attenuation``: Kjartansson constant-Q complex velocities (:mod:`.elastic`);
* ``free_surface``: pressure-release sea surface (source and receiver ghosts and
  free-surface multiples); off gives an absorbing top;
* ``internal_multiples``: reverberations between interfaces below the free
  surface.  Off keeps every primary reflection with its transmission losses;
* ``converted_waves``: P-SV mode conversion at interfaces.  Off keeps the exact
  elastic PP coefficients but drops all converted energy;
* ``transmission_loss`` (only with internal multiples off): False sets every
  interface transmission to 1, which gives the single-scattering (Born)
  primaries.  In finely layered media, short-period multiples restore most of
  the amplitude that transmission losses remove (O'Doherty & Anstey, 1971).
  Primaries *with* transmission loss are then far weaker than the full response,
  and the Born primaries are the better reference for isolating multiples.

Sources are explosions (or unit plane waves) in the sea; receivers can be in the
sea (hydrophone pressure) or in the solid (3C displacement), at or below the
source depth.  Point-source records are built from the plane-wave responses with
the Sommerfeld integral over horizontal slowness (cylindrical symmetry):

    exp(i k R) / R = i omega Int (p / q) J0(omega p r) exp(i omega q |z - zs|) dp

Frequencies are complex (``omega + i eps``), to
suppress time-domain wrap-around; the damping is removed after the inverse FFT.

References
----------
Kennett, B. L. N., & Kerry, N. J., 1979, Seismic waves in a stratified half space:
GJI, 57(3), 557-583. https://doi.org/10.1111/j.1365-246X.1979.tb06779.x
Fryer, G. J., & Frazer, L. N., 1984, Seismic waves in stratified anisotropic media:
GJI, 78(3), 691-710. https://doi.org/10.1111/j.1365-246X.1984.tb05065.x
O'Doherty, R. F., & Anstey, N. A., 1971, Reflections on amplitudes: Geophysical
Prospecting, 19(3), 430-458. https://doi.org/10.1111/j.1365-2478.1971.tb00610.x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..model import ElasticModel1D
from .elastic import (
    _polarization,
    complex_velocity,
    fluid_eigenvectors,
    interface_scattering,
    layer_moduli,
    vertical_slowness,
)

__all__ = ["Reflectivity", "Record", "time_axis_spectrum", "spectrum_to_trace"]

_COMPONENTS = ("ux", "uz", "tzz", "pressure", "down_p", "up_p", "down_s", "up_s")


# -- 2x2 algebra on tuples of arrays ------------------------------------------------
# A 2x2 matrix is a tuple (a00, a01, a10, a11) of arrays (or scalars) that
# broadcast over (frequency, slowness).  Explicit element-wise algebra is much
# faster than numpy's batched matmul / solve for millions of tiny matrices.

_EYE = (1.0, 0.0, 0.0, 1.0)
_ZERO = (0.0, 0.0, 0.0, 0.0)


def _mul(a, b):
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def _neg(a):
    return (-a[0], -a[1], -a[2], -a[3])


def _inv(a):
    det = a[0] * a[3] - a[1] * a[2]
    return (a[3] / det, -a[1] / det, -a[2] / det, a[0] / det)


def _vec(a, v):
    return (a[0] * v[0] + a[1] * v[1], a[2] * v[0] + a[3] * v[1])


def _sandwich(e, a):
    """diag(e) a diag(e)."""
    return (e[0] * a[0] * e[0], e[0] * a[1] * e[1], e[1] * a[2] * e[0], e[1] * a[3] * e[1])


def _left(e, a):
    """diag(e) a."""
    return (e[0] * a[0], e[0] * a[1], e[1] * a[2], e[1] * a[3])


def _right(a, e):
    """a diag(e)."""
    return (a[0] * e[0], a[1] * e[1], a[2] * e[0], a[3] * e[1])


def _from_array(x):
    return (x[..., 0, 0], x[..., 0, 1], x[..., 1, 0], x[..., 1, 1])


# -- time-frequency helpers -------------------------------------------------------


def time_axis_spectrum(dt: float, nt: int, f_max: Optional[float] = None,
                       damping: Optional[float] = None):
    """Frequencies, complex angular frequencies and damping for an ``nt``-sample trace.

    Returns ``(f, omega_c, eps, n_used)``; only frequencies up to ``f_max`` are
    computed (``n_used`` of them), the rest are zero.  The default damping
    ``ln(50) / T`` attenuates wrapped-around energy by a factor 50.
    """
    T = nt * dt
    f = np.fft.rfftfreq(nt, dt)
    eps = np.log(50.0) / T if damping is None else float(damping)
    n_used = f.size if f_max is None else int(np.searchsorted(f, f_max, side="right"))
    return f, 2.0 * np.pi * f[:n_used] - 0j + 1j * eps, eps, n_used


def spectrum_to_trace(spec, dt: float, nt: int, eps: float, omega=None,
                      t_shift: float = 0.0):
    """Time series from a spectrum on the first frequencies (``exp(-i omega t)`` convention).

    ``spec`` has the frequency axis last; frequencies beyond its length are zero.
    With ``t_shift > 0`` (and ``omega``, the complex frequencies of ``spec``) the
    signal is delayed by ``t_shift`` before the inverse transform and advanced
    afterwards.  Energy at negative times (e.g. from a zero-phase wavelet) then
    does not wrap around to the end of the trace, where the damping correction
    would amplify it.  The last ``t_shift`` of the trace is returned as zeros.
    """
    spec = np.asarray(spec)
    n0 = int(round(t_shift / dt))
    if n0:
        spec = spec * np.exp(1j * np.asarray(omega) * n0 * dt)
    full = np.zeros(spec.shape[:-1] + (nt // 2 + 1,), dtype=complex)
    full[..., : spec.shape[-1]] = spec
    trace = np.fft.irfft(np.conj(full), n=nt, axis=-1) / dt
    trace = trace * np.exp(eps * dt * np.arange(nt))
    if n0:
        out = np.zeros_like(trace)
        out[..., : nt - n0] = trace[..., n0:]
        trace = out
    return trace


def wavelet_spectrum(t, w, omega):
    """Spectrum ``Int w(t) exp(i omega t) dt`` of a sampled wavelet (omega may be complex)."""
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    return np.exp(1j * np.asarray(omega)[:, None] * t[None, :]) @ np.asarray(w, float) * dt


@dataclass
class Record:
    """A synthetic record: ``data[receiver, offset, time]``."""

    t: np.ndarray
    offsets: np.ndarray
    depths: np.ndarray
    data: np.ndarray
    component: str


class Reflectivity:
    """Reflectivity-method modelling of a 1D VTI model (see module docstring)."""

    def __init__(self, model: ElasticModel1D, attenuation: bool = True,
                 free_surface: bool = True, internal_multiples: bool = True,
                 converted_waves: bool = True, transmission_loss: bool = True):
        fl = model.is_fluid
        n_fluid = int(np.argmin(fl)) if not fl.all() else fl.size
        if fl[n_fluid:].any():
            raise NotImplementedError("fluid layers are supported only at the top of the model")
        self.model = model
        self.attenuation = attenuation
        self.free_surface = free_surface
        self.internal_multiples = internal_multiples
        self.converted_waves = converted_waves
        if internal_multiples and not transmission_loss:
            raise ValueError("transmission_loss=False requires internal_multiples=False")
        self.transmission_loss = transmission_loss

    # -- layer quantities ----------------------------------------------------

    def _velocities(self, m: ElasticModel1D, omega):
        w_ref = 2.0 * np.pi * m.f_ref
        if self.attenuation:
            vp = complex_velocity(m.vp0[:, None], m.qp[:, None], omega[None, :], w_ref)
            vs = complex_velocity(m.vs0[:, None], m.qs[:, None], omega[None, :], w_ref)
        else:
            vp = np.broadcast_to(m.vp0[:, None] + 0j, (m.n_layers, omega.size))
            vs = np.broadcast_to(m.vs0[:, None] + 0j, (m.n_layers, omega.size))
        return vp, vs

    def _layer(self, m, vp, vs, j, p, scale):
        """Plane-wave description of layer j (p has shape (1, np)).

        Returns a dict with ``fluid``, ``q`` (qP, qS) and the eigenvector blocks
        ``Ad, Au`` (displacement of down/up waves) and ``Cd, Cu`` (tractions,
        divided by ``scale``), each a 2x2 tuple with columns [P, S].  For a fluid,
        ``D`` is the 2x2 array of :func:`fluid_eigenvectors` instead.
        """
        rho = m.rho[j]
        c11, c13, c33, c44 = layer_moduli(vp[j][:, None], vs[j][:, None], rho,
                                          m.eps[j], m.delta[j])
        if m.is_fluid[j]:
            qp, _ = vertical_slowness(c11, c13, c33, 0.0 * c44, rho, p)
            return {"fluid": True, "q": (qp, 0.0), "D": fluid_eigenvectors(c33, rho, p, qp),
                    "alpha": vp[j][:, None]}
        qp, qs = vertical_slowness(c11, c13, c33, c44, rho, p)
        a11, a13, a33, a55 = c11 / rho, c13 / rho, c33 / rho, c44 / rho
        px, pz = _polarization(a11, a13, a33, a55, p, qp, "P")
        sx, sz = _polarization(a11, a13, a33, a55, p, qs, "S")
        c13s, c33s, c44s = c13 / scale, c33 / scale, c44 / scale
        txp = c44s * (qp * px + p * pz)
        tzp = c13s * p * px + c33s * qp * pz
        txs = c44s * (qs * sx + p * sz)
        tzs = c13s * p * sx + c33s * qs * sz
        # upgoing waves (q -> -q): P polarization (px, -pz), S polarization (-sx, sz).
        # N: the symplectic pairing of each down wave with its up wave, D^T J D =
        # [[0, N], [-N, 0]] with J = [[0, S], [-S, 0]], S = diag(-1, 1).
        return {"fluid": False, "q": (qp, qs),
                "Ad": (px, sx, pz, sz), "Au": (px, -sx, -pz, sz),
                "Cd": (txp, txs, tzp, tzs), "Cu": (-txp, txs, tzp, -tzs),
                "N": (2.0 * (px * txp + pz * tzp), -2.0 * (sx * txs + sz * tzs))}

    @staticmethod
    def _phase(omega, lay, h):
        w = omega[:, None]
        if lay["fluid"]:
            return (np.exp(1j * w * lay["q"][0] * h), 0.0)
        return (np.exp(1j * w * lay["q"][0] * h), np.exp(1j * w * lay["q"][1] * h))

    def _same(self, m, j):
        """True when layers j-1 and j have identical properties (no interface)."""
        a, b = j - 1, j
        return all(getattr(m, n)[a] == getattr(m, n)[b]
                   for n in ("vp0", "vs0", "rho", "eps", "delta", "qp", "qs"))

    def _interface(self, upper, lower, scale):
        """Scattering blocks (RD, TD, RU, TU) of the interface between two layers."""
        if upper["fluid"] and lower["fluid"]:
            raise NotImplementedError("fluid-fluid interfaces are not supported")
        if upper["fluid"]:
            Dl = np.empty(np.broadcast(upper["D"][..., 0, 0], lower["Ad"][0]).shape + (4, 4),
                          dtype=complex)
            for (r, c), blk in (((0, 0), "Ad"), ((0, 2), "Au"), ((2, 0), "Cd"), ((2, 2), "Cu")):
                x = lower[blk]
                Dl[..., r, c], Dl[..., r, c + 1] = x[0], x[1]
                Dl[..., r + 1, c], Dl[..., r + 1, c + 1] = x[2], x[3]
            Df = upper["D"].copy()
            Df[..., 1, :] /= scale
            out = interface_scattering(Df, Dl, fluid_upper=True)
            blocks = tuple(_from_array(x) for x in out)
        else:
            # Closed form from D^-1 = M^-1 D^T J and the up/down symmetries
            # Au = S2 Ad S2, Cu = S1 Cd S2 (S1 = diag(-1, 1), S2 = diag(1, -1)):
            #   K = Cd1^T Ad2 + Ad1^T Cd2,   L = Ad1^T S1 Cd2 - Cd1^T S1 Ad2,
            #   TD = K^-1 S2 N,  RD = N^-1 L K^-1 S2 N,  RU = -K^-1 L S2,
            #   TU = N^-1 (K - L K^-1 L) S2   (N of the upper layer).
            A1, C1, A2, C2 = upper["Ad"], upper["Cd"], lower["Ad"], lower["Cd"]
            A1t = (A1[0], A1[2], A1[1], A1[3])
            C1t = (C1[0], C1[2], C1[1], C1[3])
            K = _add(_mul(C1t, A2), _mul(A1t, C2))
            # S1 acts on rows of Cd2 / Ad2: flip the sign of row 0
            L = _sub(_mul(A1t, (-C2[0], -C2[1], C2[2], C2[3])),
                     _mul(C1t, (-A2[0], -A2[1], A2[2], A2[3])))
            Ki = _inv(K)
            LKi = _mul(L, Ki)
            n0, n1 = upper["N"]
            # right-multiplying by S2 flips column 1; diag scalings by N or 1/N
            TD = (Ki[0] * n0, -Ki[1] * n1, Ki[2] * n0, -Ki[3] * n1)
            RD = (LKi[0], -LKi[1] * (n1 / n0), LKi[2] * (n0 / n1), -LKi[3])
            KiL = _mul(Ki, L)
            RU = (-KiL[0], KiL[1], -KiL[2], KiL[3])
            X = _sub(K, _mul(LKi, L))
            TU = (X[0] / n0, -X[1] / n0, X[2] / n1, -X[3] / n1)
            blocks = (RD, TD, RU, TU)
        if not self.converted_waves:
            blocks = tuple((x[0], 0.0, 0.0, x[3]) for x in blocks)
        if not self.transmission_loss:
            blocks = (blocks[0], _EYE, blocks[2], _EYE)
        return blocks

    # -- the response --------------------------------------------------------

    def plane_wave(self, omega, p, z_source: float, z_receivers: Sequence[float],
                   source: str = "downgoing", source_amplitude=None,
                   chunk: int = 4096, workers: int = 1) -> dict:
        """Wavefields at the receivers for plane waves of horizontal slowness ``p``.

        The work is done in frequency blocks of about ``chunk`` (frequency,
        slowness) pairs, which keeps the arrays in cache.  With ``workers > 1``
        the blocks run in parallel processes.  The result depends on neither
        option.  See :meth:`_plane_wave_block` for the outputs.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=complex))
        n_p = np.atleast_1d(p).size
        step = max(1, chunk // max(n_p, 1))
        if workers > 1:
            step = min(step, -(-omega.size // workers))
        amp = None if source_amplitude is None else \
            np.asarray(source_amplitude, dtype=complex) * np.ones((omega.size, n_p))
        jobs = [(omega[i:i + step], p, z_source, z_receivers, source,
                 None if amp is None else amp[i:i + step])
                for i in range(0, omega.size, step)]
        if workers > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                parts = list(ex.map(self._plane_wave_block, *zip(*jobs)))
        else:
            parts = [self._plane_wave_block(*job) for job in jobs]
        return {k: np.concatenate([q[k] for q in parts], axis=1) for k in parts[0]}

    def _plane_wave_block(self, omega, p, z_source: float, z_receivers: Sequence[float],
                          source: str = "downgoing", source_amplitude=None) -> dict:
        """Wavefields at the receivers for plane waves of horizontal slowness ``p``.

        ``omega`` (nf,) may be complex; ``p`` (np,) real.  ``source`` is
        ``"downgoing"`` (unit downgoing P at the source depth) or ``"explosion"``
        (unit downgoing and upgoing P).  ``source_amplitude`` (nf, np) multiplies
        both.  Returns a dict of arrays (n_receivers, nf, np) for ``ux``, ``uz``,
        ``tzz`` (traction / (i omega)), ``pressure`` (fluid receivers; NaN in a
        solid) and the wave amplitudes ``down_p``, ``up_p``, ``down_s``, ``up_s``
        just below each receiver depth.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=complex))
        p = np.atleast_1d(np.asarray(p, dtype=float))[None, :] + 0j
        z_rec = np.atleast_1d(np.asarray(z_receivers, dtype=float))
        model = self.model
        if not (model.interfaces[0] < z_source < model.interfaces[-1]):
            raise ValueError("source depth must be inside the model and below the surface")
        if not model.is_fluid[model.layer_index(z_source)]:
            raise NotImplementedError("the source must be in the sea (fluid) layer")
        if np.any(z_rec < z_source) or np.any(z_rec >= model.interfaces[-1]):
            raise ValueError("receivers must lie between the source depth and the model bottom")

        m = model.split_at(np.concatenate([[z_source], z_rec]))
        vp, vs = self._velocities(m, omega)
        nf, npp = omega.size, p.shape[1]
        shape = (nf, npp)
        scale = float(np.mean(m.rho * m.vp0))
        s_idx = int(np.searchsorted(m.interfaces, z_source))
        r_idx = np.searchsorted(m.interfaces, z_rec)
        levels = set(r_idx.tolist()) | {s_idx}
        mult = self.internal_multiples

        def layer(j):
            return self._layer(m, vp, vs, j, p, scale)

        # upward pass: reflection matrix from below at every level
        R = _ZERO
        R_below = {}
        lower = layer(m.n_layers - 1)
        for j in range(m.n_layers - 1, 0, -1):
            if j in levels:
                R_below[j] = R
            upper = layer(j - 1)
            if not self._same(m, j):
                RD, TD, RU, TU = self._interface(upper, lower, scale)
                if mult:
                    R = _add(RD, _mul(_mul(TU, R), _mul(_inv(_sub(_EYE, _mul(RU, R))), TD)))
                else:
                    R = _add(RD, _mul(_mul(TU, R), TD))
            R = _sandwich(self._phase(omega, upper, m.thickness[j - 1]), R)
            lower = upper

        # region above the source: free surface and sea
        RU_fs = (-1.0, 0.0, 0.0, 0.0) if self.free_surface else _ZERO
        for j in range(0, s_idx):
            RU_fs = _sandwich(self._phase(omega, layer(j), m.thickness[j]), RU_fs)

        # source wavefield just below the source level
        amp = np.ones(shape, dtype=complex) if source_amplitude is None else \
            np.asarray(source_amplitude, dtype=complex) * np.ones(shape)
        sig_d = (amp, 0.0)
        if source == "explosion":
            sig_u = (amp, 0.0)
        elif source == "downgoing":
            sig_u = (0.0, 0.0)
        else:
            raise ValueError("source must be 'downgoing' or 'explosion'")
        RB_s = R_below[s_idx]
        rhs = _vec(RU_fs, sig_u)
        w_d = _vec(_inv(_sub(_EYE, _mul(RU_fs, RB_s))), (sig_d[0] + rhs[0], sig_d[1] + rhs[1]))

        # downward pass: transmission and reflection-from-above between source and receivers
        TDm, RUm = _EYE, _ZERO
        slab = {s_idx: (TDm, RUm)}
        r_max = int(r_idx.max())
        upper = layer(s_idx)
        for j in range(s_idx, r_max):
            e = self._phase(omega, upper, m.thickness[j])
            TDm = _left(e, TDm)
            RUm = _sandwich(e, RUm)
            lower = layer(j + 1)
            if not self._same(m, j + 1):
                RD, TD, RU, TU = self._interface(upper, lower, scale)
                if mult:
                    TDm_new = _mul(_mul(TD, _inv(_sub(_EYE, _mul(RUm, RD)))), TDm)
                    RUm = _add(RU, _mul(_mul(TD, RUm), _mul(_inv(_sub(_EYE, _mul(RD, RUm))), TU)))
                else:
                    TDm_new = _mul(TD, TDm)
                    RUm = _add(RU, _mul(_mul(TD, RUm), TU))
                TDm = TDm_new
            if j + 1 in levels:
                slab[j + 1] = (TDm, RUm)
            upper = lower

        # fields at the receivers
        out = {c: np.full((z_rec.size, nf, npp), np.nan + 0j) for c in _COMPONENTS}
        for k, r in enumerate(r_idx):
            TDm, RUm = slab[int(r)]
            RB = R_below[int(r)]
            if r == s_idx:
                v_d = w_d
            elif mult:
                v_d = _vec(_mul(_inv(_sub(_EYE, _mul(RUm, RB))), TDm), w_d)
            else:
                v_d = _vec(TDm, w_d)
            v_u = _vec(RB, v_d)
            v_d = tuple(np.broadcast_to(x, shape) for x in v_d)
            v_u = tuple(np.broadcast_to(x, shape) for x in v_u)
            out["down_p"][k], out["down_s"][k] = v_d
            out["up_p"][k], out["up_s"][k] = v_u
            lay = layer(int(r))
            if lay["fluid"]:
                D = lay["D"]
                uz = D[..., 0, 0] * v_d[0] + D[..., 0, 1] * v_u[0]
                tzz = D[..., 1, 0] * v_d[0] + D[..., 1, 1] * v_u[0]
                out["ux"][k] = lay["alpha"] * p * (v_d[0] + v_u[0])
                out["uz"][k] = uz
                out["tzz"][k] = tzz
                out["pressure"][k] = -1j * omega[:, None] * tzz
            else:
                ud = _vec(lay["Ad"], v_d)
                uu = _vec(lay["Au"], v_u)
                td = _vec(lay["Cd"], v_d)
                tu = _vec(lay["Cu"], v_u)
                out["ux"][k] = ud[0] + uu[0]
                out["uz"][k] = ud[1] + uu[1]
                out["tzz"][k] = (td[1] + tu[1]) * scale
        return out

    def record(self, offsets, z_source: float, z_receivers, dt: float, nt: int,
               wavelet, component: str = "pressure", f_max: Optional[float] = None,
               max_angle: float = 88.0, n_p: Optional[int] = None,
               taper_deg: float = 8.0, damping: Optional[float] = None,
               workers: int = 1) -> Record:
        """Point-source (explosion) record at ``offsets`` (m) and receiver depths.

        ``wavelet`` is ``(t, w)`` (sampled, may be centred on t = 0); its spectrum
        is the source-potential time function.  ``component`` is ``"pressure"``
        (receivers in the sea), ``"uz"`` (vertical displacement) or ``"ur"``
        (radial displacement).

        The slowness integral is sampled uniformly in take-off angle phi at the
        source, ``p = sin(phi) / alpha``.  Then ``(p / q) dp = sin(phi) dphi /
        alpha`` has no singularity at grazing incidence.  Angles run to
        ``max_angle`` degrees, with a cosine taper over the last ``taper_deg``.
        Waves that are evanescent in the sea (p > 1 / alpha) are omitted, and the
        truncation leaves a small artifact before the direct arrival (at times
        below about ``|z_r - z_s| / alpha``).  The direct and near-grazing waves
        in the sea are therefore approximate; reflections are not affected.  A
        receiver at the source point (zero offset at the source depth) samples
        the singular source field and is not meaningful.
        """
        from scipy.special import jv

        offsets = np.atleast_1d(np.asarray(offsets, dtype=float))
        z_rec = np.atleast_1d(np.asarray(z_receivers, dtype=float))
        model = self.model
        i_src = model.layer_index(z_source)
        alpha = model.vp0[i_src]
        # Truncating the slowness integral leaves acausal energy (the incoming half
        # of J0) down to t = -offset * p_max, and a zero-phase wavelet puts energy
        # at negative times too.  Delaying everything by t_shift on a padded time
        # axis keeps that energy out of the returned trace.
        t_shift = max(0.0, -float(wavelet[0][0])) + offsets.max() / alpha
        n_pad = int(np.ceil(t_shift / dt))
        nt_int = nt + n_pad
        f, omega, eps, n_used = time_axis_spectrum(dt, nt_int, f_max, damping)
        phi_max = np.radians(max_angle)
        if n_p is None:
            span = offsets.max() + 2.0 * (model.interfaces[-1] - z_source)
            n_p = int(np.clip(4.0 * phi_max * f[n_used - 1] * span / alpha, 200, 8000))
        dphi = phi_max / n_p
        phi = (np.arange(n_p) + 0.5) * dphi
        p = np.sin(phi) / alpha
        tap = np.ones(n_p)
        edge = phi > phi_max - np.radians(taper_deg)
        if edge.any():
            x = (phi[edge] - (phi_max - np.radians(taper_deg))) / np.radians(taper_deg)
            tap[edge] = 0.5 * (1.0 + np.cos(np.pi * x))
        # source: potential S/(4 pi) exp(ikR)/R = S/(4 pi) i omega Int (p/q) J0 exp(i omega q|z|) dp,
        # times i omega / alpha to convert potential to displacement amplitude
        if self.attenuation:
            v_src = complex_velocity(alpha, model.qp[i_src], omega, 2.0 * np.pi * model.f_ref)
        else:
            v_src = np.full(omega.shape, alpha + 0j)
        q_src = np.sqrt(1.0 / v_src[:, None] ** 2 - p[None, :] ** 2 + 0j)
        q_src = np.where(q_src.imag < 0, -q_src, q_src)
        dp = (np.cos(phi) / alpha * dphi)[None, :]
        W = wavelet_spectrum(wavelet[0], wavelet[1], omega)
        amp = (1j * omega[:, None] / v_src[:, None]) * (1j * omega[:, None]) \
            * (p[None, :] / q_src) * dp * tap[None, :] * W[:, None] / (4.0 * np.pi)
        key = {"pressure": "pressure", "uz": "uz", "ur": "ux"}[component]
        fields = self.plane_wave(omega, p, z_source, z_rec, source="explosion",
                                 source_amplitude=amp, workers=workers)[key]
        data = np.empty((z_rec.size, offsets.size, nt))
        for i, r in enumerate(offsets):
            arg = omega[:, None] * p[None, :] * r
            kern = jv(0, arg) if component != "ur" else 1j * jv(1, arg)
            spec = np.sum(fields * kern[None, :, :], axis=-1)
            data[:, i, :] = spectrum_to_trace(spec, dt, nt_int, eps, omega, n_pad * dt)[..., :nt]
        return Record(np.arange(nt) * dt, offsets, z_rec, data, component)
