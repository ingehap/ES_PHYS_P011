"""One-dimensional elastic VTI earth model with attenuation.

The model is a stack of homogeneous layers between ``interfaces`` (depths, metres,
positive down).  Each layer is transversely isotropic with a vertical symmetry
axis (VTI) and is described by Thomsen's (1986) parameters:

* ``vp0``, ``vs0``: vertical P and S velocities (m/s) at the reference
  frequency ``f_ref`` (Hz);
* ``rho``: density (kg/m^3);
* ``eps``, ``delta``, ``gamma``: Thomsen's anisotropy parameters;
* ``qp``, ``qs``: P and S quality factors (``inf`` = no attenuation).

A layer with ``vs0 == 0`` is a fluid (sea water); its anisotropy parameters must
be zero.

The stiffness relations are exact (Thomsen, 1986):

    C33 = rho Vp0^2,   C44 = rho Vs0^2
    eps   = (C11 - C33) / (2 C33)
    gamma = (C66 - C44) / (2 C44)
    delta = ((C13 + C44)^2 - (C33 - C44)^2) / (2 C33 (C33 - C44))

Inverting the last relation takes the root with C13 + C44 >= 0.  A real C13
exists only for delta >= -(1 - Vs0^2/Vp0^2)/2 (:func:`delta_min`).

Upscaling uses Backus (1962) averaging, which gives the long-wavelength
equivalent VTI medium of a stack of VTI layers:

    C33* = <1/C33>^-1          C44* = <1/C44>^-1          C66* = <C66>
    C13* = <C13/C33> C33*      C11* = <C11 - C13^2/C33> + <C13/C33>^2 C33*
    rho* = <rho>

where <.> is the thickness-weighted mean.  Attenuation is averaged so that
the vertical traveltime-weighted loss is preserved:
1/Q* = <1/(V Q)> / <1/V> with V the vertical velocity of the same wave type.

References
----------
Backus, G. E., 1962, Long-wave elastic anisotropy produced by horizontal layering:
JGR, 67(11), 4427-4440. https://doi.org/10.1029/JZ067i011p04427
Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954-1966.
https://doi.org/10.1190/1.1442051
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np

__all__ = [
    "Stiffness",
    "ElasticModel1D",
    "thomsen_to_cij",
    "cij_to_thomsen",
    "delta_min",
    "voigt_matrix",
    "is_stable",
    "phase_velocities",
    "backus_average",
]


class Stiffness(NamedTuple):
    """The five independent VTI stiffnesses (Pa).  C12 = C11 - 2 C66."""

    c11: np.ndarray
    c13: np.ndarray
    c33: np.ndarray
    c44: np.ndarray
    c66: np.ndarray


# ---------------------------------------------------------------------------
# Thomsen parameters <-> stiffnesses
# ---------------------------------------------------------------------------


def delta_min(vp0, vs0):
    """Smallest delta for which C13 is real: -(1 - Vs0^2/Vp0^2) / 2."""
    vp0 = np.asarray(vp0, dtype=float)
    vs0 = np.asarray(vs0, dtype=float)
    return -0.5 * (1.0 - (vs0 / vp0) ** 2)


def thomsen_to_cij(vp0, vs0, rho, eps, delta, gamma) -> Stiffness:
    """Stiffnesses (Pa) from Thomsen parameters.  Inputs broadcast."""
    vp0, vs0, rho, eps, delta, gamma = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (vp0, vs0, rho, eps, delta, gamma))
    )
    c33 = rho * vp0**2
    c44 = rho * vs0**2
    c11 = c33 * (1.0 + 2.0 * eps)
    c66 = c44 * (1.0 + 2.0 * gamma)
    arg = 2.0 * delta * c33 * (c33 - c44) + (c33 - c44) ** 2
    if np.any(arg < 0.0):
        raise ValueError(
            "delta is below its lower bound -(1 - vs0^2/vp0^2)/2; no real C13 exists"
        )
    c13 = np.sqrt(arg) - c44
    return Stiffness(c11, c13, c33, c44, c66)


def cij_to_thomsen(c11, c13, c33, c44, c66, rho):
    """Thomsen parameters ``(vp0, vs0, eps, delta, gamma)`` from stiffnesses.

    gamma is set to 0 where C44 = 0 (fluid).  The relation assumes
    C13 + C44 >= 0; :func:`thomsen_to_cij` then returns the same C13.
    """
    c11, c13, c33, c44, c66, rho = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (c11, c13, c33, c44, c66, rho))
    )
    vp0 = np.sqrt(c33 / rho)
    vs0 = np.sqrt(c44 / rho)
    eps = (c11 - c33) / (2.0 * c33)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(c44 > 0.0, (c66 - c44) / (2.0 * c44), 0.0)
    delta = ((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44))
    return vp0, vs0, eps, delta, gamma


def voigt_matrix(c11, c13, c33, c44, c66) -> np.ndarray:
    """6x6 Voigt stiffness matrices, shape ``(..., 6, 6)``."""
    c11, c13, c33, c44, c66 = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (c11, c13, c33, c44, c66))
    )
    c12 = c11 - 2.0 * c66
    m = np.zeros(c11.shape + (6, 6))
    m[..., 0, 0] = m[..., 1, 1] = c11
    m[..., 2, 2] = c33
    m[..., 0, 1] = m[..., 1, 0] = c12
    m[..., 0, 2] = m[..., 2, 0] = m[..., 1, 2] = m[..., 2, 1] = c13
    m[..., 3, 3] = m[..., 4, 4] = c44
    m[..., 5, 5] = c66
    return m


def is_stable(c11, c13, c33, c44, c66, allow_fluid: bool = True, rtol: float = 1e-9):
    """True where the stiffness is physically admissible.

    A solid is admissible when its Voigt matrix is positive definite (strain
    energy > 0 for every strain).  With ``allow_fluid`` an ideal fluid
    (C44 = C66 = 0, C11 = C13 = C33 > 0) is also accepted.
    """
    c11, c13, c33, c44, c66 = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (c11, c13, c33, c44, c66))
    )
    eig = np.linalg.eigvalsh(voigt_matrix(c11, c13, c33, c44, c66))
    scale = np.maximum(np.abs(c33), 1.0)
    solid = np.all(eig > rtol * scale[..., None], axis=-1)
    if not allow_fluid:
        return solid
    tol = rtol * scale
    fluid = (
        (c33 > 0.0)
        & (np.abs(c44) <= tol)
        & (np.abs(c66) <= tol)
        & (np.abs(c11 - c33) <= tol)
        & (np.abs(c13 - c33) <= tol)
    )
    return solid | fluid


def phase_velocities(theta, vp0, vs0, eps, delta, gamma):
    """Exact VTI phase velocities ``(vp, vsv, vsh)`` at phase angle ``theta`` (rad).

    ``theta`` is measured from the vertical symmetry axis.  The qP and qSV
    velocities solve the Christoffel equation for a VTI medium (Thomsen,
    1986); the squared SH velocity is ``Vs0^2 (1 + 2 gamma sin^2 theta)``.
    """
    theta, vp0, vs0, eps, delta, gamma = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (theta, vp0, vs0, eps, delta, gamma))
    )
    a = thomsen_to_cij(vp0, vs0, 1.0, eps, delta, gamma)
    s2 = np.sin(theta) ** 2
    c2 = np.cos(theta) ** 2
    b = (a.c11 + a.c44) * s2 + (a.c33 + a.c44) * c2
    d = np.sqrt(
        ((a.c11 - a.c44) * s2 - (a.c33 - a.c44) * c2) ** 2
        + 4.0 * (a.c13 + a.c44) ** 2 * s2 * c2
    )
    vp = np.sqrt(0.5 * (b + d))
    vsv = np.sqrt(np.maximum(0.5 * (b - d), 0.0))
    vsh = np.sqrt(a.c66 * s2 + a.c44 * c2)
    return vp, vsv, vsh


def backus_average(thickness, stiffness: Stiffness, rho):
    """Backus (1962) average of a stack of VTI layers.

    Returns ``(Stiffness, rho)`` of the long-wavelength equivalent medium.  All
    layers must be solid, or all fluid; a mixed fluid-solid stack has no VTI
    equivalent with finite Thomsen gamma.
    """
    h = np.asarray(thickness, dtype=float)
    w = h / h.sum()
    c11, c13, c33, c44, c66 = (np.asarray(c, dtype=float) for c in stiffness)
    rho = np.asarray(rho, dtype=float)
    fluid = c44 <= 0.0
    if fluid.any() and not fluid.all():
        raise ValueError("backus_average: cannot mix fluid and solid layers")
    c33s = 1.0 / np.sum(w / c33)
    c13s = np.sum(w * c13 / c33) * c33s
    c11s = np.sum(w * (c11 - c13**2 / c33)) + np.sum(w * c13 / c33) ** 2 * c33s
    c44s = 0.0 if fluid.all() else 1.0 / np.sum(w / c44)
    c66s = np.sum(w * c66)
    return Stiffness(c11s, c13s, c33s, c44s, c66s), float(np.sum(w * rho))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

_PROPS = ("vp0", "vs0", "rho", "eps", "delta", "gamma", "qp", "qs")


def _inv(x):
    """1/x with 1/inf = 0 and 1/0 = inf, without warnings."""
    with np.errstate(divide="ignore"):
        return 1.0 / np.asarray(x, dtype=float)


@dataclass(eq=False)
class ElasticModel1D:
    """A stack of homogeneous VTI layers with attenuation.

    Parameters
    ----------
    interfaces : (n+1,) array
        Depths of the layer boundaries (m, positive down, strictly increasing).
        Layer ``i`` spans ``interfaces[i]`` to ``interfaces[i+1]``.
    vp0, vs0, rho, eps, delta, gamma, qp, qs : (n,) arrays
        Layer properties (see module docstring).  Scalars are broadcast.
    label : (n,) array of str, optional
        Unit name per layer.
    f_ref : float
        Frequency (Hz) at which ``vp0`` and ``vs0`` apply (constant-Q
        dispersion, :mod:`iwt.qmodel`).
    """

    interfaces: np.ndarray
    vp0: np.ndarray
    vs0: np.ndarray
    rho: np.ndarray
    eps: np.ndarray = 0.0
    delta: np.ndarray = 0.0
    gamma: np.ndarray = 0.0
    qp: np.ndarray = np.inf
    qs: np.ndarray = np.inf
    label: Optional[np.ndarray] = None
    f_ref: float = 30.0

    def __post_init__(self):
        z = np.asarray(self.interfaces, dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError("interfaces must be a 1D array with at least two depths")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError("interfaces must be strictly increasing")
        self.interfaces = z
        n = z.size - 1
        for name in _PROPS:
            v = np.asarray(getattr(self, name), dtype=float)
            v = np.broadcast_to(v, (n,)).copy() if v.ndim == 0 else v.copy()
            if v.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {v.shape}")
            setattr(self, name, v)
        if self.label is None:
            self.label = np.array([""] * n, dtype=object)
        else:
            lab = np.asarray(self.label, dtype=object)
            if lab.shape != (n,):
                raise ValueError(f"label must have shape ({n},)")
            self.label = lab.copy()
        self.f_ref = float(self.f_ref)

    # -- basic geometry ----------------------------------------------------

    @property
    def n_layers(self) -> int:
        return self.interfaces.size - 1

    def __len__(self) -> int:
        return self.n_layers

    @property
    def top(self) -> np.ndarray:
        return self.interfaces[:-1]

    @property
    def bottom(self) -> np.ndarray:
        return self.interfaces[1:]

    @property
    def thickness(self) -> np.ndarray:
        return np.diff(self.interfaces)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.interfaces[:-1] + self.interfaces[1:])

    @property
    def is_fluid(self) -> np.ndarray:
        return self.vs0 <= 0.0

    def layer_index(self, z) -> np.ndarray:
        """Index of the layer containing depth ``z``.

        A depth on an interface belongs to the layer below it; the model's
        bottom depth belongs to the last layer.
        """
        z = np.asarray(z, dtype=float)
        if np.any((z < self.interfaces[0]) | (z > self.interfaces[-1])):
            raise ValueError("depth outside the model")
        i = np.searchsorted(self.interfaces, z, side="right") - 1
        return np.clip(i, 0, self.n_layers - 1)

    def at(self, z) -> dict:
        """Properties of the layers containing depths ``z``."""
        i = self.layer_index(z)
        out = {name: getattr(self, name)[i] for name in _PROPS}
        out["label"] = self.label[i]
        return out

    # -- stiffness and checks ----------------------------------------------

    def cij(self) -> Stiffness:
        """Stiffnesses (Pa) of every layer."""
        return thomsen_to_cij(self.vp0, self.vs0, self.rho, self.eps, self.delta, self.gamma)

    @classmethod
    def from_cij(cls, interfaces, stiffness: Stiffness, rho, qp=np.inf, qs=np.inf,
                 label=None, f_ref=30.0) -> "ElasticModel1D":
        vp0, vs0, eps, delta, gamma = cij_to_thomsen(*stiffness, rho)
        return cls(interfaces, vp0, vs0, rho, eps, delta, gamma, qp, qs, label, f_ref)

    def problems(self) -> list:
        """Human-readable list of physically invalid layers (empty if valid)."""
        out = []

        def _flag(mask, text):
            idx = np.flatnonzero(mask)
            if idx.size:
                out.append(f"{text}: layers {idx[:10].tolist()}"
                           + (" ..." if idx.size > 10 else ""))

        _flag(~(self.vp0 > 0.0), "vp0 must be positive")
        _flag(~(self.rho > 0.0), "rho must be positive")
        _flag(~(self.vs0 >= 0.0), "vs0 must be non-negative")
        _flag(~(self.vs0 < self.vp0), "vs0 must be below vp0")
        _flag(~(self.qp > 0.0) | ~(self.qs > 0.0), "Q must be positive")
        fl = self.is_fluid
        _flag(fl & ((self.eps != 0.0) | (self.delta != 0.0) | (self.gamma != 0.0)),
              "fluid layers must be isotropic")
        bad_delta = ~fl & (self.delta < delta_min(self.vp0, np.where(fl, 0.0, self.vs0)))
        _flag(bad_delta, "delta below its lower bound")
        ok = ~bad_delta & (self.vp0 > 0.0) & (self.rho > 0.0) & (self.vs0 < self.vp0)
        if ok.any():
            c = thomsen_to_cij(self.vp0[ok], self.vs0[ok], self.rho[ok], self.eps[ok],
                               self.delta[ok], self.gamma[ok])
            stable = np.ones(self.n_layers, dtype=bool)
            stable[ok] = is_stable(*c)
            _flag(ok & ~stable, "stiffness not positive definite")
        return out

    def validate(self) -> "ElasticModel1D":
        """Raise ``ValueError`` listing every invalid layer; return ``self`` if valid."""
        p = self.problems()
        if p:
            raise ValueError("invalid model: " + "; ".join(p))
        return self

    # -- traveltimes -------------------------------------------------------

    def interface_times(self, wave: str = "P") -> np.ndarray:
        """One-way vertical traveltime (s) from the model top to each interface.

        S times through a fluid layer are infinite.
        """
        v = self._vertical_velocity(wave)
        return np.concatenate([[0.0], np.cumsum(self.thickness * _inv(v))])

    def vertical_time(self, z, wave: str = "P") -> np.ndarray:
        """One-way vertical traveltime (s) from the model top to depth ``z``.

        Exact for the piecewise-constant model, at the reference frequency.
        """
        z = np.asarray(z, dtype=float)
        i = self.layer_index(z)
        t = self.interface_times(wave)
        slow = _inv(self._vertical_velocity(wave))
        dz = z - self.interfaces[i]
        with np.errstate(invalid="ignore"):
            return t[i] + np.where(dz > 0.0, dz * slow[i], 0.0)

    def twt(self, z) -> np.ndarray:
        """Two-way vertical P traveltime (s) to depth ``z``."""
        return 2.0 * self.vertical_time(z, "P")

    def depth_at_time(self, t, wave: str = "P") -> np.ndarray:
        """Depth reached after one-way vertical time ``t`` (inverse of vertical_time)."""
        times = self.interface_times(wave)
        if not np.all(np.isfinite(times)):
            raise ValueError(f"{wave} time is infinite in a fluid layer")
        t = np.asarray(t, dtype=float)
        if np.any((t < 0.0) | (t > times[-1])):
            raise ValueError("time outside the model")
        return np.interp(t, times, self.interfaces)

    def _vertical_velocity(self, wave: str) -> np.ndarray:
        if wave.upper() == "P":
            return self.vp0
        if wave.upper() == "S":
            return self.vs0
        raise ValueError("wave must be 'P' or 'S'")

    # -- restructuring -----------------------------------------------------

    def _take(self, idx, interfaces) -> "ElasticModel1D":
        props = {name: getattr(self, name)[idx] for name in _PROPS}
        return ElasticModel1D(interfaces, label=self.label[idx], f_ref=self.f_ref, **props)

    def split_at(self, depths) -> "ElasticModel1D":
        """Insert interfaces at ``depths`` (inside the model); properties are copied."""
        d = np.asarray(depths, dtype=float).ravel()
        d = d[(d > self.interfaces[0]) & (d < self.interfaces[-1])]
        z = np.union1d(self.interfaces, d)
        return self._take(self.layer_index(0.5 * (z[:-1] + z[1:])), z)

    def subset(self, top: float, bottom: float) -> "ElasticModel1D":
        """The part of the model between depths ``top`` and ``bottom``."""
        if not (self.interfaces[0] <= top < bottom <= self.interfaces[-1]):
            raise ValueError("subset bounds must lie inside the model")
        m = self.split_at([top, bottom])
        keep = (m.mid > top) & (m.mid < bottom)
        z = np.concatenate([m.top[keep], [m.bottom[keep][-1]]])
        return m._take(np.flatnonzero(keep), z)

    def regular(self, dz: float, top: Optional[float] = None,
                bottom: Optional[float] = None) -> "ElasticModel1D":
        """Point-sample the model onto cells of thickness ``dz``.

        Each cell takes the properties of the layer at its midpoint, so the
        result is exact only where interfaces fall on cell boundaries.  Use
        :meth:`block` or :meth:`backus` to average instead.
        """
        top = self.interfaces[0] if top is None else float(top)
        bottom = self.interfaces[-1] if bottom is None else float(bottom)
        n = int(round((bottom - top) / dz))
        if n < 1 or not np.isclose(top + n * dz, bottom):
            raise ValueError("(bottom - top) must be a positive multiple of dz")
        z = top + dz * np.arange(n + 1)
        return self._take(self.layer_index(0.5 * (z[:-1] + z[1:])), z)

    def _q_average(self, w_slow, w_slow_q):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_q = np.where(w_slow > 0.0, w_slow_q / w_slow, 0.0)
        return _inv(inv_q)

    def block(self, boundaries: Sequence[float]) -> "ElasticModel1D":
        """Average the model into blocks between ``boundaries`` (Backus, 1962).

        The model top and bottom, and every fluid-solid boundary (e.g. the
        seabed), are added to ``boundaries``.  Density is the thickness-weighted
        mean; Q is averaged so that vertical traveltime attenuation is
        preserved; the label is the one with the largest total thickness.
        """
        fl = self.is_fluid
        b = np.union1d(np.asarray(boundaries, dtype=float),
                       np.concatenate([[self.interfaces[0], self.interfaces[-1]],
                                       self.interfaces[1:-1][fl[1:] != fl[:-1]]]))
        b = b[(b >= self.interfaces[0]) & (b <= self.interfaces[-1])]
        m = self.split_at(b)
        block_id = np.searchsorted(b, m.mid) - 1
        h = m.thickness
        c = m.cij()
        cols = {name: np.empty(b.size - 1) for name in _PROPS}
        labels = np.empty(b.size - 1, dtype=object)
        sp, ss = _inv(m.vp0), _inv(m.vs0)
        for k in range(b.size - 1):
            sel = block_id == k
            stiff, rho = backus_average(h[sel], Stiffness(*(x[sel] for x in c)), m.rho[sel])
            vp0, vs0, eps, delta, gamma = cij_to_thomsen(*stiff, rho)
            if m.is_fluid[sel].all():
                # rounding can leave eps, delta of order 1e-16 in a fluid
                eps = delta = gamma = 0.0
            cols["vp0"][k], cols["vs0"][k], cols["rho"][k] = vp0, vs0, rho
            cols["eps"][k], cols["delta"][k], cols["gamma"][k] = eps, delta, gamma
            hs = h[sel]
            cols["qp"][k] = self._q_average(np.sum(hs * sp[sel]),
                                            np.sum(hs * sp[sel] * _inv(m.qp[sel])))
            if m.is_fluid[sel].all():
                cols["qs"][k] = np.inf
            else:
                cols["qs"][k] = self._q_average(np.sum(hs * ss[sel]),
                                                np.sum(hs * ss[sel] * _inv(m.qs[sel])))
            labs = m.label[sel]
            names = list(dict.fromkeys(labs))
            totals = [hs[labs == name].sum() for name in names]
            labels[k] = names[int(np.argmax(totals))]
        return ElasticModel1D(b, label=labels, f_ref=self.f_ref, **cols)

    def window_mean(self, values, centers, length: float,
                    segments: Optional[Sequence[float]] = None) -> np.ndarray:
        """Thickness-weighted mean of per-layer ``values`` over windows.

        Each window spans ``center +- length/2``, truncated at the model top
        and bottom and at any depth in ``segments``, so that it never crosses
        those depths.  The mean is exact for the piecewise-constant model.
        """
        values = np.asarray(values, dtype=float)
        centers = np.asarray(centers, dtype=float)
        cum = np.concatenate([[0.0], np.cumsum(self.thickness * values)])
        edges = np.asarray([self.interfaces[0], self.interfaces[-1]], dtype=float)
        if segments is not None:
            edges = np.union1d(edges, np.asarray(segments, dtype=float))
            edges = edges[(edges >= self.interfaces[0]) & (edges <= self.interfaces[-1])]
        seg = np.clip(np.searchsorted(edges, centers, side="right") - 1, 0, edges.size - 2)
        a = np.maximum(centers - 0.5 * length, edges[seg])
        b = np.minimum(centers + 0.5 * length, edges[seg + 1])
        fa = np.interp(a, self.interfaces, cum)
        fb = np.interp(b, self.interfaces, cum)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (fb - fa) / (b - a)
        # Zero-length windows (length 0) take the value of the layer itself.
        zero = ~(b > a)
        if zero.any():
            mean[zero] = values[self.layer_index(centers[zero])]
        return mean

    def backus(self, window: float, respect: Optional[Sequence[float]] = None) -> "ElasticModel1D":
        """Running Backus average over a window of length ``window`` (m).

        Every layer is replaced by the Backus average of the model within
        ``window`` centred on its midpoint; the interfaces are unchanged.  The
        window never crosses a fluid-solid boundary or any depth in
        ``respect`` (for example the seabed or a unit top).  A window of a
        few metres to about a third of the shortest wavelength gives the
        medium "seen" by the seismic wave.
        """
        fl = self.is_fluid
        change = self.interfaces[1:-1][fl[1:] != fl[:-1]]
        seg = change if respect is None else np.union1d(change, np.asarray(respect, float))
        c = self.cij()
        x = self.mid

        def mean(v):
            return self.window_mean(v, x, window, seg)

        inv_c33 = mean(1.0 / c.c33)
        c33s = 1.0 / inv_c33
        r13 = mean(c.c13 / c.c33)
        c13s = r13 * c33s
        c11s = mean(c.c11 - c.c13**2 / c.c33) + r13**2 * c33s
        with np.errstate(divide="ignore"):
            inv_c44 = mean(np.where(fl, 0.0, 1.0 / np.where(fl, 1.0, c.c44)))
        c44s = np.where(fl, 0.0, _inv(inv_c44))
        c66s = np.where(fl, 0.0, mean(c.c66))
        rhos = mean(self.rho)
        sp = _inv(self.vp0)
        qp = self._q_average(mean(sp), mean(sp * _inv(self.qp)))
        ss = np.where(fl, 0.0, _inv(np.where(fl, 1.0, self.vs0)))
        qs = np.where(fl, np.inf, self._q_average(mean(ss), mean(ss * _inv(self.qs))))
        vp0, vs0, eps, delta, gamma = cij_to_thomsen(c11s, c13s, c33s, c44s, c66s, rhos)
        # Fluid layers: rounding can leave eps, delta of order 1e-16.
        eps, delta = np.where(fl, 0.0, eps), np.where(fl, 0.0, delta)
        return ElasticModel1D(self.interfaces.copy(), vp0, vs0, rhos, eps, delta, gamma,
                              qp, qs, self.label.copy(), self.f_ref)

    def copy(self) -> "ElasticModel1D":
        return self._take(np.arange(self.n_layers), self.interfaces.copy())

    def as_dict(self) -> dict:
        """Per-layer columns, including top and bottom depths."""
        d = {"top": self.top.copy(), "bottom": self.bottom.copy()}
        d.update({name: getattr(self, name).copy() for name in _PROPS})
        d["label"] = self.label.copy()
        return d
