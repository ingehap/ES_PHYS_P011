"""A synthetic "Wisting-like" well: true earth model and wireline logs.

The Hanssen data behind Lewis et al. (2016) are proprietary.  This module builds a
well with the same *structure*, with known true properties against which every
estimator in the package can be checked.

What is taken from public sources (:mod:`iwt.paper`):

* water depth 417.5 m; TD 1679 m TVDSS in the Snadd Formation;
* top Stø at 672 m TVDSS (254.5 m below the seabed; the article says "some 250 m");
* a 20-25 m gross oil column in the Stø (here 22.5 m);
* the stratigraphic order Hekkingen - Fuglen - Stø - Nordmela - Fruholmen - Snadd.

What is assumed (and labelled as such): every other formation top, every
velocity, density, anisotropy and Q value, the thin calcite-cemented stringers,
and the fine-scale heterogeneity.  The values are plausible for shallow, uplifted
Barents Sea clastics, but they are not Wisting measurements.  The overburden
shales are given strong VTI anisotropy and low Q, which the article reports
(p. 3) without giving numbers.

The model is sampled at 0.5 m below the seabed (the article's model sampling,
p. 3); the water column is a single layer.

The Stø oil leg is derived from the water leg by Gassmann (1951) substitution
(:mod:`iwt.rockphysics`), so the two legs share one rock frame and one
fine-scale heterogeneity field.

Logs are "measured" from the true model as follows:

* the sonic and dipole velocities are dispersed from the model's reference
  frequency to the tool frequency by the constant-Q law of :mod:`iwt.qmodel`;
* tools average slowness (sonic) or density over their aperture;
* noise is added.

In a vertical VTI well the monopole sonic measures Vp0 and both dipole modes
measure Vs0.

For WP4, :func:`caliper_log` adds washouts in the overburden shales and
:func:`stoneley_log` a Stoneley slowness log, computed with the exact
borehole mode of :mod:`iwt.sonic` in the running Backus average of the model
over the array aperture, at the Stoneley frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from . import paper
from .model import ElasticModel1D
from .qmodel import phase_velocity
from .rockphysics import fluid_substitution, reuss_fluid

__all__ = [
    "Unit",
    "Stringer",
    "WellLogs",
    "SEA_WATER",
    "WISTING_LIKE_UNITS",
    "STRINGERS",
    "OWC_TVDSS",
    "STO_PHI",
    "wisting_like_model",
    "synthetic_logs",
    "wisting_like_well",
    "WASHOUT",
    "caliper_log",
    "StoneleyLog",
    "stoneley_log",
]


@dataclass(frozen=True)
class Unit:
    """A stratigraphic unit with uniform intrinsic properties.

    ``vp0``, ``vs0`` and ``rho`` apply at the unit top; ``grad`` (1/m) increases
    velocities by the factor ``1 + grad (z - top)`` and density by
    ``1 + grad (z - top) / 4``.  ``sigma`` is the relative standard deviation of
    the fine-scale Vp fluctuations (Vs: 1.3 sigma, density: 0.35 sigma), with
    exponential correlation length ``corr_length`` (m).  Units with the same
    ``body`` share one fluctuation field (one rock across a fluid contact).
    """

    name: str
    top: float
    vp0: float
    vs0: float
    rho: float
    eps: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    qp: float = np.inf
    qs: float = np.inf
    grad: float = 0.0
    sigma: float = 0.0
    corr_length: float = 1.0
    body: str = ""
    fluid: str = "brine"


@dataclass(frozen=True)
class Stringer:
    """A thin isotropic bed inserted into the model (e.g. a calcite-cemented stringer)."""

    top: float
    thickness: float
    vp0: float = 4300.0
    vs0: float = 2350.0
    rho: float = 2620.0
    qp: float = 150.0
    qs: float = 100.0


#: Sea water (Barents Sea bottom water, near 0 deg C; assumed).
SEA_WATER = Unit("Sea water", 0.0, 1470.0, 0.0, 1028.0, fluid="water")

# -- Stø reservoir: water leg (assumed) and Gassmann-derived oil leg ---------

#: Assumed Stø porosity and mineral/fluid moduli (Pa) and densities (kg/m^3).
#: Quartz K = 36.6 GPa (Mavko et al., 2009); cold, low-pressure brine and light
#: oil (reservoir at about 7 MPa and below 25 deg C: assumed).
STO_PHI = 0.27
K_QUARTZ = 36.6e9
K_BRINE, RHO_BRINE = 2.35e9, 1030.0
K_OIL, RHO_OIL = 1.2e9, 840.0
STO_SW_OIL_LEG = 0.15

#: Oil-water contact: top Stø plus the middle of the published 20-25 m gross column.
OWC_TVDSS = paper.STO_TOP_TVDSS + 0.5 * sum(paper.OIL_COLUMN_GROSS)

_STO_WATER = Unit("Stø", OWC_TVDSS, 3050.0, 1650.0, 2210.0, 0.02, 0.01, 0.02,
                  qp=90.0, qs=60.0, sigma=0.03, corr_length=1.5, body="Stø")


def _oil_leg(water: Unit, top: float) -> Unit:
    k_fl, rho_fl = reuss_fluid([K_BRINE, K_OIL], [RHO_BRINE, RHO_OIL],
                               [STO_SW_OIL_LEG, 1.0 - STO_SW_OIL_LEG])
    vp, vs, rho = fluid_substitution(water.vp0, water.vs0, water.rho, STO_PHI, K_QUARTZ,
                                     K_BRINE, RHO_BRINE, k_fl, rho_fl)
    return replace(water, name="Stø", top=top, vp0=float(vp), vs0=float(vs),
                   rho=float(rho), fluid="oil")


_STO_OIL = _oil_leg(_STO_WATER, paper.STO_TOP_TVDSS)

#: The units below the seabed, top down.  Tops other than the seabed, top Stø,
#: the OWC and TD are schematic; all property values are assumptions.
WISTING_LIKE_UNITS = (
    SEA_WATER,
    Unit("Quaternary", paper.WATER_DEPTH, 1750.0, 450.0, 1950.0, 0.02, 0.01, 0.03,
         qp=30.0, qs=15.0, sigma=0.02, corr_length=1.0),
    Unit("Lower Cretaceous shale", 440.0, 2400.0, 950.0, 2280.0, 0.20, 0.08, 0.24,
         qp=45.0, qs=25.0, grad=2.0e-4, sigma=0.03, corr_length=1.5),
    Unit("Hekkingen", 600.0, 2300.0, 1050.0, 2180.0, 0.26, 0.10, 0.30,
         qp=40.0, qs=25.0, sigma=0.04, corr_length=1.0),
    Unit("Fuglen", 631.0, 2800.0, 1300.0, 2450.0, 0.12, 0.05, 0.14,
         qp=60.0, qs=35.0, sigma=0.04, corr_length=1.0),
    _STO_OIL,
    _STO_WATER,
    Unit("Nordmela", 702.0, 2950.0, 1480.0, 2350.0, 0.08, 0.03, 0.09,
         qp=70.0, qs=45.0, sigma=0.05, corr_length=0.8),
    Unit("Fruholmen", 740.0, 3350.0, 1800.0, 2420.0, 0.05, 0.02, 0.06,
         qp=80.0, qs=50.0, sigma=0.04, corr_length=1.5),
    Unit("Snadd", 800.0, 3150.0, 1600.0, 2450.0, 0.10, 0.04, 0.12,
         qp=70.0, qs=40.0, grad=1.5e-4, sigma=0.04, corr_length=2.0),
)

#: Thin calcite-cemented stringers (assumed), common in Barents Sea shales.
#: They are strong short-period reflectors, which matters for interbed
#: multiples and stratigraphic filtering (later work packages).
STRINGERS = (
    Stringer(505.0, 1.5),
    Stringer(548.0, 2.0),
    Stringer(583.0, 1.0),
    Stringer(648.0, 1.0),
    Stringer(661.0, 1.5),
)


def _ar1(n: int, dz: float, corr_length: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) series: exponential correlation with length ``corr_length``."""
    a = np.exp(-dz / corr_length) if corr_length > 0.0 else 0.0
    w = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = w[0]
    s = np.sqrt(1.0 - a * a)
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * w[i]
    return x


def wisting_like_model(dz: float = paper.MODEL_SAMPLING, td: float = paper.TD_TVDSS,
                       units: Sequence[Unit] = WISTING_LIKE_UNITS,
                       stringers: Sequence[Stringer] = STRINGERS,
                       heterogeneity: bool = True, seed: int = 2016,
                       f_ref: float = 30.0) -> ElasticModel1D:
    """The true earth model of the synthetic well.

    The first unit must be the water column (a single layer); below it the model
    is sampled every ``dz`` metres down to ``td``.  ``heterogeneity=False`` gives
    the blocky model without fine-scale fluctuations or stringers.
    """
    if units[0].vs0 != 0.0:
        raise ValueError("the first unit must be the water column")
    seabed = units[1].top
    tops = np.array([u.top for u in units])
    if np.any(np.diff(tops) <= 0.0):
        raise ValueError("unit tops must increase")
    n = int(round((td - seabed) / dz))
    if not np.isclose(seabed + n * dz, td):
        raise ValueError("td - seabed must be a multiple of dz")
    z = seabed + dz * np.arange(n + 1)
    mid = 0.5 * (z[:-1] + z[1:])
    k = np.searchsorted(tops, mid, side="right") - 1

    cols = {name: np.empty(n) for name in
            ("vp0", "vs0", "rho", "eps", "delta", "gamma", "qp", "qs")}
    label = np.empty(n, dtype=object)
    rng = np.random.default_rng(seed)
    fields = {}
    for j, u in enumerate(units[1:], start=1):
        sel = k == j
        if not sel.any():
            continue
        depth = mid[sel] - u.top
        f = 1.0 + u.grad * depth
        x = np.zeros(sel.sum())
        if heterogeneity and u.sigma > 0.0:
            key = u.body or u.name
            if key not in fields:
                body_sel = np.isin(k, [i for i, w in enumerate(units) if (w.body or w.name) == key])
                fields[key] = (np.flatnonzero(body_sel),
                               np.clip(_ar1(body_sel.sum(), dz, u.corr_length, rng), -3.0, 3.0))
            idx, xf = fields[key]
            x = xf[np.searchsorted(idx, np.flatnonzero(sel))]
        cols["vp0"][sel] = u.vp0 * f * (1.0 + u.sigma * x)
        cols["vs0"][sel] = u.vs0 * f * (1.0 + 1.3 * u.sigma * x)
        cols["rho"][sel] = u.rho * (1.0 + 0.25 * u.grad * depth) * (1.0 + 0.35 * u.sigma * x)
        for name in ("eps", "delta", "gamma", "qp", "qs"):
            cols[name][sel] = getattr(u, name)
        label[sel] = u.name if u.fluid != "oil" else f"{u.name} (oil)"

    if heterogeneity:
        for s in stringers:
            sel = (mid > s.top) & (mid < s.top + s.thickness)
            cols["vp0"][sel], cols["vs0"][sel], cols["rho"][sel] = s.vp0, s.vs0, s.rho
            cols["eps"][sel] = cols["delta"][sel] = cols["gamma"][sel] = 0.0
            cols["qp"][sel], cols["qs"][sel] = s.qp, s.qs
            label[sel] = [f"{lab} stringer" for lab in label[sel]]

    w = units[0]
    interfaces = np.concatenate([[w.top], z])
    full = {name: np.concatenate([[getattr(w, name)], v]) for name, v in cols.items()}
    labels = np.concatenate([np.array([w.name], dtype=object), label])
    return ElasticModel1D(interfaces, label=labels, f_ref=f_ref, **full).validate()


@dataclass
class WellLogs:
    """Wireline logs in SI units, indexed by depth (m TVDSS).

    ``vp`` and ``vs`` are the velocities the tools report (at their own
    frequencies), ``rho`` the bulk density.  ``meta`` records how the logs were
    made.
    """

    depth: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    meta: dict = field(default_factory=dict)

    def las_curves(self, kb: float = paper.KB_ELEVATION) -> dict:
        """Curves in customary log units: DEPT (m RKB, vertical well), DTCO and DTSM
        (us/ft), RHOB (g/cm^3), TVDSS (m)."""
        us_per_ft = 1.0e6 * 0.3048
        return {
            "DEPT": self.depth + kb,
            "TVDSS": self.depth.copy(),
            "DTCO": us_per_ft / self.vp,
            "DTSM": us_per_ft / self.vs,
            "RHOB": self.rho / 1000.0,
        }


def synthetic_logs(model: ElasticModel1D, top: float = 470.0, bottom: float = 1670.0,
                   dz: float = paper.MODEL_SAMPLING, f_sonic_p: float = 12.0e3,
                   f_sonic_s: float = 2.0e3, aperture_p: float = 0.6,
                   aperture_s: float = 1.0, aperture_rho: float = 0.45,
                   noise_p: float = 0.005, noise_s: float = 0.01,
                   noise_rho: float = 15.0, seed: Optional[int] = 7) -> WellLogs:
    """Wireline logs "measured" in a vertical well through ``model``.

    ``top`` is the top of the open-hole log (assumed below the conductor and
    surface casing).  Velocities are dispersed from ``model.f_ref`` to the tool
    frequencies ``f_sonic_p`` (monopole) and ``f_sonic_s`` (dipole flexural);
    the logs are slowness (or density) averages over the tool apertures (m),
    with relative noise on slowness and absolute noise (kg/m^3) on density.
    ``seed=None`` gives noise-free logs.
    """
    if top < model.interfaces[0] or bottom > model.interfaces[-1]:
        raise ValueError("log interval outside the model")
    if model.is_fluid[model.layer_index([top, bottom])].any():
        raise ValueError("the log interval must start below the seabed")
    depth = top + dz * np.arange(int(round((bottom - top) / dz)) + 1)
    fluid = model.is_fluid
    vp_f = phase_velocity(model.vp0, model.qp, f_sonic_p, model.f_ref)
    vs_f = np.where(fluid, 0.0, phase_velocity(np.where(fluid, 1.0, model.vs0),
                                                 model.qs, f_sonic_s, model.f_ref))
    sp = model.window_mean(1.0 / vp_f, depth, aperture_p)
    with np.errstate(divide="ignore"):
        ss = model.window_mean(np.where(fluid, 0.0, 1.0 / np.where(fluid, 1.0, vs_f)),
                               depth, aperture_s)
    rho = model.window_mean(model.rho, depth, aperture_rho)
    if seed is not None:
        rng = np.random.default_rng(seed)
        sp = sp * (1.0 + noise_p * rng.standard_normal(depth.size))
        ss = ss * (1.0 + noise_s * rng.standard_normal(depth.size))
        rho = rho + noise_rho * rng.standard_normal(depth.size)
    meta = dict(f_sonic_p=f_sonic_p, f_sonic_s=f_sonic_s, f_ref=model.f_ref,
                aperture_p=aperture_p, aperture_s=aperture_s, aperture_rho=aperture_rho,
                noise_p=noise_p, noise_s=noise_s, noise_rho=noise_rho, seed=seed)
    return WellLogs(depth, 1.0 / sp, 1.0 / ss, rho, meta)


def wisting_like_well(seed: int = 2016, log_seed: Optional[int] = 7):
    """``(model, logs)`` for the default synthetic well."""
    model = wisting_like_model(seed=seed)
    return model, synthetic_logs(model, seed=log_seed)


#: Assumed washout (fractional enlargement of the bit radius) per unit: the soft,
#: shallow shales wash out, the sands and the Snadd hardly.
WASHOUT = {"Lower Cretaceous shale": 0.06, "Hekkingen": 0.08, "Fuglen": 0.03}


def caliper_log(model: ElasticModel1D, depth, bit_radius: float = 0.108,
                washout: Optional[dict] = None, noise: float = 0.001,
                seed: Optional[int] = 5):
    """``(radius, caliper)``: true borehole radius and its measurement (m) at ``depth``.

    The enlargement in a unit is ``washout[unit]`` times a positive, smooth
    random factor (mean about 1, correlation length 2 m); ``noise`` (m) is the
    caliper's standard deviation.
    """
    washout = WASHOUT if washout is None else washout
    depth = np.asarray(depth, dtype=float)
    rng = np.random.default_rng(seed)
    dz = float(np.median(np.diff(depth))) if depth.size > 1 else 1.0
    factor = np.clip(1.0 + 0.5 * _ar1(depth.size, dz, 2.0, rng), 0.0, None)
    names = np.array([str(lab).replace(" stringer", "").replace(" (oil)", "")
                      for lab in model.at(depth)["label"]])
    frac = np.array([washout.get(n, 0.0) for n in names])
    radius = bit_radius * (1.0 + frac * factor)
    caliper = radius + noise * rng.standard_normal(depth.size)
    return radius, caliper


@dataclass
class StoneleyLog:
    """Stoneley phase slowness (s/m) and radiation attenuation 1/Q at ``frequency`` (Hz)."""

    depth: np.ndarray
    slowness: np.ndarray
    inv_q: np.ndarray
    frequency: float
    meta: dict = field(default_factory=dict)


def stoneley_log(model: ElasticModel1D, depth, mud, radius, tool_radius: float = 0.046,
                 tool_modulus: float = np.inf, frequency: float = 500.0,
                 aperture: float = 1.8, noise: float = 0.003,
                 seed: Optional[int] = 11) -> StoneleyLog:
    """Stoneley slowness log "measured" in a vertical well through ``model``.

    The model is dispersed from ``model.f_ref`` to ``frequency`` (constant Q),
    Backus averaged over the array ``aperture`` (m), and the exact Stoneley mode
    of the TI formation is computed with :func:`iwt.sonic.stoneley_slowness` for
    the ``mud`` (:class:`iwt.sonic.Mud`), borehole ``radius`` (m, scalar or per
    depth) and a concentric tool.  ``noise`` is the relative standard deviation
    of the slowness.  Where the mode cannot be followed the log is NaN.
    """
    from .model import thomsen_to_cij
    from .sonic import stoneley_slowness

    depth = np.asarray(depth, dtype=float)
    m = model.copy()
    m.vp0 = phase_velocity(model.vp0, model.qp, frequency, model.f_ref)
    m.vs0 = np.where(model.is_fluid, 0.0,
                     phase_velocity(np.where(model.is_fluid, 1.0, model.vs0), model.qs, frequency,
                                    model.f_ref))
    eff = m.backus(aperture, respect=[model.interfaces[np.argmax(~model.is_fluid)]]).at(depth)
    c = thomsen_to_cij(eff["vp0"], eff["vs0"], eff["rho"], eff["eps"], eff["delta"], eff["gamma"])
    rad = np.broadcast_to(np.asarray(radius, dtype=float), depth.shape)
    s = stoneley_slowness(frequency, (c.c11, c.c13, c.c33, c.c44, c.c66), eff["rho"],
                          mud.slowness, mud.density, rad, tool_radius, tool_modulus, full=True)
    inv_q = 2.0 * s.imag / s.real
    slow = s.real
    if seed is not None:
        slow = slow * (1.0 + noise * np.random.default_rng(seed).standard_normal(depth.size))
    meta = dict(mud_velocity=mud.velocity, mud_density=mud.density, tool_radius=tool_radius,
                tool_modulus=tool_modulus, aperture=aperture, noise=noise, seed=seed)
    return StoneleyLog(depth, slow, inv_q, frequency, meta)
