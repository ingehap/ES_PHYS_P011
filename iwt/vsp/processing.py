"""The ZO-VSP processing chain of WP3 in one call, for recorded data and the command line.

:func:`process_zovsp` runs, in order:

1. first-break peak picks and parametric up/down separation
   (:mod:`iwt.vsp.zovsp`);
2. traveltimes as phase delays of the downgoing first arrival at the seismic
   frequency, converted to vertical one-way times below sea level (straight
   rays from the source offset and depth);
3. interval velocities, blocked or regularized;
4. t* from the centroid-frequency shift of the downgoing first arrivals
   (Quan & Harris, 1997) and interval Q in blocks (:mod:`iwt.vsp.attenuation`).
   With a stratigraphic model (an elastic model from the logs, no Q), the t*
   of a synthetic VSP through it, processed identically, is subtracted, which
   leaves the intrinsic Q (plan, critical point 4); otherwise the Q is apparent;
5. optionally, deconvolution of the upgoing by the downgoing wavefield and the
   corridor stack in two-way time (the Fig. 2 trace).

The stratigraphic synthetic is driven by the recorded downgoing first arrival
at the shallowest receiver (Hann-tapered over the spectral window), injected
as a downgoing wave without source ghost: the reference spectrum of the t*
measurement is then the same in the data and in the synthetic.  On the
synthetic well this recovers the intrinsic Q of the three thick blocks to 2 %;
a Ricker wavelet at the first arrivals' peak frequency (47 Hz) biased the
middle block by 20 %, and even the true 40 Hz source Ricker by 2 %.

References
----------
Quan, Y., & Harris, J. M., 1997, Seismic attenuation tomography using the
frequency shift method: Geophysics, 62(3), 895-905. https://doi.org/10.1190/1.1444197
Hardage, B. A., 1985, Vertical Seismic Profiling, Part A: Principles (2nd ed.):
Geophysical Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..model import ElasticModel1D
from ..modeling.convolution import ricker
from . import attenuation as A
from . import zovsp as Z

__all__ = ["ZOVSPResult", "peak_frequency", "source_wavelet", "process_zovsp"]


@dataclass
class ZOVSPResult:
    """Times, velocities, attenuation and (optionally) the corridor stack of a ZO-VSP."""

    depths: np.ndarray
    picks: np.ndarray
    times: np.ndarray
    interval: Z.IntervalVelocity
    dt_star: np.ndarray
    dt_star_strat: Optional[np.ndarray]
    q: A.IntervalQ
    q_kind: str
    f_peak: float
    t: np.ndarray
    corridor: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """JSON-ready summary (the command line's output)."""
        iv, q = self.interval, self.q
        out = dict(depths=self.depths.tolist(), first_break_peaks=self.picks.tolist(),
                   vertical_time=self.times.tolist(), f_peak=self.f_peak,
                   interval_velocity=dict(top=iv.top.tolist(), bottom=iv.bottom.tolist(),
                                          velocity=iv.velocity.tolist()),
                   dt_star=self.dt_star.tolist(),
                   q=dict(kind=self.q_kind, top=q.top.tolist(), bottom=q.bottom.tolist(),
                          inv_q=q.inv_q.tolist(), inv_q_std=q.inv_q_std.tolist()))
        if self.dt_star_strat is not None:
            out["dt_star_stratigraphic"] = self.dt_star_strat.tolist()
        return out


def peak_frequency(t, down, picks, n: int = 5) -> float:
    """Peak of the mean first-arrival amplitude spectrum of the ``n`` shallowest receivers."""
    f, S = A.first_arrival_spectra(t, down[:n], picks[:n])
    s = np.mean(S, axis=0)
    s = np.convolve(s, np.ones(3) / 3.0, mode="same")
    return float(f[1 + int(np.argmax(s[1:]))])


def source_wavelet(t, down, picks, before: float = 0.05, after: float = 0.1):
    """The downgoing first arrival of the shallowest receiver as ``(t, w)``, centred on its pick
    and Hann-tapered over ``[pick - before, pick + after]``."""
    dt = float(t[1] - t[0])
    i0 = int(round((picks[0] - before - t[0]) / dt))
    i1 = int(round((picks[0] + after - t[0]) / dt))
    i0, i1 = max(i0, 0), min(i1, t.size)
    return t[i0:i1] - picks[0], down[0, i0:i1] * np.hanning(i1 - i0)


def _chain(t, data, f_seismic, band):
    pk = Z.pick_first_breaks(t, data)
    down, up = Z.separate_wavefields(t, data, pk)
    phase = Z.phase_delay_times(t, down, f_seismic, pk)
    f, S = A.first_arrival_spectra(t, down, pk)
    return pk, down, up, phase, A.centroid_shift(f, S, band)


def process_zovsp(vsp: Z.VSP, datum_velocity: Optional[float] = None, f_seismic: float = 30.0,
                  band: Tuple[float, float] = (10.0, 100.0),
                  q_blocks: Optional[Sequence[float]] = None,
                  velocity_blocks: Optional[Sequence[float]] = None,
                  strat_model: Optional[ElasticModel1D] = None,
                  corridor_peak: Optional[float] = None, corridor_length: float = 0.1,
                  workers: int = 1) -> ZOVSPResult:
    """Process a zero-offset VSP (see the module docstring).

    ``datum_velocity`` (the sea-water velocity) adds the time from sea level to
    the source depth.  ``q_blocks`` and ``velocity_blocks`` are block boundaries
    (m TVDSS) inside the receiver range.  ``strat_model``: the log-based elastic
    model (no Q) for the stratigraphic t*.  ``corridor_peak`` (Hz): make the
    corridor stack with a zero-phase Ricker of that peak frequency as the
    deconvolution target, over ``corridor_length`` s below each receiver.
    """
    t, data = vsp.t, vsp.data
    pk, down, up, phase, cs = _chain(t, data, f_seismic, band)
    tv = Z.vertical_times(phase, vsp.depths, vsp.source_depth, vsp.source_offset,
                          datum_velocity=datum_velocity)
    iv = Z.interval_velocity(vsp.depths, tv, boundaries=velocity_blocks)
    f_peak = peak_frequency(t, down, pk)
    cs_strat = None
    if strat_model is not None:
        dt = vsp.dt
        f_max = min(4.0 * f_peak, 0.8 / (2.0 * dt))
        syn = Z.synthetic_zovsp(strat_model, vsp.depths, vsp.source_depth, dt, t.size,
                                source_wavelet(t, down, pk), f_max=f_max, attenuation=False,
                                source="downgoing", workers=workers)
        cs_strat = _chain(syn.t, syn.data, f_seismic, band)[4]
    dts = cs if cs_strat is None else cs - cs_strat
    q = A.interval_q(vsp.depths, tv, dts, q_blocks)
    corridor = None
    if corridor_peak is not None:
        dec = Z.deconvolve(t, up, down, tv, ricker(corridor_peak, vsp.dt))
        corridor = Z.corridor_stack(t, dec, tv, length=corridor_length)[0]
    return ZOVSPResult(vsp.depths.copy(), pk, tv, iv, cs, cs_strat, q,
                       "apparent" if strat_model is None else "intrinsic", f_peak, t, corridor)
