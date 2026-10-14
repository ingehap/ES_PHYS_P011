"""Interbed-multiple and mode-conversion contamination of reservoir reflections.

The article found the shallow Wisting reservoir reflections "contaminated by
interbed multiples originating in the overburden" by full-waveform modelling.
Here the contamination is measured by switching physics on and off in the
reflectivity method (:mod:`iwt.modeling.reflectivity`) and comparing the gathers
after identical processing (mute, migration, AVA picking):

* ``full``: internal multiples and P-S mode conversion;
* ``primaries``: single reflections, with every mode conversion on the way
  (P down, S up and back to P at the seabed, for example) and the transmission
  and conversion losses;
* ``no mode conversion``: multiples of P waves only (exact PP coefficients, the
  converted energy dropped);
* ``primaries with transmission loss``: P-wave primaries only;
* ``Born primaries``: single scattering, unit transmission.

The contamination is "full minus Born primaries".  Differences between the
intermediate variants do not isolate mechanisms in finely layered media.
Short-period multiples restore much of the amplitude that transmission losses
remove (O'Doherty & Anstey, 1971), so "full minus primaries" mostly measures
that restoration; with mode conversion on, the converted primaries and the
multiples are each larger than their sum and largely cancel.  Dropping mode
conversion is not a physical medium: at oblique incidence the P-only stack loses
the converted energy at every thin layer without getting any of it back.  It
still bounds the part of the contamination that P-wave interbed multiples alone
could cause.

References
----------
O'Doherty, R. F., & Anstey, N. A., 1971, Reflections on amplitudes: Geophysical
Prospecting, 19(3), 430-458. https://doi.org/10.1111/j.1365-2478.1971.tb00610.x
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .model import ElasticModel1D
from .modeling.reflectivity import Reflectivity

__all__ = ["PHYSICS", "gathers", "relative_rms"]

PHYSICS: Dict[str, dict] = {
    "full": {},
    "primaries": dict(internal_multiples=False),
    "no mode conversion": dict(converted_waves=False),
    "primaries with transmission loss": dict(internal_multiples=False, converted_waves=False),
    "Born primaries": dict(internal_multiples=False, converted_waves=False,
                           transmission_loss=False),
}


def gathers(model: ElasticModel1D, offsets, z_source: float, z_receiver: float, dt: float,
            nt: int, wavelet, physics: Optional[Dict[str, dict]] = None,
            attenuation: bool = True, f_max: Optional[float] = None,
            workers: int = 1) -> Dict[str, np.ndarray]:
    """Hydrophone CMP gathers ``data[offset, time]`` for each physics variant (no free surface)."""
    out = {}
    for name, kw in (PHYSICS if physics is None else physics).items():
        r = Reflectivity(model, attenuation=attenuation, free_surface=False, **kw)
        out[name] = r.record(offsets, z_source, [z_receiver], dt, nt, wavelet, "pressure",
                             f_max=f_max, workers=workers).data[0]
    return out


def relative_rms(a, b, mask=None) -> float:
    """``rms(a - b) / rms(b)`` over ``mask`` (NaN ignored)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.sqrt(np.nanmean((a - b) ** 2) / np.nanmean(b * b)))
