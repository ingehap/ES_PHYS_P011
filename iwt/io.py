"""File formats: SEG-Y traces, LAS logs, and CSV models and time-depth tables.

SEG-Y is read and written by a small numpy implementation of the fixed-length
trace layout of revisions 1 and 2 (Barry et al., 1975; SEG Technical Standards
Committee, 2002 and 2017).  It supports:

* the 3200-byte textual header (EBCDIC or ASCII), the 400-byte binary header
  and 240-byte trace headers;
* data sample formats 1 (IBM float), 2 (int32), 3 (int16), 5 (IEEE float) and
  8 (int8) for reading, 1 and 5 for writing;
* big-endian files (the standard) and little-endian files (revision 2,
  recognized by the byte-order constant or by the sample format code).

It does not support variable trace lengths, extended textual headers beyond
skipping them, more than 65535 samples per trace, or the additional trace
headers and other extensions of revisions 2.0 and 2.1 (2023).  Header words are named in :data:`TRACE_HEADER`; the
elevation and coordinate words are returned already multiplied by their scalars
(bytes 69-70 and 71-72: positive multiplies, negative divides).

segyio is not required; the test suite checks this reader against it when it is
installed.  LAS files are read with lasio (optional dependency, ``pip install
lasio``): sonic slownesses and densities in the common units are converted to SI,
and measured depth to TVDSS for a vertical well.

References
----------
Barry, K. M., Cavers, D. A., & Kneale, C. W., 1975, Recommended standards for
digital tape formats: Geophysics, 40(2), 344-352.
Norris, M. W., & Faichney, A. K. (eds.), 2002, SEG Y rev 1 Data Exchange format:
SEG Technical Standards Committee.
Hagelund, R., & Levin, S. A. (eds.), 2017, SEG-Y_r2.0: SEG-Y revision 2.0 Data
Exchange format: SEG Technical Standards Committee.
Canadian Well Logging Society, LAS 2.0: a digital standard for logs
(https://www.cwls.org).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .model import ElasticModel1D
from .synthetic import WellLogs
from .vsp.zovsp import VSP

__all__ = [
    "TRACE_HEADER",
    "Segy",
    "read_segy",
    "write_segy",
    "ibm_to_ieee",
    "ieee_to_ibm",
    "read_zovsp",
    "write_zovsp",
    "Gather",
    "read_gather",
    "write_gather",
    "read_las",
    "write_las",
    "read_logs_csv",
    "write_logs_csv",
    "read_logs",
    "write_model_csv",
    "read_model_csv",
    "write_time_depth",
    "read_time_depth",
]

# -- SEG-Y ----------------------------------------------------------------------------

#: Trace header words: name -> (first byte, 1-based; size in bytes; scalar group).
TRACE_HEADER: Dict[str, tuple] = {
    "trace_sequence_line": (1, 4, None),
    "trace_sequence_file": (5, 4, None),
    "field_record": (9, 4, None),
    "trace_number": (13, 4, None),
    "source_point": (17, 4, None),
    "cdp": (21, 4, None),
    "cdp_trace": (25, 4, None),
    "trace_id": (29, 2, None),
    "offset": (37, 4, None),
    "receiver_elevation": (41, 4, "elevation"),
    "source_surface_elevation": (45, 4, "elevation"),
    "source_depth": (49, 4, "elevation"),
    "receiver_datum_elevation": (53, 4, "elevation"),
    "source_datum_elevation": (57, 4, "elevation"),
    "source_water_depth": (61, 4, "elevation"),
    "receiver_water_depth": (65, 4, "elevation"),
    "elevation_scalar": (69, 2, None),
    "coordinate_scalar": (71, 2, None),
    "source_x": (73, 4, "coordinate"),
    "source_y": (77, 4, "coordinate"),
    "receiver_x": (81, 4, "coordinate"),
    "receiver_y": (85, 4, "coordinate"),
    "coordinate_units": (89, 2, None),
    "delay": (109, 2, None),
    "n_samples": (115, 2, None),
    "sample_interval": (117, 2, None),
    "cdp_x": (181, 4, "coordinate"),
    "cdp_y": (185, 4, "coordinate"),
}
_SCALAR = {"elevation": "elevation_scalar", "coordinate": "coordinate_scalar"}
_UNSIGNED = {"n_samples", "sample_interval"}
_FORMATS = {1: ("u4", 4), 2: ("i4", 4), 3: ("i2", 2), 5: ("f4", 4), 8: ("i1", 1)}
# binary header words (first byte, 1-based; size)
_BIN = {"interval": (3217, 2), "samples": (3221, 2), "format": (3225, 2),
        "traces_per_ensemble": (3213, 2), "sorting": (3229, 2), "measurement": (3255, 2),
        "byte_order": (3297, 4), "revision": (3501, 2), "fixed_length": (3503, 2),
        "extended_headers": (3505, 2)}


def ibm_to_ieee(words) -> np.ndarray:
    """IBM System/360 single-precision words (uint32) to float64.

    Sign bit, 7-bit base-16 exponent biased by 64 and a 24-bit fraction:
    ``(-1)^s 0.f 16^(e - 64)``; for example 0xC276A000 is -118.625.
    """
    w = np.asarray(words, dtype=np.uint32)
    sign = np.where(w >> np.uint32(31), -1.0, 1.0)
    exponent = ((w >> np.uint32(24)) & np.uint32(0x7F)).astype(np.int64) - 64
    fraction = (w & np.uint32(0x00FFFFFF)).astype(np.float64) / float(1 << 24)
    return sign * fraction * np.power(16.0, exponent)


def ieee_to_ibm(values) -> np.ndarray:
    """Floats to IBM single-precision words (uint32), rounding the fraction to nearest.

    The relative precision is 2^-21 to 2^-24 (hexadecimal normalization);
    magnitudes below 16^-65 become zero, above about 7.2e75 raise.
    """
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("IBM floats cannot hold NaN or infinity")
    out = np.zeros(x.shape, dtype=np.uint32)
    m = np.abs(x)
    nz = m > 0.0
    if not np.any(nz):
        return out
    mv = m[nz]
    e = np.floor(np.log2(mv) / 4.0).astype(np.int64) + 1          # mv = f 16^e, f in [1/16, 1)
    f = mv / np.power(16.0, e)
    hi = f >= 1.0
    e[hi] += 1
    f[hi] /= 16.0
    lo = f < 1.0 / 16.0
    e[lo] -= 1
    f[lo] *= 16.0
    frac = np.round(f * (1 << 24)).astype(np.int64)
    carry = frac >= (1 << 24)                                     # rounded up to 1.0
    frac[carry] = 1 << 20
    e[carry] += 1
    if np.any(e + 64 > 127):
        raise OverflowError("value too large for an IBM float")
    under = e + 64 < 0
    frac[under] = 0
    e[under] = -64
    sign = (x[nz] < 0).astype(np.uint32) << np.uint32(31)
    words = sign | ((e + 64).astype(np.uint32) << np.uint32(24)) | frac.astype(np.uint32)
    words[frac == 0] = 0                                          # underflow: plain zero
    out[nz] = words
    return out


@dataclass
class Segy:
    """A SEG-Y file in memory: ``data[trace, sample]`` and its headers.

    ``headers`` maps the names of :data:`TRACE_HEADER` to one value per trace
    (elevations, depths and coordinates scaled).  ``t0`` is the delay recording
    time of the first trace (s); ``binary`` holds the binary-header words read.
    """

    data: np.ndarray
    dt: float
    t0: float = 0.0
    headers: Dict[str, np.ndarray] = field(default_factory=dict)
    text: str = ""
    binary: Dict[str, int] = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.data.shape[1])


def _word(buf: bytes, first: int, size: int, order: str) -> int:
    kind = {2: "i2", 4: "i4"}[size]
    return int(np.frombuffer(buf[first - 1:first - 1 + size], dtype=order + kind)[0])


def _decode_text(raw: bytes) -> str:
    ascii_like = sum(32 <= b < 127 for b in raw[:400]) > 350 and raw[:1] == b"C"
    text = raw.decode("ascii", errors="replace") if ascii_like else raw.decode("cp037")
    return "\n".join(text[i:i + 80].rstrip() for i in range(0, len(text), 80)).rstrip()


def _byte_order(buf: bytes) -> str:
    const = buf[3296:3300]
    if const == b"\x01\x02\x03\x04":
        return ">"
    if const == b"\x04\x03\x02\x01":
        return "<"
    if _word(buf, 3225, 2, ">") in _FORMATS:
        return ">"
    if _word(buf, 3225, 2, "<") in _FORMATS:
        return "<"
    raise ValueError("unsupported or unreadable data sample format code")


def read_segy(path) -> Segy:
    """Read a SEG-Y file with fixed-length traces (see the module docstring)."""
    buf = Path(path).read_bytes()
    if len(buf) < 3600:
        raise ValueError(f"{path}: too short for a SEG-Y file")
    order = _byte_order(buf)
    binary = {k: _word(buf, a, n, order) for k, (a, n) in _BIN.items()}
    fmt = binary["format"]
    kind, size = _FORMATS[fmt]
    ns = binary["samples"] & 0xFFFF
    start = 3600 + 3200 * max(binary["extended_headers"], 0)
    trace_bytes = 240 + ns * size
    n_traces, rest = divmod(len(buf) - start, trace_bytes)
    if rest or ns == 0:
        raise ValueError(f"{path}: variable trace length or a corrupt file is not supported")
    raw = np.frombuffer(buf, dtype=np.uint8, offset=start).reshape(n_traces, trace_bytes)
    headers = {}
    for name, (first, nbytes, _) in TRACE_HEADER.items():
        kind_h = ("u" if name in _UNSIGNED else "i") + str(nbytes)
        cols = np.ascontiguousarray(raw[:, first - 1:first - 1 + nbytes])
        headers[name] = cols.view(order + kind_h).ravel().astype(np.int64)
    for name, (_, _, group) in TRACE_HEADER.items():
        if group is not None:
            s = headers[_SCALAR[group]].astype(float)
            factor = np.ones_like(s)
            factor[s > 0] = s[s > 0]                              # positive: multiply
            factor[s < 0] = 1.0 / -s[s < 0]                       # negative: divide
            headers[name] = headers[name] * factor
    samples = np.ascontiguousarray(raw[:, 240:]).view(order + kind).reshape(n_traces, ns)
    data = ibm_to_ieee(samples) if fmt == 1 else samples.astype(np.float64)
    dt_us = binary["interval"] & 0xFFFF or int(headers["sample_interval"][0])
    t0 = float(headers["delay"][0]) * 1e-3 if n_traces else 0.0
    return Segy(data, dt_us * 1e-6, t0, headers, _decode_text(buf[:3200]), binary)


def _scalar_for(values) -> int:
    """The SEG-Y scalar that keeps centimetres where the values allow it."""
    big = float(np.max(np.abs(values))) if np.size(values) else 0.0
    for s in (-100, -10, 1, 10, 100, 1000):
        if big * (-s if s < 0 else 1.0 / s) < 2 ** 31 - 1:
            return s
    raise OverflowError("header value too large for SEG-Y")


def write_segy(path, data, dt: float, headers: Optional[Dict[str, Sequence]] = None,
               text: str = "", t0: float = 0.0, fmt: int = 5, byteorder: str = ">",
               measurement: int = 1) -> None:
    """Write ``data[trace, sample]`` as SEG-Y (revision 1 big-endian, or revision 2 little-endian).

    ``headers`` maps names of :data:`TRACE_HEADER` to per-trace values (scaled
    words in their units, e.g. metres; the scalars are chosen here).  ``dt`` and
    ``t0`` are in seconds; ``t0`` is rounded to whole milliseconds (the delay
    word).  ``fmt`` is 5 (IEEE float) or 1 (IBM float); ``measurement`` 1 for
    metres, 2 for feet.  The trace sequence numbers default to 1, 2, ...
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n, ns = data.shape
    if fmt not in (1, 5):
        raise ValueError("write_segy writes IBM (1) or IEEE (5) floats")
    if byteorder not in (">", "<"):
        raise ValueError("byteorder must be '>' or '<'")
    dt_us = int(round(dt * 1e6))
    if not (0 < dt_us < 65536 and 0 < ns < 65536):
        raise ValueError("sample interval or trace length outside the SEG-Y range")
    o = byteorder
    given = (text or "").splitlines()[:39]
    lines = [f"C{i + 1:2d} {ln}" for i, ln in enumerate(given)]
    lines += [f"C{i + 1:2d}" for i in range(len(lines), 39)] + ["C40 END TEXTUAL HEADER"]
    out = bytearray("".join(ln[:80].ljust(80) for ln in lines).encode("cp037"))
    binh = bytearray(400)

    def put(first, size, value):
        binh[first - 3201:first - 3201 + size] = np.array(value, dtype=o + f"i{size}").tobytes()

    put(3213, 2, min(n, 32767))
    put(3217, 2, dt_us if dt_us < 32768 else dt_us - 65536)
    put(3221, 2, ns if ns < 32768 else ns - 65536)
    put(3225, 2, fmt)
    put(3255, 2, measurement)
    if o == "<":
        binh[3297 - 3201:3301 - 3201] = np.array(0x01020304, dtype="<u4").tobytes()
        binh[3501 - 3201:3503 - 3201] = bytes([2, 0])          # revision 2.0: major, minor
    else:
        binh[3501 - 3201:3503 - 3201] = bytes([1, 0])          # revision 1.0
    put(3503, 2, 1)                                            # fixed-length traces
    out += binh
    th = np.zeros((n, 240), dtype=np.uint8)
    hv = {k: np.broadcast_to(np.asarray(v, dtype=float), (n,)) for k, v in (headers or {}).items()}
    unknown = set(hv) - set(TRACE_HEADER)
    if unknown:
        raise KeyError(f"unknown trace header words: {sorted(unknown)}")
    hv.setdefault("trace_sequence_line", np.arange(1, n + 1, dtype=float))
    hv.setdefault("trace_sequence_file", np.arange(1, n + 1, dtype=float))
    hv.setdefault("trace_id", np.ones(n))
    hv["delay"] = np.full(n, round(t0 * 1e3))
    hv["n_samples"] = np.full(n, ns, dtype=float)
    hv["sample_interval"] = np.full(n, dt_us, dtype=float)
    for group, sc in _SCALAR.items():
        names = [k for k, (_, _, g) in TRACE_HEADER.items() if g == group and k in hv]
        if names:
            s = _scalar_for(np.concatenate([hv[k] for k in names]))
            hv[sc] = np.full(n, s, dtype=float)
            for k in names:
                hv[k] = hv[k] * (-s if s < 0 else 1.0 / s)
    for name, v in hv.items():
        first, size, _ = TRACE_HEADER[name]
        kind = ("u" if name in _UNSIGNED else "i") + str(size)
        th[:, first - 1:first - 1 + size] = np.round(v).astype(o + kind).view(np.uint8).reshape(n, size)
    samples = ieee_to_ibm(data).astype(o + "u4") if fmt == 1 else data.astype(o + "f4")
    body = np.concatenate([th, samples.view(np.uint8).reshape(n, -1)], axis=1)
    Path(path).write_bytes(bytes(out) + body.tobytes())


# -- VSPs and gathers -----------------------------------------------------------------


def read_zovsp(path, depth_field: str = "receiver_elevation", depth_sign: Optional[float] = None,
               depth_shift: float = 0.0, source_depth: Optional[float] = None,
               source_offset: Optional[float] = None) -> VSP:
    """A zero-offset VSP from SEG-Y: one trace per receiver level.

    Receiver depths (m TVDSS) are ``depth_sign * header[depth_field] +
    depth_shift``.  The default reads the receiver group elevation (bytes 41-44)
    as a negative elevation below sea level (sign -1); for a depth word use
    ``depth_sign=1``; ``depth_shift`` converts another datum (for example
    ``-KB`` for depths below the rig floor in a vertical well).  The source depth
    defaults to bytes 49-52 and the source offset to the median of bytes 37-40.
    Traces are sorted by depth.
    """
    sg = read_segy(path)
    if depth_sign is None:
        depth_sign = -1.0 if "elevation" in depth_field else 1.0
    depths = depth_sign * sg.headers[depth_field] + depth_shift
    order = np.argsort(depths, kind="stable")
    zs = float(np.median(sg.headers["source_depth"])) if source_depth is None else source_depth
    xs = float(np.median(np.abs(sg.headers["offset"]))) if source_offset is None else source_offset
    return VSP(sg.t, depths[order], sg.data[order], zs, xs)


def write_zovsp(path, vsp: VSP, text: str = "", fmt: int = 5) -> None:
    """Write a :class:`~iwt.vsp.zovsp.VSP` (receiver depth as negative elevation, bytes 41-44)."""
    n = vsp.depths.size
    write_segy(path, vsp.data, vsp.dt, dict(
        receiver_elevation=-vsp.depths, source_depth=np.full(n, vsp.source_depth),
        offset=np.full(n, vsp.source_offset), field_record=np.ones(n),
        trace_number=np.arange(1, n + 1)), text=text or "ZERO-OFFSET VSP WRITTEN BY IWT", fmt=fmt)


@dataclass
class Gather:
    """A CMP gather: ``data[offset, time]`` with source and receiver depths (m below sea level)."""

    t: np.ndarray
    offsets: np.ndarray
    data: np.ndarray
    source_depth: float = 0.0
    receiver_depth: float = 0.0

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def read_gather(path) -> Gather:
    """A CMP gather from SEG-Y: offsets from bytes 37-40, sorted; receiver depth from -elevation."""
    sg = read_segy(path)
    off = np.abs(sg.headers["offset"]).astype(float)
    order = np.argsort(off, kind="stable")
    return Gather(sg.t, off[order], sg.data[order], float(np.median(sg.headers["source_depth"])),
                  float(np.median(-sg.headers["receiver_elevation"])))


def write_gather(path, gather: Gather, text: str = "", fmt: int = 5) -> None:
    n = gather.offsets.size
    write_segy(path, gather.data, gather.dt, dict(
        offset=gather.offsets, cdp=np.ones(n), cdp_trace=np.arange(1, n + 1),
        source_depth=np.full(n, gather.source_depth),
        receiver_elevation=np.full(n, -gather.receiver_depth)),
        text=text or "CMP GATHER WRITTEN BY IWT", t0=float(gather.t[0]), fmt=fmt)


# -- logs -----------------------------------------------------------------------------

#: Curve mnemonics tried in order (case-insensitive).
P_SONIC = ("DTCO", "DTC", "DT", "DTCOMP", "DT4P", "AC")
S_SONIC = ("DTSM", "DTS", "DTSH", "DTSHEAR", "DT4S")
DENSITY = ("RHOB", "RHOZ", "DEN", "ZDEN", "DENS")
TVDSS = ("TVDSS", "TVDS", "TVD_SS")
KB_ITEMS = ("EKB", "KB", "EREF", "EDF", "APD")

_SLOWNESS = {"US/F": 0.3048e6, "US/FT": 0.3048e6, "USEC/FT": 0.3048e6, "USEC/F": 0.3048e6,
             "US/M": 1.0e6, "USEC/M": 1.0e6}
_DENSITY = {"G/CC": 1000.0, "G/C3": 1000.0, "G/CM3": 1000.0, "GM/CC": 1000.0,
            "K/M3": 1.0, "KG/M3": 1.0}
_LENGTH = {"M": 1.0, "METER": 1.0, "METERS": 1.0, "METRES": 1.0, "F": 0.3048, "FT": 0.3048,
           "FEET": 0.3048}


def _unit(table, unit, what):
    key = (unit or "").strip().upper().replace(" ", "")
    if key not in table:
        raise ValueError(f"unknown {what} unit {unit!r}; expected one of {sorted(table)}")
    return table[key]


def _pick(names, available, explicit=None):
    if explicit is not None:
        if explicit.upper() not in available:
            raise KeyError(f"curve {explicit} not in the file")
        return explicit.upper()
    for n in names:
        if n in available:
            return n
    return None


def _finish_logs(depth, vp, vs, rho, meta, step=None) -> WellLogs:
    ok = np.isfinite(depth) & np.isfinite(vp) & np.isfinite(vs) & np.isfinite(rho)
    if ok.sum() < 2:
        raise ValueError("fewer than two depths with Vp, Vs and density")
    depth, vp, vs, rho = depth[ok], vp[ok], vs[ok], rho[ok]
    order = np.argsort(depth, kind="stable")
    depth, vp, vs, rho = depth[order], vp[order], vs[order], rho[order]
    keep = np.concatenate([[True], np.diff(depth) > 0])
    depth, vp, vs, rho = depth[keep], vp[keep], vs[keep], rho[keep]
    meta["dropped_samples"] = int((~ok).sum())
    if step is not None:
        z = np.arange(depth[0], depth[-1] + 0.5 * step, step)
        vp = 1.0 / np.interp(z, depth, 1.0 / vp)                # slowness interpolated
        vs = 1.0 / np.interp(z, depth, 1.0 / vs)
        rho = np.interp(z, depth, rho)
        depth = z
    return WellLogs(depth, vp, vs, rho, meta)


def read_las(path, kb: Optional[float] = None, p_curve: Optional[str] = None,
             s_curve: Optional[str] = None, rho_curve: Optional[str] = None,
             tvdss_curve: Optional[str] = None, f_sonic_p: float = 12.0e3,
             f_sonic_s: float = 2.0e3, step: Optional[float] = None) -> WellLogs:
    """Wireline logs from a LAS file (lasio), in SI units and m TVDSS.

    The P and S slownesses and the density are found by the mnemonics in
    :data:`P_SONIC`, :data:`S_SONIC` and :data:`DENSITY` (or given), and
    converted from us/ft, us/m, g/cm^3 or kg/m^3.  Depth is a TVDSS curve if
    present (:data:`TVDSS`); otherwise the index (MD, metres or feet) minus the
    kelly-bushing elevation ``kb`` (m above sea level; default from the EKB, KB,
    EREF, EDF or APD item), which assumes a vertical well.  Samples with any of
    the three curves missing are dropped (``meta["dropped_samples"]``);
    ``step`` (m) resamples to a regular grid (slowness interpolated).
    ``f_sonic_p`` and ``f_sonic_s`` record the tool frequencies (assumed: they
    are not in a LAS file) for the dispersion correction.
    """
    try:
        import lasio
    except ImportError as exc:                                 # pragma: no cover
        raise ImportError("reading LAS needs lasio: pip install lasio") from exc
    las = lasio.read(str(path))
    names = {c.mnemonic.upper(): c for c in las.curves}
    index = las.curves[0]
    z_unit = _unit(_LENGTH, index.unit, "depth")
    tv = _pick(TVDSS, names, tvdss_curve)
    if tv is not None:
        depth = np.asarray(names[tv].data, float) * _unit(_LENGTH, names[tv].unit, "depth")
        kb_used = None
    else:
        if kb is None:
            for sec in (las.well, las.params):
                for item in KB_ITEMS:
                    if item in sec and str(sec[item].value).strip() not in ("", "None"):
                        kb = float(sec[item].value) * _unit(_LENGTH, sec[item].unit or "M",
                                                              "elevation")
                        break
                if kb is not None:
                    break
        if kb is None:
            raise ValueError("no TVDSS curve and no kelly-bushing elevation: give kb")
        depth = np.asarray(index.data, float) * z_unit - kb
        kb_used = kb
    keys = []
    for names_, explicit, table, what in ((P_SONIC, p_curve, _SLOWNESS, "slowness"),
                                          (S_SONIC, s_curve, _SLOWNESS, "slowness"),
                                          (DENSITY, rho_curve, _DENSITY, "density")):
        k = _pick(names_, names, explicit)
        if k is None:
            raise KeyError(f"no {what} curve among {names_}")
        keys.append(k)
    p, s, r = (names[k] for k in keys)
    vp = _unit(_SLOWNESS, p.unit, "slowness") / np.asarray(p.data, float)
    vs = _unit(_SLOWNESS, s.unit, "slowness") / np.asarray(s.data, float)
    rho = np.asarray(r.data, float) * _unit(_DENSITY, r.unit, "density")
    well = str(las.well["WELL"].value) if "WELL" in las.well else ""
    meta = dict(f_sonic_p=f_sonic_p, f_sonic_s=f_sonic_s, source=str(path), well=well,
                kb=kb_used, curves=dict(zip(("vp", "vs", "rho"), keys)))
    return _finish_logs(depth, vp, vs, rho, meta, step)


def write_las(path, logs: WellLogs, kb: float = 0.0, well: str = "IWT SYNTHETIC") -> None:
    """Write logs as LAS 2.0 (lasio): DEPT (m MD = TVDSS + kb), TVDSS, DTCO, DTSM, RHOB."""
    import lasio

    las = lasio.LASFile()
    las.well["WELL"].value = well
    las.well["NULL"].value = -999.25
    las.params.append(lasio.HeaderItem("EKB", unit="M", value=float(kb),
                                       descr="kelly bushing elevation above sea level"))
    c = logs.las_curves(kb)
    for name, unit, descr in (("DEPT", "M", "measured depth"), ("TVDSS", "M", "true vertical depth subsea"),
                              ("DTCO", "US/F", "compressional slowness"),
                              ("DTSM", "US/F", "shear slowness"), ("RHOB", "G/C3", "bulk density")):
        las.append_curve(name, c[name], unit=unit, descr=descr)
    with open(path, "w") as fh:
        las.write(fh, version=2.0)


_LOG_COLUMNS = ("depth_tvdss_m", "vp_m_s", "vs_m_s", "rho_kg_m3")


def write_logs_csv(path, logs: WellLogs) -> None:
    """Logs as CSV in SI units (``depth_tvdss_m, vp_m_s, vs_m_s, rho_kg_m3``)."""
    np.savetxt(path, np.column_stack([logs.depth, logs.vp, logs.vs, logs.rho]), delimiter=",",
               header=",".join(_LOG_COLUMNS), comments="", fmt="%.6f")


def read_logs_csv(path, f_sonic_p: float = 12.0e3, f_sonic_s: float = 2.0e3,
                  step: Optional[float] = None) -> WellLogs:
    """Logs from a CSV with the columns of :func:`write_logs_csv` (SI units, m TVDSS)."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    missing = [c for c in _LOG_COLUMNS if c not in a.dtype.names]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    meta = dict(f_sonic_p=f_sonic_p, f_sonic_s=f_sonic_s, source=str(path))
    return _finish_logs(*(np.atleast_1d(a[c]).astype(float) for c in _LOG_COLUMNS), meta, step)


def read_logs(path, **kw) -> WellLogs:
    """LAS (``.las``) or CSV logs, by extension."""
    if str(path).lower().endswith(".las"):
        return read_las(path, **kw)
    kw = {k: v for k, v in kw.items() if k in ("f_sonic_p", "f_sonic_s", "step")}
    return read_logs_csv(path, **kw)


# -- models and time-depth tables -----------------------------------------------------

_MODEL_COLUMNS = ("top", "bottom", "vp0", "vs0", "rho", "eps", "delta", "gamma", "qp", "qs")


def write_model_csv(path, model: ElasticModel1D) -> None:
    """One row per layer: top, bottom (m TVDSS), vp0, vs0 (m/s at ``f_ref``), rho, Thomsen
    parameters, Q and label; ``f_ref`` in a comment line."""
    with open(path, "w", newline="") as fh:
        fh.write(f"# iwt ElasticModel1D, velocities at f_ref = {model.f_ref:g} Hz\n")
        w = csv.writer(fh)
        w.writerow(_MODEL_COLUMNS + ("label",))
        for i in range(model.n_layers):
            row = [model.interfaces[i], model.interfaces[i + 1]]
            row += [getattr(model, k)[i] for k in _MODEL_COLUMNS[2:]]
            w.writerow([f"{v:.10g}" for v in row] + [model.label[i]])


def read_model_csv(path) -> ElasticModel1D:
    """The model written by :func:`write_model_csv` (contiguous layers)."""
    f_ref = 30.0
    with open(path, newline="") as fh:
        lines = fh.readlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    for ln in lines:
        if ln.startswith("#") and "f_ref =" in ln:
            f_ref = float(ln.split("f_ref =")[1].split()[0])
    rows = list(csv.DictReader(body))
    if not rows:
        raise ValueError(f"{path}: no layers")
    col = {k: np.array([float(r[k]) for r in rows]) for k in _MODEL_COLUMNS}
    if not np.allclose(col["top"][1:], col["bottom"][:-1]):
        raise ValueError(f"{path}: layers are not contiguous")
    interfaces = np.concatenate([col["top"], col["bottom"][-1:]])
    return ElasticModel1D(interfaces, **{k: col[k] for k in _MODEL_COLUMNS[2:]},
                          label=np.array([r.get("label", "") for r in rows]), f_ref=f_ref)


def write_time_depth(path, depth, time, **columns) -> None:
    """A time-depth table: depth (m TVDSS), one-way vertical time (s) and extra columns."""
    names = ["depth_tvdss_m", "time_s"] + list(columns)
    cols = [np.asarray(depth, float), np.asarray(time, float)]
    cols += [np.asarray(v, float) for v in columns.values()]
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(names), comments="",
               fmt="%.9g")


def read_time_depth(path):
    """``(depth, time)`` from a CSV with columns ``depth_tvdss_m`` and ``time_s`` (or the first two)."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    names = a.dtype.names
    z = a["depth_tvdss_m"] if "depth_tvdss_m" in names else a[names[0]]
    t = a["time_s"] if "time_s" in names else a[names[1]]
    return np.atleast_1d(z).astype(float), np.atleast_1d(t).astype(float)
