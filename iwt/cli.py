"""The ``iwt`` command: the integrated well tie on files.

::

    iwt demo  OUTDIR                    synthetic well data as files (LAS/CSV logs, SEG-Y VSP and trace)
    iwt vsp   VSP.sgy  -o vsp.json      ZO-VSP: time-depth, interval velocity, Q, corridor stack
    iwt model LOGS     -o model.csv     elastic model from logs (+ VSP Q and gap velocity, anisotropy)
    iwt tie   MODEL.csv TRACE.sgy       synthetic, wavelet and tie metrics against a seismic trace

``python -m iwt`` is the same command.  Depths are m TVDSS (positive down), times
in seconds; ``iwt <command> --help`` lists the options and their defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__
from . import io as IO

__all__ = ["main", "build_parser"]


def _blocks(values):
    return None if not values else [float(v) for v in values]


def _log_kw(args) -> dict:
    kw = dict(f_sonic_p=args.f_sonic_p, f_sonic_s=args.f_sonic_s)
    if args.kb is not None:
        kw["kb"] = args.kb
    return kw


def _add_log_options(parser) -> None:
    parser.add_argument("--kb", type=float, default=None,
                        help="kelly-bushing elevation (m) for LAS measured depths")
    parser.add_argument("--f-sonic-p", type=float, default=12.0e3,
                        help="monopole sonic frequency (Hz), for the dispersion correction")
    parser.add_argument("--f-sonic-s", type=float, default=2.0e3, help="dipole frequency (Hz)")


def _q_curve(z, top, bottom, q):
    """Piecewise-constant Q at depths ``z``; the first and last blocks extend outwards."""
    top, bottom, q = (np.asarray(a, float) for a in (top, bottom, q))
    idx = np.clip(np.searchsorted(bottom, z, side="left"), 0, q.size - 1)
    return q[idx]


# -- demo -----------------------------------------------------------------------------


def cmd_demo(args) -> int:
    from . import paper
    from .modeling.convolution import normal_incidence, ricker
    from .synthetic import SEA_WATER, wisting_like_well
    from .vsp import zovsp as Z

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, logs = wisting_like_well()
    IO.write_logs_csv(out / "logs.csv", logs)
    files = ["logs.csv"]
    try:
        IO.write_las(out / "logs.las", logs, kb=paper.KB_ELEVATION, well="IWT SYNTHETIC 7324/7-2-LIKE")
        files.append("logs.las")
    except ImportError:
        print("lasio not installed: logs written as CSV only")
    if args.quick:
        depths, nt, f_max = np.arange(450.0, 1050.1, 15.0), 1100, 120.0
    else:
        depths, nt, f_max = np.arange(450.0, 1650.1, 10.0), 1600, 150.0
    dt = 0.001
    clean = Z.synthetic_zovsp(model, depths, 6.0, dt, nt, ricker(40.0, dt), f_max=f_max,
                              workers=args.workers)
    vsp = clean.with_data(Z.add_noise(clean.data, 0.003, dt, f_max, seed=3))
    IO.write_zovsp(out / "zovsp.sgy", vsp, text="SYNTHETIC ZERO-OFFSET VSP, WISTING-LIKE WELL\n"
                   "SOURCE 6 M BELOW SEA LEVEL, 40 HZ RICKER, NOISE 0.3 % OF PEAK\n"
                   "RECEIVER DEPTH = -(BYTES 41-44) M TVDSS")
    files.append("zovsp.sgy")
    dts, nts = 0.002, 650
    tr = normal_incidence(model, dts, nts, ricker(30.0, dts), attenuation=True, f_max=90.0)[1]
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(nts)
    noise *= 0.05 * np.sqrt(np.mean(tr[tr != 0] ** 2)) / noise.std()
    IO.write_segy(out / "seismic.sgy", (tr + noise)[None, :], dts, dict(cdp=[1]),
                  text="SYNTHETIC ZERO-OFFSET TRACE AT THE WELL\n30 HZ RICKER, PRIMARIES, Q, 5 % NOISE")
    files.append("seismic.sgy")
    logs_in = "logs.las" if "logs.las" in files else "logs.csv"
    steps = [
        f"iwt vsp {out / 'zovsp.sgy'} --datum-velocity {SEA_WATER.vp0:g} --q-blocks 600 740 800 "
        f"--logs {out / logs_in} --water-depth {paper.WATER_DEPTH:g} "
        f"--corridor {out / 'corridor.sgy'} -o {out / 'vsp.json'}",
        f"iwt model {out / logs_in} --water-depth {paper.WATER_DEPTH:g} --vsp {out / 'vsp.json'} "
        f"-o {out / 'model.csv'}",
        f"iwt tie {out / 'model.csv'} {out / 'corridor.sgy'} -o {out / 'tie_corridor.json'}",
        f"iwt tie {out / 'model.csv'} {out / 'seismic.sgy'} --wavelet estimate --attenuation "
        f"-o {out / 'tie_seismic.json'}",
    ]
    (out / "README.txt").write_text(
        "Synthetic Wisting-like well (iwt demo). Files: " + ", ".join(files) + "\n\nNext steps:\n"
        + "\n".join(steps) + "\n")
    print(f"wrote {', '.join(files)} to {out}\nnext:\n  " + "\n  ".join(steps))
    return 0


# -- vsp ------------------------------------------------------------------------------


def cmd_vsp(args) -> int:
    from .synthetic import SEA_WATER
    from .vsp.processing import process_zovsp
    from .workflow import model_from_logs

    vsp = IO.read_zovsp(args.vsp, depth_field=args.depth_field, depth_sign=args.depth_sign,
                        depth_shift=args.depth_shift, source_depth=args.source_depth,
                        source_offset=args.source_offset)
    strat = None
    if args.logs:
        if args.water_depth is None:
            raise SystemExit("--logs needs --water-depth")
        logs = IO.read_logs(args.logs, **_log_kw(args))
        strat = model_from_logs(logs, args.water_depth, args.water_velocity or SEA_WATER.vp0,
                                args.water_density, disperse_vs=False)
    res = process_zovsp(vsp, args.datum_velocity, args.f_seismic, tuple(args.band),
                        _blocks(args.q_blocks), _blocks(args.velocity_blocks), strat,
                        corridor_peak=args.corridor_peak if args.corridor else None,
                        workers=args.workers)
    out = res.to_dict()
    out.update(source=str(args.vsp), source_depth=vsp.source_depth,
               source_offset=vsp.source_offset, datum_velocity=args.datum_velocity,
               f_seismic=args.f_seismic, band=list(args.band))
    Path(args.output).write_text(json.dumps(out, indent=1))
    if args.td:
        iv = res.interval
        v = _q_curve(res.depths, iv.top, iv.bottom, iv.velocity)
        IO.write_time_depth(args.td, res.depths, res.times, interval_velocity_m_s=v)
    if args.corridor:
        IO.write_segy(args.corridor, res.corridor[None, :], vsp.dt, dict(cdp=[1]), t0=float(vsp.t[0]),
                      text=f"VSP CORRIDOR STACK FROM {Path(args.vsp).name}\n"
                           f"TWO-WAY TIME BELOW SEA LEVEL, {args.corridor_peak:g} HZ RICKER TARGET")
    q = res.q
    print(f"{vsp.depths.size} levels {vsp.depths[0]:.1f}-{vsp.depths[-1]:.1f} m; one-way time "
          f"{res.times[0]:.4f}-{res.times[-1]:.4f} s; first-arrival peak {res.f_peak:.0f} Hz")
    print(f"{res.q_kind} Q per block: " + "; ".join(
        f"{a:.0f}-{b:.0f} m: {('inf' if iq <= 0 else f'{1 / iq:.0f}')}"
        for a, b, iq in zip(q.top, q.bottom, q.inv_q)))
    print(f"wrote {args.output}" + (f", {args.td}" if args.td else "")
          + (f", {args.corridor}" if args.corridor else ""))
    return 0


# -- model ----------------------------------------------------------------------------


def cmd_model(args) -> int:
    from .synthetic import SEA_WATER
    from .workflow import model_from_logs

    logs = IO.read_logs(args.logs, **_log_kw(args))
    z = logs.depth
    vw = args.water_velocity or SEA_WATER.vp0
    qp, gap_vp, note = args.q, args.gap_velocity, []
    vsp = None
    if args.vsp:
        vsp = json.loads(Path(args.vsp).read_text())
        if qp is None:
            q = vsp["q"]
            inv = np.asarray(q["inv_q"], float)
            qv = np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), np.inf)
            qp = _q_curve(z, q["top"], q["bottom"], qv)
            note.append(f"{q['kind']} Q from {args.vsp}")
        dz, tz = np.asarray(vsp["depths"], float), np.asarray(vsp["vertical_time"], float)
        if gap_vp is None:
            if dz[0] <= z[0] <= dz[-1]:
                t_sb = args.water_depth / vw
                gap_vp = (z[0] - args.water_depth) / (float(np.interp(z[0], dz, tz)) - t_sb)
                note.append(f"gap velocity {gap_vp:.0f} m/s from the VSP")
            else:
                note.append("no VSP level at the top of the logs: gap velocity from the logs")
    qp = np.inf if qp is None else qp
    aniso = dict(eps=0.0, delta=0.0, gamma=0.0)
    if args.anisotropy:
        a = np.genfromtxt(args.anisotropy, delimiter=",", names=True)
        za = np.atleast_1d(a[a.dtype.names[0]]).astype(float)
        for k in ("eps", "delta", "gamma"):
            if k in a.dtype.names:
                aniso[k] = np.interp(z, za, np.atleast_1d(a[k]).astype(float))
        note.append(f"anisotropy from {args.anisotropy}")
    model = model_from_logs(logs, args.water_depth, vw, args.water_density, qp=qp,
                            qs=args.qs_ratio * np.asarray(qp, float), f_seismic=args.f_seismic,
                            disperse_vs=args.disperse_vs, gap_vp=gap_vp, **aniso)
    IO.write_model_csv(args.output, model)
    print(f"{model.n_layers} layers to {model.interfaces[-1]:.1f} m; logs {z[0]:.1f}-{z[-1]:.1f} m"
          + (f" ({logs.meta.get('dropped_samples', 0)} samples dropped)"
             if logs.meta.get("dropped_samples") else ""))
    for n in note:
        print("  " + n)
    if vsp is not None:
        dz, tz = np.asarray(vsp["depths"], float), np.asarray(vsp["vertical_time"], float)
        r = model.vertical_time(dz) - tz
        print(f"  model vertical time minus VSP: mean {1e3 * np.mean(r):+.2f} ms, RMS "
              f"{1e3 * np.sqrt(np.mean(r * r)):.2f} ms")
    print(f"wrote {args.output}")
    return 0


# -- tie ------------------------------------------------------------------------------


def cmd_tie(args) -> int:
    from scipy.signal.windows import tukey

    from .modeling.convolution import normal_incidence, ricker
    from .welltie import estimate_wavelet, tie_metrics

    model = IO.read_model_csv(args.model)
    sg = IO.read_segy(args.seismic)
    obs = sg.data[args.trace]
    t, dt = sg.t, sg.dt
    i0 = int(round(sg.t0 / dt))
    if abs(i0 * dt - sg.t0) > 1e-9:
        raise SystemExit("the trace start time is not a multiple of the sample interval")
    nt = i0 + t.size
    f_max = min(0.8 / (2.0 * dt), args.f_max)
    if args.window:
        win = (t >= args.window[0]) & (t <= args.window[1])
    else:
        k = min(2, model.n_layers - 1)
        top = 2.0 * float(model.vertical_time(model.interfaces[k])) + 0.02
        bottom = 2.0 * float(model.vertical_time(model.interfaces[-1]))
        win = (t >= top) & (t <= bottom)
    if win.sum() < 10:
        raise SystemExit("the tie window holds fewer than 10 samples")
    kw = dict(attenuation=args.attenuation, f_max=f_max)
    if args.wavelet == "ricker":
        wav = ricker(args.peak, dt)
        syn = normal_incidence(model, dt, nt, wav, **kw)[1][i0:]
        wavelet = dict(kind="ricker", peak_hz=args.peak)
    else:
        spike = (np.array([0.0, dt]), np.array([1.0, 0.0]))
        refl = normal_incidence(model, dt, nt, spike, **kw)[1][i0:]
        tw, w = estimate_wavelet(obs[win], refl[win], dt, args.half_length, args.damping)
        w = w * tukey(w.size, 0.5)                      # taper the noisy tails, keep the centre
        syn = np.convolve(refl, w)[(w.size - 1) // 2:(w.size - 1) // 2 + refl.size]
        wav = (tw, w)
        wavelet = dict(kind="estimated", t=tw.tolist(), w=w.tolist())
    m = tie_metrics(obs[win], syn[win], dt, args.max_shift)
    out = dict(model=str(args.model), seismic=str(args.seismic), trace=args.trace,
               window=[float(t[win][0]), float(t[win][-1])], attenuation=args.attenuation,
               wavelet=wavelet, metrics=dict(m._asdict(), phase_deg=float(np.degrees(m.phase))))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=1))
    print(f"window {t[win][0]:.3f}-{t[win][-1]:.3f} s, {wavelet['kind']} wavelet: correlation "
          f"{m.correlation:.3f} (best {m.max_correlation:.3f}), shift {1e3 * m.shift:+.1f} ms, "
          f"phase {np.degrees(m.phase):+.0f} deg, scale {m.scale:.3g}, PEP {m.pep:.2f}")
    if args.plot:
        _plot_tie(args.plot, t, obs, syn * m.scale, win, m, wav)
    return 0


def _plot_tie(path, t, obs, syn, win, m, wav):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return
    from .plotting import PALETTE

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 6.0), gridspec_kw=dict(width_ratios=[2, 1]))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax = axes[0]
    s = np.max(np.abs(obs[win]))
    ax.plot(obs / s, t, color=PALETTE["series"][0], lw=1.0, label="seismic")
    ax.plot(syn / s - 1.5, t, color=PALETTE["series"][1], lw=1.0, label="synthetic (shifted left)")
    ax.set_ylim(t[win][-1], t[win][0])
    ax.set_xlim(-3.0, 1.5)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("time (s)")
    ax.set_title(f"correlation {m.correlation:.2f}, shift {1e3 * m.shift:+.1f} ms, "
                 f"phase {np.degrees(m.phase):+.0f} deg", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[1]
    ax.plot(wav[0], wav[1], color=PALETTE["ink"], lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_title("wavelet", fontsize=9, loc="left")
    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(color=PALETTE["grid"], lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


# -- parser ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iwt", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"iwt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    D = argparse.ArgumentDefaultsHelpFormatter

    d = sub.add_parser("demo", formatter_class=D,
                       help="write the synthetic well's logs, ZO-VSP and seismic trace as files")
    d.add_argument("outdir")
    d.add_argument("--quick", action="store_true", help="fewer VSP levels (450-1050 m)")
    d.add_argument("--workers", type=int, default=1)
    d.set_defaults(func=cmd_demo)

    v = sub.add_parser("vsp", formatter_class=D, help="process a zero-offset VSP (SEG-Y)")
    v.add_argument("vsp", help="SEG-Y, one trace per receiver level")
    v.add_argument("-o", "--output", required=True, help="JSON result")
    v.add_argument("--td", help="also write the time-depth table (CSV)")
    v.add_argument("--corridor", help="also write the corridor stack (SEG-Y, two-way time)")
    v.add_argument("--corridor-peak", type=float, default=30.0, help="Ricker target (Hz)")
    v.add_argument("--datum-velocity", type=float, default=None,
                   help="sea-water velocity: adds the time from sea level to the source")
    v.add_argument("--source-depth", type=float, default=None,
                   help="m below sea level (default: trace header bytes 49-52)")
    v.add_argument("--source-offset", type=float, default=None,
                   help="m from the well (default: trace header bytes 37-40)")
    v.add_argument("--depth-field", default="receiver_elevation",
                   help="trace header word holding the receiver depth (see iwt.io.TRACE_HEADER)")
    v.add_argument("--depth-sign", type=float, default=None,
                   help="multiplier of the header word (default -1 for elevations, else 1)")
    v.add_argument("--depth-shift", type=float, default=0.0,
                   help="added after the sign (e.g. -KB for depths below the rig floor)")
    v.add_argument("--f-seismic", type=float, default=30.0, help="frequency of the traveltimes (Hz)")
    v.add_argument("--band", type=float, nargs=2, default=[10.0, 100.0], help="t* band (Hz)")
    v.add_argument("--q-blocks", type=float, nargs="*", help="Q block boundaries (m TVDSS)")
    v.add_argument("--velocity-blocks", type=float, nargs="*",
                   help="interval-velocity block boundaries (default: regularized, per interval)")
    v.add_argument("--logs", help="LAS or CSV logs: intrinsic Q (stratigraphic t* removed)")
    _add_log_options(v)
    v.add_argument("--water-depth", type=float, default=None)
    v.add_argument("--water-velocity", type=float, default=None)
    v.add_argument("--water-density", type=float, default=1025.0)
    v.add_argument("--workers", type=int, default=1)
    v.set_defaults(func=cmd_vsp)

    m = sub.add_parser("model", formatter_class=D, help="build the elastic model from logs")
    m.add_argument("logs", help="LAS (lasio) or CSV logs")
    m.add_argument("-o", "--output", required=True, help="model CSV")
    m.add_argument("--water-depth", type=float, required=True)
    m.add_argument("--water-velocity", type=float, default=None, help="default 1470 m/s")
    m.add_argument("--water-density", type=float, default=1025.0)
    _add_log_options(m)
    m.add_argument("--vsp", help="JSON from 'iwt vsp': Q per block and the gap velocity")
    m.add_argument("--q", type=float, default=None, help="constant Qp (overrides the VSP)")
    m.add_argument("--qs-ratio", type=float, default=0.5, help="Qs / Qp")
    m.add_argument("--gap-velocity", type=float, default=None,
                   help="Vp between the seabed and the first log (default: from the VSP)")
    m.add_argument("--f-seismic", type=float, default=30.0,
                   help="velocities are moved from the tool frequencies to this one (Hz)")
    m.add_argument("--disperse-vs", action="store_true", help="also move the dipole Vs")
    m.add_argument("--anisotropy", help="CSV: depth_tvdss_m, eps, delta[, gamma]")
    m.set_defaults(func=cmd_model)

    t = sub.add_parser("tie", formatter_class=D, help="tie a model synthetic to a seismic trace")
    t.add_argument("model", help="model CSV from 'iwt model'")
    t.add_argument("seismic", help="SEG-Y in two-way time below sea level")
    t.add_argument("--trace", type=int, default=0, help="trace index in the file")
    t.add_argument("--window", type=float, nargs=2, help="tie window (s); default: logged interval")
    t.add_argument("--wavelet", choices=("ricker", "estimate"), default="ricker")
    t.add_argument("--peak", type=float, default=30.0, help="Ricker peak frequency (Hz)")
    t.add_argument("--half-length", type=float, default=0.08, help="estimated wavelet (s)")
    t.add_argument("--damping", type=float, default=1e-2, help="relative damping of the estimate")
    t.add_argument("--max-shift", type=float, default=0.02, help="largest shift searched (s)")
    t.add_argument("--f-max", type=float, default=100.0,
                   help="highest frequency modelled (Hz); the estimated wavelet has none above")
    t.add_argument("--attenuation", action="store_true",
                   help="model Q in the synthetic (for a trace that is not Q-compensated)")
    t.add_argument("-o", "--output", help="JSON result")
    t.add_argument("--plot", help="PNG of the tie (matplotlib)")
    t.set_defaults(func=cmd_tie)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
