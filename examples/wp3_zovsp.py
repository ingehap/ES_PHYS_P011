"""Work package 3: ZO-VSP processing, Q and the sonic calibration.

The "measured" ZO-VSP is a vertical-incidence (p = 0) reflectivity synthetic of
the Wisting-like well: free surface, source ghost, all internal multiples, Q,
and band-limited noise (0.3 % of each trace's peak, about 50 dB).  It is
processed as field data would be:

1. picks (onset, peak, and phase delay at 30 Hz), time-depth and interval
   velocity (plain differencing against regularized least squares);
2. wavefield separation (parametric against median), deconvolution and the
   corridor stack, compared with the true reflectivity;
3. Q by spectral ratio and centroid shift: apparent, stratigraphic (from an
   elastic synthetic of the 0.5 m *log* model) and intrinsic, against the model;
4. the drift curve for each pick convention, the classic knee calibration, and
   the physical calibration (intrinsic Q dispersion + Backus average).

Writes ``docs/figures/wp3_*.png`` and prints the numbers quoted in the README.

    python examples/wp3_zovsp.py [--workers 4]
"""

import argparse
from pathlib import Path

import numpy as np

from iwt import calibration as C
from iwt import paper
from iwt.model import ElasticModel1D
from iwt.modeling.convolution import normal_incidence, ricker
from iwt.plotting import unit_tops
from iwt.synthetic import SEA_WATER, synthetic_logs, wisting_like_model
from iwt.vsp import attenuation as A
from iwt.vsp import zovsp as Z

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
DT, NT = 0.001, 1600
SOURCE_DEPTH = 6.0
DEPTHS = np.arange(450.0, 1650.1, 10.0)
WAVELET = ricker(40.0, DT)
F_MAX = 150.0
NOISE = 0.003                    # noise standard deviation / trace peak (50 dB)
BAND = (10.0, 100.0)             # Hz, for t*
Q_BLOCKS = [600.0, 740.0, 800.0]  # Hekkingen, Fruholmen and Snadd tops
F_SEIS, F_SONIC = 30.0, 12.0e3
BACKUS = 40.0                    # m, about half the 30 Hz wavelength


def log_model(logs, model):
    """Elastic 0.5 m model from the logs (sonic, dipole, density), as built in practice.

    Sea from the article's water depth; between the seabed and the top of the
    logs (470 m) one layer with the mean of the first 10 m of logs.  Isotropic,
    no Q: it only has to predict the stratigraphic filtering at vertical
    incidence.
    """
    d = logs.depth
    s_p = 0.5 * (1 / logs.vp[1:] + 1 / logs.vp[:-1])
    s_s = 0.5 * (1 / logs.vs[1:] + 1 / logs.vs[:-1])
    rho = 0.5 * (logs.rho[1:] + logs.rho[:-1])
    top = d < d[0] + 10.0
    over = (np.mean(logs.vp[top]), np.mean(logs.vs[top]), np.mean(logs.rho[top]))
    return ElasticModel1D(
        np.concatenate([[0.0, paper.WATER_DEPTH], d]),
        vp0=np.concatenate([[SEA_WATER.vp0, over[0]], 1 / s_p]),
        vs0=np.concatenate([[0.0, over[1]], 1 / s_s]),
        rho=np.concatenate([[SEA_WATER.rho, over[2]], rho]),
        f_ref=model.f_ref)


def true_inv_q(model, edges):
    """Traveltime-weighted 1/Q of the model in each block."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = model.split_at([a, b])
        i = (s.mid > a) & (s.mid < b)
        w = s.thickness[i] / s.vp0[i]
        out.append(np.sum(w / s.qp[i]) / np.sum(w))
    return np.array(out)


def process(t, data):
    """Picks, separation, 30 Hz phase times and t* (spectral ratio, centroid)."""
    pk = Z.pick_first_breaks(t, data)
    down, up = Z.separate_wavefields(t, data, pk)
    phase = Z.phase_delay_times(t, down, F_SEIS, pk)
    f, S = A.first_arrival_spectra(t, down, pk)
    sr = A.spectral_ratio(f, S, BAND)[0]
    cs = A.centroid_shift(f, S, BAND)
    return dict(pk=pk, down=down, up=up, phase=phase, sr=sr, cs=cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    model = wisting_like_model()
    logs = synthetic_logs(model)
    lm = log_model(logs, model)
    kw = dict(f_max=F_MAX, workers=args.workers)

    clean = Z.synthetic_zovsp(model, DEPTHS, SOURCE_DEPTH, DT, NT, WAVELET, **kw)
    exact_down = Z.synthetic_zovsp(model, DEPTHS, SOURCE_DEPTH, DT, NT, WAVELET,
                                   component="uz_down", **kw).data
    vsp = clean.with_data(Z.add_noise(clean.data, NOISE, DT, F_MAX, seed=3))
    strat_log = Z.synthetic_zovsp(lm, DEPTHS, SOURCE_DEPTH, DT, NT, WAVELET,
                                  attenuation=False, **kw)
    strat_true = Z.synthetic_zovsp(model, DEPTHS, SOURCE_DEPTH, DT, NT, WAVELET,
                                   attenuation=False, **kw)
    t = vsp.t
    r = process(t, vsp.data)
    tops = [(z, n) for z, n in unit_tops(model)[1:] if z > 440.0]
    t30 = model.vertical_time(DEPTHS)

    # 1. picks and time-depth ------------------------------------------------------
    onset = Z.pick_first_breaks(t, vsp.data, "onset")
    rel = {"onset": onset, "peak": r["pk"], "phase 30 Hz (downgoing)": r["phase"]}
    print("1. Pick conventions: time at 1650 m minus time at 450 m, minus the same for the"
          " model's 30 Hz vertical time (ms)")
    for k, v in rel.items():
        print(f"  {k:26s} {1e3 * ((v[-1] - v[0]) - (t30[-1] - t30[0])):+6.2f}")
    tv = Z.vertical_times(r["phase"], DEPTHS, SOURCE_DEPTH, datum_velocity=SEA_WATER.vp0)
    raw = Z.interval_velocity(DEPTHS, tv, smoothing=0.0)
    reg = Z.interval_velocity(DEPTHS, tv)
    # Snadd has a velocity gradient: one constant-velocity block for it would bias
    # its thin neighbours, so it is split every 100 m
    b_units = [z for z, _ in tops] + list(np.arange(900.0, DEPTHS[-1], 100.0))
    blk = Z.interval_velocity(DEPTHS, tv, boundaries=b_units)
    v_true = 10.0 / np.diff(t30)                     # 30 Hz model, 10 m intervals
    print(f"  noise of the phase times (from second differences): {1e3 * reg.sigma:.2f} ms")
    print("  unit interval velocities (m/s), blocked least squares | model 30 Hz time average:")
    for a_, b_, v_ in zip(blk.top, blk.bottom, blk.velocity):
        vm = (b_ - a_) / float(model.vertical_time(b_) - model.vertical_time(a_))
        if a_ < 800.0 or b_ >= DEPTHS[-1]:
            print(f"    {a_:6.1f}-{b_:6.1f} m: {v_:6.0f} | {vm:6.0f}  ({100 * (v_ / vm - 1):+.1f} %)")
    for name, iv in (("plain differencing", raw), ("regularized", reg)):
        e = iv.velocity - v_true
        print(f"  interval velocity, {name:19s}: RMS error {np.sqrt(np.mean(e ** 2)):5.0f} m/s,"
              f" mean {np.mean(e):+5.0f} m/s")

    # 2. separation, deconvolution, corridor stack ----------------------------------
    upx = clean.data - exact_down
    late = t[None, :] > r["pk"][:, None] + 0.03
    print("\n2. Wavefield separation: RMS(upgoing error) / RMS(exact upgoing) after the first arrival")
    for method, length in (("parametric", 5), ("median", 9)):
        _, u = Z.separate_wavefields(t, clean.data, r["pk"], length, method)
        e = np.sqrt(np.sum(((u - upx) ** 2 * late)[5:-5]) / np.sum((upx ** 2 * late)[5:-5]))
        print(f"  {method:10s} ({length} traces, noise-free data): {e:.2f}")
    target = ricker(30.0, DT)
    dec = Z.deconvolve(t, r["up"], r["down"], tv, target)
    corridor, _ = Z.corridor_stack(t, dec, tv, length=0.1)
    full_stack, _ = Z.corridor_stack(t, dec, tv, length=None)
    _, born = normal_incidence(model, DT, NT, target)
    _, prim_tl = normal_incidence(model, DT, NT, target, transmission_loss=True)
    w = (t > 2 * tv.min() + 0.02) & (t < 2 * tv.max())
    print("  correlation with the Born primaries (TWT window of the receivers):"
          f" corridor {np.corrcoef(corridor[w], born[w])[0, 1]:.3f},"
          f" full stack {np.corrcoef(full_stack[w], born[w])[0, 1]:.3f}")
    a = np.dot(corridor[w], born[w]) / np.dot(born[w], born[w])
    print(f"  corridor amplitude / Born amplitude (least squares): {a:.2f}")

    # 3. Q -------------------------------------------------------------------------
    edges = np.concatenate([[DEPTHS[0]], Q_BLOCKS, [DEPTHS[-1]]])
    tq = true_inv_q(model, edges)
    rs_log, rs_true = process(t, strat_log.data), process(t, strat_true.data)
    print("\n3. Interval 1/Q x 1000 per block "
          + " | ".join(f"{a:.0f}-{b:.0f} m" for a, b in zip(edges[:-1], edges[1:])))
    print(f"  {'model (true intrinsic)':44s}" + "".join(f"{1e3 * x:8.2f}" for x in tq))
    iq = {}
    for est in ("sr", "cs"):
        lab = "spectral ratio" if est == "sr" else "centroid shift"
        for name, ts in (("apparent", r[est]),
                         ("stratigraphic, log model", rs_log[est]),
                         ("intrinsic = apparent - strat. (log model)", r[est] - rs_log[est]),
                         ("intrinsic, strat. from the true model", r[est] - rs_true[est])):
            q = A.interval_q(DEPTHS, tv, ts, Q_BLOCKS)
            iq[(est, name)] = q
            print(f"  {lab + ', ' + name:44s}" + "".join(f"{1e3 * x:8.2f}" for x in q.inv_q)
                  + "   (+-" + " ".join(f"{1e3 * x:.2f}" for x in q.inv_q_std) + ")")

    print("  noise sensitivity, intrinsic 1/Q x 1000 (strat. from the log model):")
    for level in (0.0, NOISE, 0.01):
        data = Z.add_noise(clean.data, level, DT, F_MAX, seed=3) if level else clean.data
        rn = process(t, data)
        tvn = Z.vertical_times(rn["phase"], DEPTHS, SOURCE_DEPTH)
        for est in ("sr", "cs"):
            q = A.interval_q(DEPTHS, tvn, rn[est] - rs_log[est], Q_BLOCKS)
            print(f"    noise {100 * level:3.1f} % of the trace peak, {est}:"
                  + "".join(f"{1e3 * x:8.2f}" for x in q.inv_q)
                  + "   (+-" + " ".join(f"{1e3 * x:.2f}" for x in q.inv_q_std) + ")")

    # 4. drift and calibration ------------------------------------------------------
    z = logs.depth
    print("\n4. Drift (VSP minus integrated 12 kHz sonic, tied at 470 m), ms at 600 / 800 / 1200 / 1650 m")
    zq = [600.0, 800.0, 1200.0, 1650.0]
    drifts = {}
    for name, times in rel.items():
        tvi = Z.vertical_times(times, DEPTHS, SOURCE_DEPTH)
        drifts[name] = C.drift(z, logs.vp, DEPTHS, tvi)
        print(f"  {name:26s}" + "".join(f"{1e3 * x:7.2f}" for x in np.interp(zq, DEPTHS, drifts[name])))
    nl = synthetic_logs(model, seed=None)
    nl30 = synthetic_logs(model, seed=None, f_sonic_p=F_SEIS)
    disp = C.drift(z, nl.vp, DEPTHS, C.sonic_time(z, nl30.vp, DEPTHS) + 1.0)
    print(f"  {'(intrinsic dispersion only)':26s}" + "".join(f"{1e3 * x:7.2f}" for x in np.interp(zq, DEPTHS, disp)))
    rho = logs.rho

    def physical(q, backus=True):
        return C.physical_calibration(z, logs.vp, rho, q.at(z, q.q[0]), F_SONIC, F_SEIS,
                                      backus_length=BACKUS if backus else None)

    intr, appa = "intrinsic = apparent - strat. (log model)", "apparent"
    cal = {"knee (knees at 600, 740, 800 m)":
           C.knee_calibration(z, logs.vp, DEPTHS, tv, knees=Q_BLOCKS).velocity}
    for est, lab in (("cs", "centroid"), ("sr", "spectral ratio")):
        cal[f"intrinsic Q ({lab}) + Backus"] = physical(iq[(est, intr)])
        cal[f"apparent Q ({lab}) + Backus"] = physical(iq[(est, appa)])
        cal[f"apparent Q ({lab}), no Backus"] = physical(iq[(est, appa)], backus=False)
    cal["model Q + Backus"] = C.physical_calibration(
        z, logs.vp, rho, model.qp[model.layer_index(z)], F_SONIC, F_SEIS, backus_length=BACKUS)
    resid = {}
    print("  residual drift after calibration: RMS / at 1650 m (ms)")
    for name, v in cal.items():
        resid[name] = C.drift(z, v, DEPTHS, tv)
        print(f"  {name:38s} {1e3 * np.sqrt(np.nanmean(resid[name] ** 2)):5.2f} / {1e3 * resid[name][-1]:+5.2f}")
    v_backus = C.backus_vertical_velocity(z, nl30.vp, nl30.rho, BACKUS)
    lay = C.sonic_time(z, v_backus)[-1] - C.sonic_time(z, nl30.vp)[-1]
    print(f"  layering (Backus {BACKUS:.0f} m vs time average) over the logged interval: {1e3 * lay:.2f} ms")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from iwt.plotting import depth_profiles, gather_panels, trace_columns
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    scale = np.percentile(np.abs(vsp.data), 99)
    gather_panels(t, DEPTHS, {"ZO-VSP (vertical, noise added)": vsp.data / scale,
                              "downgoing": r["down"] / scale,
                              "upgoing (x 10)": 10.0 * r["up"] / scale},
                  tmin=0.2, tmax=1.4, clip=1.0, xlabel="receiver depth (m TVDSS)",
                  title="ZO-VSP of the synthetic well and its parametric wavefield separation"
                  ).savefig(FIG / "wp3_vsp.png", dpi=150)
    m_tops = {n: float(model.twt(zz)) for zz, n in tops if n in ("Hekkingen", "Stø", "Fruholmen")}
    sc = np.max(np.abs(born[w]))
    trace_columns(t, {"corridor stack": (corridor, sc), "full stack": (full_stack, sc),
                      "Born primaries": (born, sc),
                      "primaries with transmission loss": (prim_tl, sc)},
                  tmin=2 * tv.min(), tmax=2 * tv.max(), marks=m_tops,
                  title="Corridor stack (100 ms) against the true reflectivity (30 Hz Ricker,"
                        " one amplitude scale)").savefig(FIG / "wp3_corridor.png", dpi=150)

    def steps(q):
        zz = np.repeat(np.concatenate([q.top, q.bottom[-1:]]), 2)[1:-1]
        return np.repeat(1e3 * q.inv_q, 2), zz

    tq_steps = (np.repeat(1e3 * tq, 2), np.repeat(edges, 2)[1:-1])
    depth_profiles({
        "t* relative to 450 m (centroid shift)": ("t* (ms)", {
            "apparent": (1e3 * r["cs"], DEPTHS),
            "stratigraphic (log model)": (1e3 * rs_log["cs"], DEPTHS),
            "intrinsic = apparent - stratigraphic": (1e3 * (r["cs"] - rs_log["cs"]), DEPTHS)}),
        "interval attenuation": ("1/Q x 1000", {
            "apparent": steps(iq[("cs", appa)]),
            "intrinsic (VSP)": steps(iq[("cs", intr)]),
            "model": tq_steps}),
    }, top=440.0, bottom=1660.0, tops=tops, title="Q from the ZO-VSP: apparent, stratigraphic"
       " and intrinsic").savefig(FIG / "wp3_q.png", dpi=150)
    depth_profiles({
        "drift by pick convention": ("drift (ms)", {
            k: (1e3 * v, DEPTHS) for k, v in drifts.items()} | {
            "intrinsic dispersion only": (1e3 * disp, DEPTHS)}),
        "residual after calibration": ("residual drift (ms)", {
            k: (1e3 * resid[k], DEPTHS) for k in (
                "knee (knees at 600, 740, 800 m)", "intrinsic Q (centroid) + Backus",
                "apparent Q (spectral ratio) + Backus", "apparent Q (centroid) + Backus")}),
        "interval velocity (10 m)": ("m/s", {
            "plain differencing": (raw.velocity, 0.5 * (raw.top + raw.bottom)),
            "regularized": (reg.velocity, 0.5 * (reg.top + reg.bottom)),
            "model, 30 Hz": (v_true, 0.5 * (DEPTHS[1:] + DEPTHS[:-1]))}),
    }, top=440.0, bottom=1660.0, tops=tops, title="Sonic-to-VSP drift and its calibration",
       width=3.3).savefig(FIG / "wp3_drift.png", dpi=150)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
