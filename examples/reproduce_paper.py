"""Work package 7: the integrated loop of Lewis et al. (2016) on the synthetic well.

The article refines one earth model with the logs, the ZO-VSP, the walkaway VSP and
the surface seismic "until the predictions match the measurements, or until the
remaining discrepancies are explained".  This script runs that loop with the
methods of WP1-WP6 and records it in a register (``iwt.workflow``): every model
version, every discrepancy (observation, hypothesis, test, result, decision) and
the acceptance criteria fixed in the plan before any tuning.

0. v0: the model from the logs as logged (isotropic, no Q).
1. ZO-VSP (WP3): time-depth, interval Q, drift; v1 = sonic moved to 30 Hz with the
   VSP intrinsic Q, the blind gap above the logs filled from the VSP.
   Fig. 2 analogue: corridor stack against the v1 synthetic.
2. Anisotropy (WP4 + WP5): sonic gamma, walkaway eps and delta at six levels
   (priors from this loop's ZO-VSP and dipole), the gamma -> (eps, delta)
   calibration; v2 = v1 with eps, delta and gamma curves.
3. Surface seismic (WP6): the "observed" CMP gather is the full-physics
   reflectivity response of the true earth (internal multiples, mode conversion,
   Q; no free surface: assumed deghosted).  It is migrated with v2 (VTI, Q
   compensation) and compared with elastic synthetics of v2, as in the article
   (Figs. 2 and 4): the near-offset tie, and the AVA of the Stø and Fruholmen
   tops.
4. Discrepancies: contamination by multiples and mode conversion (full against
   Born gathers), then, on contamination-free (Born) data, the reservoir Vs
   (dipole against dispersion-corrected: the article's "rock-physics Vs"), and an
   identifiability study.  On synthetic data the truth is known: the true earth's
   own prediction separates model from processing errors (the measurement is
   re-migrated with the true Q), and swapping one part of v2 into the true earth
   tests whether the model errors can be attributed one part at a time.
5. Fig. 5 analogue: a dipping reservoir with a flat oil-water contact, imaged at
   zero offset with and without anisotropy and Q compensation.

Writes ``docs/figures/wp7_*.png`` and ``docs/discrepancy_register.md`` and prints
the numbers quoted in the README (about 6 minutes with 4 workers).

    python examples/reproduce_paper.py [--workers 4]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wp3_zovsp as wp3          # noqa: E402
import wp5_walkaway as wp5       # noqa: E402

from iwt import avo, paper, sonic as S, welltie as T, workflow as WF   # noqa: E402
from iwt import calibration as C                                        # noqa: E402
from iwt import imaging as I                                            # noqa: E402
from iwt import multiples as M                                          # noqa: E402
from iwt.calibration import bayes_linear                                # noqa: E402
from iwt.modeling.convolution import normal_incidence, ricker           # noqa: E402
from iwt.modeling.reflection import exact_pp                            # noqa: E402
from iwt.qmodel import phase_velocity                                   # noqa: E402
from iwt.raytrace import reflection_table                               # noqa: E402
from iwt.synthetic import (OWC_TVDSS, SEA_WATER, dipping_reservoir_models,   # noqa: E402
                           wisting_like_well)
from iwt.vsp import attenuation as A                                    # noqa: E402
from iwt.vsp import walkaway as W                                       # noqa: E402
from iwt.vsp import zovsp as Z                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OFFSETS = np.arange(100.0, 1001.0, 50.0)
DT, NT = 0.001, 1150
Z_S, Z_R = 6.0, 8.0
X_AVA = 850.0
FIT = (20.0, 44.0)
EVENTS = {"Stø top": paper.STO_TOP_TVDSS, "Fruholmen top": 740.0}
BORN = dict(internal_multiples=False, converted_waves=False, transmission_loss=False)
OBSERVED = {"full": {}, "primaries": dict(internal_multiples=False),
            "P only": dict(converted_waves=False),
            "P primaries": dict(internal_multiples=False, converted_waves=False), "Born": BORN}
BOTTOM = 780.0


def unit_mean(m, a, b, name):
    s = m.split_at([a, b])
    i = (s.mid > a) & (s.mid < b)
    return float(np.sum(getattr(s, name)[i] * s.thickness[i]) / np.sum(s.thickness[i]))


# -- 1. ZO-VSP --------------------------------------------------------------------------------


def zovsp_step(model, logs, reg, workers):
    z = logs.depth
    v0 = reg.new_version("logs", WF.model_from_logs(logs, paper.WATER_DEPTH, SEA_WATER.vp0,
                                                    SEA_WATER.rho, disperse_vs=False),
                         "sonic, dipole and density as logged (12 kHz / 2 kHz); isotropic; no Q")
    kw = dict(f_max=wp3.F_MAX, workers=workers)
    clean = Z.synthetic_zovsp(model, wp3.DEPTHS, wp3.SOURCE_DEPTH, wp3.DT, wp3.NT, wp3.WAVELET, **kw)
    vsp = clean.with_data(Z.add_noise(clean.data, wp3.NOISE, wp3.DT, wp3.F_MAX, seed=3))
    strat = Z.synthetic_zovsp(v0.model, wp3.DEPTHS, wp3.SOURCE_DEPTH, wp3.DT, wp3.NT, wp3.WAVELET,
                              attenuation=False, **kw)
    t = vsp.t
    r, rs = wp3.process(t, vsp.data), wp3.process(t, strat.data)
    tv = Z.vertical_times(r["phase"], wp3.DEPTHS, wp3.SOURCE_DEPTH, datum_velocity=SEA_WATER.vp0)
    q = A.interval_q(wp3.DEPTHS, tv, r["cs"] - rs["cs"], wp3.Q_BLOCKS)
    qz = q.at(z, q.q[0])
    d0 = C.drift(z, logs.vp, wp3.DEPTHS, tv)
    vcal = C.physical_calibration(z, logs.vp, logs.rho, qz, wp3.F_SONIC, wp3.F_SEIS,
                                  backus_length=wp3.BACKUS)
    d1 = C.drift(z, vcal, wp3.DEPTHS, tv)
    edges = np.concatenate([[wp3.DEPTHS[0]], wp3.Q_BLOCKS, [wp3.DEPTHS[-1]]])
    q_true = 1.0 / wp3.true_inv_q(model, edges)
    print("1. ZO-VSP: intrinsic Q per block (centroid shift, strat. from the log model) | true: "
          + "  ".join(f"{a:.0f} | {b:.0f}" for a, b in zip(q.q, q_true)))
    print(f"   drift (VSP - integrated sonic): {1e3 * d0[-1]:.1f} ms at {wp3.DEPTHS[-1]:.0f} m, "
          f"RMS {1e3 * np.sqrt(np.nanmean(d0 ** 2)):.1f} ms; after the physical calibration "
          f"RMS {1e3 * np.sqrt(np.nanmean(d1 ** 2)):.2f} ms")
    reg.log("Integrated sonic time short of the VSP time by "
            f"{1e3 * d0[-1]:.0f} ms at {wp3.DEPTHS[-1]:.0f} m",
            "sonic dispersion (12 kHz vs 30 Hz) and layering",
            "move the sonic to 30 Hz with the VSP intrinsic Q, Backus over 40 m",
            f"residual drift {1e3 * np.sqrt(np.nanmean(d1 ** 2)):.2f} ms RMS", "adopt", 1)
    # the corridor stack against the synthetic, before and after filling the blind gap
    target = ricker(30.0, wp3.DT)
    dec = Z.deconvolve(t, r["up"], r["down"], tv, target)
    corridor, _ = Z.corridor_stack(t, dec, tv, length=0.1)
    w = (t > 2 * tv.min() + 0.02) & (t < 2 * tv.max())
    base = dict(qp=qz, qs=0.5 * qz, disperse_vs=False)
    m_a = WF.model_from_logs(logs, paper.WATER_DEPTH, SEA_WATER.vp0, SEA_WATER.rho, **base)
    tie_a = T.tie_metrics(corridor[w], normal_incidence(m_a, wp3.DT, wp3.NT, target)[1][w], wp3.DT)
    t_sb = paper.WATER_DEPTH / SEA_WATER.vp0
    v_gap = (z[0] - paper.WATER_DEPTH) / (np.interp(z[0], wp3.DEPTHS, tv) - t_sb)
    m_b = WF.model_from_logs(logs, paper.WATER_DEPTH, SEA_WATER.vp0, SEA_WATER.rho, gap_vp=v_gap,
                             **base)
    syn = normal_incidence(m_b, wp3.DT, wp3.NT, target)[1]
    tie_b = T.tie_metrics(corridor[w], syn[w], wp3.DT)
    true_gap = (z[0] - paper.WATER_DEPTH) / float(model.vertical_time(z[0]) - model.vertical_time(paper.WATER_DEPTH))
    print(f"   corridor stack vs synthetic: gap from the first logs: correlation {tie_a.correlation:.2f}, "
          f"shift {1e3 * tie_a.shift:+.1f} ms; gap from the VSP ({v_gap:.0f} m/s, true {true_gap:.0f}): "
          f"correlation {tie_b.correlation:.2f} (best {tie_b.max_correlation:.2f}), shift "
          f"{1e3 * tie_b.shift:+.1f} ms, phase {np.degrees(tie_b.phase):+.0f} deg")
    reg.log(f"Synthetic {1e3 * tie_a.shift:.0f} ms late against the corridor stack "
            f"(zero-lag correlation {tie_a.correlation:.2f})",
            "the 418-470 m gap above the logs (Quaternary) is filled with log velocities",
            "fill the gap with the VSP interval velocity",
            f"correlation {tie_b.correlation:.2f}, shift {1e3 * tie_b.shift:+.1f} ms", "adopt", 1)
    v1 = reg.new_version("ZO-VSP calibrated", m_b,
                         "Vp at 30 Hz with VSP intrinsic Q; Qp from the VSP, Qs = Qp/2; gap from VSP")
    reg.check("VSP first-break residual after calibration", 1e3 * np.sqrt(np.nanmean(d1 ** 2)), 1.0,
              "<=", "ms")
    reg.check("corridor stack vs synthetic, correlation", tie_b.correlation, 0.8, ">=")
    return dict(tv=tv, qz=qz, v_gap=v_gap, t=t, corridor=corridor, syn=syn, w=w, v1=v1, tie=tie_b)


# -- 2. anisotropy ----------------------------------------------------------------------------


def anisotropy_step(model, logs, reg, zo):
    z = logs.depth
    g_sonic, g_up = wp5.sonic_gamma(model, logs)
    earth40 = model.backus(wp5.SUBARRAY, respect=[model.interfaces[1]])
    survey = W.synthetic_walkaway(earth40, wp5.OFFSETS, wp5.DEPTHS, time_noise=wp5.SIGMA_T,
                                  psi_noise=wp5.SIGMA_PSI, seed=5)
    est, rows = [], []
    for lv in wp5.LEVELS:
        d = W.measure_slowness(survey, lv, time_noise=wp5.SIGMA_T, psi_noise=wp5.SIGMA_PSI)
        th = np.degrees(d.theta)
        d = d.select((th >= wp5.ANGLES[0]) & (th <= wp5.ANGLES[1]))
        a, b = lv - wp5.SUBARRAY / 2, lv + wp5.SUBARRAY / 2
        vp_prior = (b - a) / (np.interp(b, wp3.DEPTHS, zo["tv"]) - np.interp(a, wp3.DEPTHS, zo["tv"]))
        win = (z >= a) & (z <= b)
        c44 = 1.0 / np.mean(1.0 / (logs.rho[win] * logs.vs[win] ** 2))
        vs_b = np.sqrt(c44 / np.mean(logs.rho[win]))
        vs_prior = float(phase_velocity(vs_b, 0.5 * np.interp(lv, z, zo["qz"]), 30.0,
                                        logs.meta["f_sonic_s"]))
        post = W.invert_level(d, {"vp0": (vp_prior, 0.01 * vp_prior),
                                  "vs0": (vs_prior, 0.05 * vs_prior)})
        est.append((post["eps"], post["delta"]))
        t = earth40.at(lv)
        rows.append(f"{lv:.0f} m: eps {post['eps'][0]:.3f} ({float(t['eps']):.3f}), delta "
                    f"{post['delta'][0]:+.3f} ({float(t['delta']):+.3f})")
    gx = np.array([np.interp(lv, z, g_up) for lv in wp5.LEVELS])
    cal = {}
    for k, n in enumerate(("eps", "delta")):
        y = np.array([e[k][0] for e in est])
        s = np.array([e[k][1] for e in est])
        cal[n] = bayes_linear(gx, y, s, prior_mean=(0.0, 0.5), prior_std=(0.1, 2.0))
    ok = np.isfinite(g_sonic)
    g5 = S.running_average(z, np.interp(z, z[ok], g_sonic[ok]), 5.0)
    eps_c, delta_c = cal["eps"].predict(g5)[0], cal["delta"].predict(g5)[0]
    print("\n2. Walkaway (priors: Vp0 from this ZO-VSP, Vs0 from the dipole at 30 Hz), estimate (true):")
    print("   " + "; ".join(rows))
    print(f"   eps = {cal['eps'].coef[0]:+.3f} + {cal['eps'].coef[1]:.2f} gamma, delta = "
          f"{cal['delta'].coef[0]:+.3f} + {cal['delta'].coef[1]:.2f} gamma")
    ti = model.layer_index(z)
    units = ((470, 600, "L. Cret."), (600, 631, "Hekkingen"), (631, 672, "Fuglen"),
             (672, 702, "Stø"), (702, 740, "Nordmela"), (740, 800, "Fruholmen"))
    print("   unit means, calibrated (true): " + "; ".join(
        f"{n} eps {eps_c[(z >= a) & (z < b)].mean():.3f} ({model.eps[ti][(z >= a) & (z < b)].mean():.3f})"
        f" delta {delta_c[(z >= a) & (z < b)].mean():+.3f} ({model.delta[ti][(z >= a) & (z < b)].mean():+.3f})"
        for a, b, n in units))
    reg.log("Isotropic v1 cannot predict non-hyperbolic moveout or walkaway slownesses",
            "VTI overburden (shales)", "slowness-polarization inversion at six levels, "
            "gamma -> (eps, delta) calibration",
            f"eps slope {cal['eps'].coef[1]:.2f}, delta slope {cal['delta'].coef[1]:.2f} per gamma",
            "adopt eps, delta, gamma curves", 2)
    v1 = zo["v1"].model
    v2m = WF.model_from_logs(logs, paper.WATER_DEPTH, SEA_WATER.vp0, SEA_WATER.rho, qp=zo["qz"],
                             qs=0.5 * zo["qz"], eps=eps_c, delta=delta_c, gamma=g5,
                             gap_vp=zo["v_gap"], disperse_vs=False)
    assert np.allclose(v2m.vp0, v1.vp0)
    v2 = reg.new_version("anisotropic", v2m, "eps, delta from the calibrated sonic gamma")
    return dict(v2=v2, g5=g5, eps=eps_c, delta=delta_c, cal=cal, qz=zo["qz"])


# -- 3. surface seismic -----------------------------------------------------------------------


def process(data, mig, q_comp, sel, z, tab):
    t = np.arange(data.shape[1]) * DT
    t_sb = reflection_table(mig, [mig.interfaces[1] - 0.01], OFFSETS, Z_S, Z_R).time[0]
    w = np.clip((t[None, :] - (t_sb[:, None] + 0.035)) / 0.01, 0.0, 1.0)
    mute = 0.5 - 0.5 * np.cos(np.pi * w)
    return I.migrate_cmp((data * mute)[sel], DT, OFFSETS[sel], mig, z, Z_S, Z_R,
                         q_compensation="amplitude" if q_comp else None, gain_limit=10.0,
                         high_cut=80.0, table=tab)


def ava(img, z0):
    return avo.pick_event(img.depth, img.data, img.theta, z0, 6.0)


def misfit(ref, other, scale=1.0):
    g = np.radians(np.arange(FIT[0], FIT[1] + 0.01, 1.0))
    a = np.interp(g, ref.theta, ref.amplitude)
    b = np.interp(g, other.theta, other.amplitude * scale)
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2)))


def surface_step(model, reg, an, workers):
    v2 = an["v2"].model
    wav = ricker(30.0, DT)
    run = dict(dt=DT, nt=NT, wavelet=wav, f_max=100.0, workers=workers)
    t0 = time.time()
    obs = M.gathers(model.subset(0.0, BOTTOM), OFFSETS, Z_S, Z_R, physics=OBSERVED,
                    attenuation=True, **run)
    pred = M.gathers(v2.subset(0.0, BOTTOM), OFFSETS, Z_S, Z_R, physics={"full": {}, "Born": BORN},
                     attenuation=False, **run)
    print(f"\n3. Surface CMP gathers (reflectivity, 19 offsets to 1000 m, 7 physics variants): "
          f"{time.time() - t0:.0f} s")
    mig = v2.backus(20.0, respect=[v2.interfaces[1]])
    sel = OFFSETS <= X_AVA
    z = np.arange(600.0, BOTTOM, 0.5)
    tab = reflection_table(mig, z, OFFSETS[sel], Z_S, Z_R)
    meas = {k: process(v, mig, True, sel, z, tab) for k, v in obs.items()}
    pr = {k: process(v, mig, False, sel, z, tab) for k, v in pred.items()}
    near = OFFSETS[sel] <= 300.0
    wz = (z >= 620.0) & (z <= 770.0)
    stack = lambda im: np.nanmean(im.data[:, near], axis=1)       # noqa: E731
    tie = T.tie_metrics(stack(meas["full"])[wz], stack(pr["full"])[wz], 0.5, max_shift=10.0)
    v_res = unit_mean(mig, 640.0, 760.0, "vp0")
    print(f"   near-offset (<= 300 m) migrated stack vs elastic synthetic of v2, 620-770 m: "
          f"correlation {tie.correlation:.2f} (best {tie.max_correlation:.2f}), shift {tie.shift:+.1f} m "
          f"({2e3 * tie.shift / v_res:+.1f} ms), phase {np.degrees(tie.phase):+.0f} deg, scale "
          f"{tie.scale:.2f}")
    reg.check("reservoir event time shift (near-offset tie)", abs(2e3 * tie.shift / v_res), 2.0,
              "<=", "ms")
    out = dict(meas=meas, pred=pr, z=z, sel=sel, tab=tab, mig=mig, tie=tie, stack=stack, wz=wz,
               obs=obs)
    rows = {}
    for name, z0 in EVENTS.items():
        z0m = z0 - 2.0
        pm, pp = ava(meas["full"], z0m), ava(pr["full"], z0m)
        pmb, ppb = ava(meas["Born"], z0m), ava(pr["Born"], z0m)
        rows[name] = dict(full=(pm, pp), born=(pmb, ppb))
        fm, fp = (avo.intercept_gradient(p.theta, p.amplitude, FIT) for p in (pm, pp))
        print(f"   {name}: as acquired (full physics), predicted/measured gradient "
              f"{fp.gradient / fm.gradient:+.2f}, AVA misfit {misfit(pm, pp):.2f}; "
              f"contamination-free (Born) misfit {misfit(pmb, ppb):.2f}")
    out["rows"] = rows
    return out


def contamination(reg, sf):
    """RMS differences between physics variants in the migrated reservoir windows, relative to
    the Born primaries: all offsets, near (<= 300 m) and far."""
    meas, z = sf["meas"], sf["z"]
    near = OFFSETS[sf["sel"]] <= 300.0
    out = {}
    for name, (lo, hi) in (("Stø", (650.0, 690.0)), ("Fruholmen", (720.0, 760.0))):
        wz = (z >= lo) & (z <= hi)
        for part, cols in (("all", slice(None)), ("near", near), ("far", ~near)):
            d = {k: im.data[wz][:, cols] for k, im in meas.items()}
            born = np.sqrt(np.nanmean(d["Born"] ** 2))
            rel = lambda a: float(np.sqrt(np.nanmean(a ** 2)) / born)     # noqa: E731
            out[name, part] = {"full - Born": rel(d["full"] - d["Born"]),
                               "P multiples": rel(d["P only"] - d["P primaries"]),
                               "P primaries": rel(d["P primaries"]),
                               "primaries - Born": rel(d["primaries"] - d["Born"]),
                               "full - primaries": rel(d["full"] - d["primaries"])}
    print("   contamination after migration, rms relative to the Born primaries "
          "(all offsets / <= 300 m / beyond):")
    for key in out["Stø", "all"]:
        print(f"     {key:17s} " + "; ".join(
            f"{n} " + " / ".join(f"{out[n, part][key]:.2f}" for part in ("all", "near", "far"))
            for n in ("Stø", "Fruholmen")))
    st = sf["rows"]["Stø top"]
    a, b = out["Stø", "all"], out["Fruholmen", "all"]
    reg.log("Reservoir AVA beyond 20 deg erratic and not predicted (misfit "
            f"{misfit(*st['full']):.2f} at the Stø top)",
            "interbed multiples and mode conversion from the overburden stringers",
            "compare physics variants of the true-earth gathers after identical processing",
            f"full - Born = {a['full - Born']:.1f} x Born (Stø), {b['full - Born']:.1f} x (Fruholmen), "
            f"growing with offset ({out['Stø', 'near']['full - Born']:.2f} at <= 300 m); "
            f"without mode conversion the P-wave multiples are only {a['P multiples']:.2f}, "
            f"{b['P multiples']:.2f} x Born; converted primaries and multiples each exceed the "
            "total and largely cancel; contamination-free misfit "
            f"{misfit(*st['born']):.2f}",
            "elastic (converted-wave) contamination: an acoustic demultiple would not repair "
            "the AVA; test the remaining hypotheses on contamination-free data")
    return out


VS = "dipole Vs at 30 Hz (Qs = Qp/2)"
TRUE = "true earth"
TRUE_ANISO = "true earth, calibrated eps and delta"
TRUE_ELASTIC = "true earth, log Vp, Vs and rho"


def scenarios(model, logs, reg, an, sf, workers):
    """Born-domain predictions: the Vs hypothesis, then (synthetic only) the truth, alone and
    with one part of v2 swapped in, and the measurement re-migrated with the true Q."""
    v2 = an["v2"].model
    v3m = WF.model_from_logs(logs, paper.WATER_DEPTH, SEA_WATER.vp0, SEA_WATER.rho,
                             qp=an["qz"], qs=0.5 * an["qz"], eps=an["eps"], delta=an["delta"],
                             gamma=an["g5"], gap_vp=float(v2.vp0[1]), disperse_vs=True)
    true = model.subset(0.0, BOTTOM)
    below = true.mid > logs.depth[0]
    est = v2.at(true.mid[below])
    t_aniso, t_elastic = true.copy(), true.copy()
    for k in ("eps", "delta"):
        getattr(t_aniso, k)[below] = est[k]
    for k in ("vp0", "vs0", "rho"):
        getattr(t_elastic, k)[below] = est[k]
    models = {VS: v3m.subset(0.0, BOTTOM), TRUE: true, TRUE_ANISO: t_aniso, TRUE_ELASTIC: t_elastic}
    run = dict(dt=DT, nt=NT, wavelet=ricker(30.0, DT), f_max=100.0, workers=workers)
    t0 = time.time()
    preds = {k: M.gathers(m, OFFSETS, Z_S, Z_R, physics={"Born": BORN}, attenuation=False,
                          **run)["Born"] for k, m in models.items()}
    print(f"\n4. Scenario predictions (Born, elastic): {time.time() - t0:.0f} s")
    out = {"v2 (dipole Vs as logged)": sf["pred"]["Born"]}
    for k, d in preds.items():
        out[k] = process(d, sf["mig"], False, sf["sel"], sf["z"], sf["tab"])
    table, ratio = {}, {}
    for name, z0 in EVENTS.items():
        pm = sf["rows"][name]["born"][0]
        row = {}
        for k, im in out.items():
            p = ava(im, z0 - 2.0)
            row[k] = misfit(pm, p)
            ratio[name, k] = np.interp(pm.theta, p.theta, p.amplitude) / pm.amplitude
        table[name] = row
        print(f"   {name}: AVA misfit against the contamination-free measurement: "
              + "; ".join(f"{k} {v:.3f}" for k, v in row.items()))
    v3 = None
    gain = {k: np.mean([table[e][k] for e in EVENTS]) for k in (VS, "v2 (dipole Vs as logged)")}
    adopt = gain[VS] < gain["v2 (dipole Vs as logged)"]
    st, fr = table["Stø top"], table["Fruholmen top"]
    reg.log(f"Contamination-free reservoir AVA still off: misfit {st['v2 (dipole Vs as logged)']:.2f} "
            f"(Stø), {fr['v2 (dipole Vs as logged)']:.2f} (Fruholmen)",
            "sonic and seismic see Vs differently: dipole Vs is dispersed (2 kHz vs 30 Hz)",
            "Vs moved to 30 Hz with Qs = Qp/2 (the article's rock-physics Vs analogue)",
            f"misfit {st[VS]:.2f} (Stø), {fr[VS]:.2f} (Fruholmen)",
            "adopt" if adopt else "reject: no overall improvement", 3 if adopt else None)
    if adopt:
        v3 = reg.new_version("seismic-frequency Vs", v3m, "dipole Vs moved to 30 Hz (Qs = Qp/2)")
    # model or processing?  The true earth's own prediction (synthetic only)
    th = sf["rows"]["Stø top"]["born"][0].theta
    r_true = ratio["Stø top", TRUE][(th >= np.radians(FIT[0])) & (th <= np.radians(FIT[1]))]
    sel = sf["sel"]
    tq = model.backus(20.0, respect=[model.interfaces[1]])
    q_fix = {}
    for label, upper in (("above the logs", logs.depth[0]), ("everywhere", np.inf)):
        mq = sf["mig"].copy()
        k = mq.mid < upper
        mq.qp[k] = tq.at(mq.mid[k])["qp"]
        mq.qp[0] = np.inf
        tab = reflection_table(mq, sf["z"], OFFSETS[sel], Z_S, Z_R)
        mb = process(sf["obs"]["Born"], mq, True, sel, sf["z"], tab)
        q_fix[label] = np.mean([misfit(ava(mb, z0 - 2.0), ava(out[TRUE], z0 - 2.0))
                                for z0 in EVENTS.values()])
    print(f"   true-earth prediction / measurement at the Stø top, 20-44 deg: "
          f"{r_true.min():.2f}-{r_true.max():.2f}; "
          f"mean misfit {np.mean([table[e][TRUE] for e in EVENTS]):.3f}, with the true Q above the "
          f"logs {q_fix['above the logs']:.3f}, everywhere {q_fix['everywhere']:.3f}")
    reg.log(f"(synthetic only) The true earth itself predicts {100 * (np.median(r_true) - 1):+.0f}% "
            "amplitude at all angles",
            "Q compensation too weak: no VSP receivers between the seabed and 450 m, so the "
            f"Quaternary (true Q 30) takes the first block's Q ({v2.qp[1]:.0f})",
            "re-migrate the measurement with the true Q above the logs, then everywhere",
            f"mean misfit {np.mean([table[e][TRUE] for e in EVENTS]):.3f} -> "
            f"{q_fix['above the logs']:.3f} -> {q_fix['everywhere']:.3f}",
            "explained; constrain the shallow t* (VSP source signature, seabed multiples) on "
            "field data")
    reg.log("(synthetic only) v2's Stø AVA falls with angle against the true earth's",
            "the calibrated eps and delta, or the log resolution of the stringers",
            "swap one part of v2 into the true earth",
            f"Stø misfit {st[TRUE_ANISO]:.2f} with v2's eps and delta, {st[TRUE_ELASTIC]:.2f} with "
            f"v2's Vp, Vs and rho; {st['v2 (dipole Vs as logged)']:.2f} with both",
            "not separable: the 5 m-smoothed gamma gives the isotropic stringers shale anisotropy, "
            "which v2's equally smoothed velocities partly cancel; keep one resolution")
    return dict(table=table, images=out, v3=v3, ratio=ratio, q_fix=q_fix)


def identifiability(sf, an):
    """Which single change explains the contamination-free Stø-top AVA residual, and what
    the residual can resolve jointly (linearized, exact interface coefficient + Q)."""
    v2 = an["v2"].model
    pm, pp = sf["rows"]["Stø top"]["born"]
    th = np.radians(np.arange(FIT[0], FIT[1] + 0.01, 2.0))
    resid = np.interp(th, pm.theta, pm.amplitude) / np.interp(th, pp.theta, pp.amplitude) - 1.0
    z0 = paper.STO_TOP_TVDSS
    up = [unit_mean(v2, z0 - 8.0, z0, k) for k in ("vp0", "vs0", "rho", "eps", "delta")]
    lo = [unit_mean(v2, z0, z0 + 8.0, k) for k in ("vp0", "vs0", "rho", "eps", "delta")]
    R0 = exact_pp(th, tuple(up), tuple(lo)).real
    t_star = np.interp(th, sf["tab"].theta[np.searchsorted(sf["z"], z0) - 1],
                       sf["tab"].tstar[np.searchsorted(sf["z"], z0) - 1])

    def rel(side, k, step):
        u, lw = list(up), list(lo)
        (u if side == "above" else lw)[k] += step
        return (exact_pp(th, tuple(u), tuple(lw)).real / R0 - 1.0) / step

    params = {  # name: (sensitivity of the relative amplitude, prior std, unit)
        "Vs below (Stø)": (rel("below", 1, 10.0) * lo[1] / 100.0, 3.0, "%"),
        "Vs above (Fuglen)": (rel("above", 1, 10.0) * up[1] / 100.0, 3.0, "%"),
        "rho below": (rel("below", 2, 5.0) * lo[2] / 100.0, 2.0, "%"),
        "delta below": (rel("below", 4, 0.005), 0.03, ""),
        "eps below": (rel("below", 3, 0.005), 0.03, ""),
        "overburden Q": (-np.pi * 35.0 * t_star / 100.0, 30.0, "%"),   # 1 % lower Q in the model
    }
    print("\n   identifiability: the contamination-free Stø-top residual (measured/predicted - 1) is "
          + ", ".join(f"{100 * r:+.0f}%" for r in resid[::3]) + f" at {', '.join(f'{d:.0f}' for d in np.degrees(th[::3]))} deg")
    sig = 0.02
    for k, (j, _, unit) in params.items():
        dp = float(j @ resid / (j @ j))
        chi = np.sqrt(np.mean((resid - j * dp) ** 2)) / np.sqrt(np.mean(resid ** 2))
        print(f"     {k:18s} alone: change {dp:+.3g}{unit}, explains the residual to "
              f"{100 * chi:.0f}% of its RMS")
    J = np.column_stack([v[0] for v in params.values()])
    res = avo.linear_resolution(J, [v[1] for v in params.values()], sig ** 2 * np.eye(th.size))
    names = list(params)
    print("     jointly (2 % data noise per angle): data weight " + ", ".join(
        f"{n} {w:.2f}" for n, w in zip(names, res.data_weight)))
    i, j_ = names.index("Vs below (Stø)"), names.index("delta below")
    print(f"     posterior correlation Vs below / delta below {res.correlation[i, j_]:+.2f}, "
          f"Vs below / overburden Q {res.correlation[i, names.index('overburden Q')]:+.2f}")
    return dict(resid=resid, th=th, params=names, resolution=res)


# -- 5. the Fig. 5 analogue -------------------------------------------------------------------


def section_step():
    xs = np.arange(0.0, 1001.0, 20.0)
    wav = ricker(30.0, DT)
    oil, brine = dipping_reservoir_models(xs), dipping_reservoir_models(xs, oil=False)
    nt = 1000
    sec = {k: np.array([normal_incidence(m, DT, nt, wav, attenuation=True, f_max=100.0)[1]
                        for m in ms]) for k, ms in (("oil", oil), ("brine", brine))}
    rng = np.random.default_rng(4)
    t = np.arange(nt) * DT
    noise = rng.standard_normal(sec["oil"].shape)
    f = np.fft.rfftfreq(nt, DT)
    spec = np.fft.rfft(noise)
    spec[..., f > 100.0] = 0.0
    noise = np.fft.irfft(spec, n=nt)
    win = (t > 0.7) & (t < 0.85)
    noise *= 0.05 * np.sqrt(np.mean(sec["oil"][:, win] ** 2)) / noise.std()
    z = np.arange(600.0, 760.0, 0.5)
    from scipy.signal import hilbert

    def image(data, iso, qc):
        out = np.empty((xs.size, z.size))
        for j, m in enumerate(oil):
            mm = m.backus(20.0, respect=[m.interfaces[1]])
            if iso:
                mm = mm.copy()
                mm.vp0 = mm.vp0 * np.sqrt(1.0 + 2.0 * mm.delta)
                mm.eps[:] = 0.0
                mm.delta[:] = 0.0
            out[j] = I.migrate_cmp(data[j][None, :], DT, [0.0], mm, z, 0.0, 0.0,
                                   q_compensation="amplitude" if qc else None, gain_limit=10.0,
                                   high_cut=80.0, true_amplitude=False).data[:, 0]
        return out

    col = OWC_TVDSS - (paper.STO_TOP_TVDSS + xs * np.tan(np.radians(1.2)))
    variants = {"VTI + Q compensation": (False, True), "VTI, no Q compensation": (False, False),
                "isotropic (NMO v) + Q compensation": (True, True),
                "isotropic (NMO v), no Q compensation": (True, False)}
    print("\n5. Dipping reservoir, flat oil-water contact at 694.5 m (oil column 22 -> 2 m over 1 km), "
          "zero-offset images, 5 % noise")
    images, fluid, metrics = {}, {}, {}
    for name, (iso, qc) in variants.items():
        so, sb, sn = image(sec["oil"] + noise, iso, qc), image(sec["brine"], iso, qc), image(noise, iso, qc)
        diff = so - sb - sn                       # the noise-free fluid signal
        env = np.abs(hilbert(diff, axis=1))
        thick = np.flatnonzero(col >= 12.0)
        pk = [z[np.argmax(env[j])] for j in thick]
        width = []
        for j in thick:
            e = env[j] / env[j].max()
            width.append(np.ptp(z[e > 0.5]))
        snr = float(np.max(np.abs(diff)) / np.sqrt(np.mean(sn ** 2)))
        metrics[name] = (np.mean(pk), np.ptp(pk), np.mean(width), snr)
        images[name], fluid[name] = so, diff
        print(f"   {name:38s} fluid signal at {np.mean(pk):.1f} m (spread {np.ptp(pk):.1f} m), "
              f"half-max thickness {np.mean(width):.1f} m, peak / noise RMS {snr:.1f}")
    return dict(xs=xs, z=z, images=images, fluid=fluid, metrics=metrics,
                time=sec["oil"] + noise, t=t)


# -- figures ----------------------------------------------------------------------------------


def figures(zo, sf, sc, sec):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from iwt.plotting import PALETTE
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    col = PALETTE["series"]

    def style(ax):
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(color=PALETTE["grid"], lw=0.5)

    # Fig. 2 analogue
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.0))
    fig.patch.set_facecolor(PALETTE["surface"])
    t, w = zo["t"], zo["w"]
    ax = axes[0]
    sc_ = np.max(np.abs(zo["corridor"][w]))
    ax.plot(zo["corridor"] / sc_, t, color=col[0], lw=1.1, label="VSP corridor stack")
    ax.plot(zo["syn"] / np.max(np.abs(zo["syn"][w])) - 1.2, t, color=col[1], lw=1.1,
            label="synthetic, v1 (shifted left)")
    ax.set_ylim(0.95, 0.6)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("two-way time (s)")
    ax.set_title(f"ZO-VSP tie: correlation {zo['tie'].correlation:.2f}, shift "
                 f"{1e3 * zo['tie'].shift:+.1f} ms", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[1]
    z, wz = sf["z"], sf["wz"]
    m_, p_ = sf["stack"](sf["meas"]["full"]), sf["stack"](sf["pred"]["full"])
    s = np.nanmax(np.abs(m_[wz]))
    ax.plot(m_ / s, z, color=col[0], lw=1.1, label="surface, near offsets (Q-PSDM, v2)")
    ax.plot(p_ * sf["tie"].scale / s - 1.2, z, color=col[1], lw=1.1,
            label="elastic synthetic of v2 (shifted left)")
    for name, z0 in EVENTS.items():
        ax.axhline(z0, color=PALETTE["ink2"], lw=0.6, ls="--")
        ax.annotate(name, (0.62, z0 - 1.5), fontsize=7, color=PALETTE["ink2"])
    ax.set_ylim(770.0, 620.0)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"Surface tie: correlation {sf['tie'].correlation:.2f}, shift "
                 f"{round(sf['tie'].shift, 1) + 0.0:+.1f} m", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "wp7_tie.png", dpi=150)

    # Fig. 4 analogue
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    fig.patch.set_facecolor(PALETTE["surface"])
    for c, (name, z0) in enumerate(EVENTS.items()):
        pm, pp = sf["rows"][name]["full"]
        ax = axes[0, c]
        ax.plot(np.degrees(pm.theta), pm.amplitude, "o-", color=PALETTE["ink"], ms=3, lw=1.6,
                label="measured (as acquired)")
        ax.plot(np.degrees(pp.theta), pp.amplitude, "s--", color=col[1], ms=3, lw=1.2,
                label="elastic synthetic, v2 (full physics)")
        ax.set_title(f"{name}: full physics (multiples, mode conversion)", fontsize=9, loc="left")
        ax = axes[1, c]
        pm = sf["rows"][name]["born"][0]
        ax.plot(np.degrees(pm.theta), pm.amplitude, "o-", color=PALETTE["ink"], ms=3, lw=1.6,
                label="measured, contamination-free")
        for k, (key, im) in enumerate(sc["images"].items()):
            p = ava(im, z0 - 2.0)
            ax.plot(np.degrees(p.theta), p.amplitude, ("s--", "^-.", "v:", "d--", "x-")[k],
                    color=(tuple(col) + (PALETTE["muted"],))[k],
                    ms=3, lw=1.1, label=key)
        ax.set_title(f"{name}: contamination-free (Born)", fontsize=9, loc="left")
    for ax in axes.ravel():
        ax.axvspan(*FIT, color=PALETTE["grid"], alpha=0.5, zorder=0)
        ax.set_xlabel("reflection angle (deg)")
        ax.set_ylabel("migrated amplitude")
        ax.legend(fontsize=6.5, frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "wp7_ava.png", dpi=150)

    # Fig. 5 analogue
    panels = {"oil section, time,\nno Q compensation": None,
              "oil section,\nVTI + Q compensation": sec["images"]["VTI + Q compensation"]}
    panels.update({f"fluid signal (oil - brine),\n{k}": sec["fluid"][k]
                   for k in ("VTI + Q compensation", "isotropic (NMO v) + Q compensation")})
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.8))
    fig.patch.set_facecolor(PALETTE["surface"])
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("d", ("#1c5cab", "#f0efec", "#c23b3a"))
    xs, zz = sec["xs"], sec["z"]
    for ax, (name, im) in zip(axes, panels.items()):
        if im is None:
            tsel = (sec["t"] > 0.72) & (sec["t"] < 0.88)
            d = sec["time"][:, tsel]
            clip = np.percentile(np.abs(d), 99)
            ax.imshow(d.T, aspect="auto", cmap=cmap, vmin=-clip, vmax=clip,
                      extent=[xs[0], xs[-1], sec["t"][tsel][-1], sec["t"][tsel][0]])
            ax.set_ylabel("two-way time (s)")
        else:
            clip = np.max(np.abs(im))
            ax.imshow(im.T, aspect="auto", cmap=cmap, vmin=-clip, vmax=clip,
                      extent=[xs[0], xs[-1], zz[-1], zz[0]])
            ax.axhline(OWC_TVDSS, color=PALETTE["ink2"], lw=0.7, ls="--")
            ax.set_ylim(740.0, 640.0)
            ax.set_ylabel("depth (m)")
        ax.set_title(name, fontsize=8.5, loc="left")
        ax.set_xlabel("distance (m)")
    fig.tight_layout()
    fig.savefig(FIG / "wp7_section.png", dpi=150)
    print("\nfigures written to", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_start = time.time()
    model, logs = wisting_like_well()
    reg = WF.Register()
    zo = zovsp_step(model, logs, reg, args.workers)
    an = anisotropy_step(model, logs, reg, zo)
    sf = surface_step(model, reg, an, args.workers)
    contamination(reg, sf)
    sc = scenarios(model, logs, reg, an, sf, args.workers)
    identifiability(sf, an)
    sec = section_step()
    pm, pp = sf["rows"]["Stø top"]["born"]
    fm, fp = (avo.intercept_gradient(p.theta, p.amplitude, FIT) for p in (pm, pp))
    reg.check("Stø-top AVA gradient, contamination-free, |predicted/measured - 1|",
              abs(fp.gradient / fm.gradient - 1.0), 0.10, "<=")
    pm, pp = sf["rows"]["Stø top"]["full"]
    fm, fp = (avo.intercept_gradient(p.theta, p.amplitude, FIT) for p in (pm, pp))
    reg.check("Stø-top AVA gradient, as acquired, |predicted/measured - 1|",
              abs(fp.gradient / fm.gradient - 1.0), 0.10, "<=")
    m = sec["metrics"]
    reg.log("Flat oil-water contact hard to see under a dipping reservoir (Fig. 5)",
            "Q-PSDM sharpens it; anisotropy positions it",
            "zero-offset images of an oil and a brine section, VTI vs isotropic, with and without Q",
            f"VTI places the fluid signal at {m['VTI + Q compensation'][0]:.0f} m, isotropic at "
            f"{m['isotropic (NMO v) + Q compensation'][0]:.0f} m; Q compensation thins it from "
            f"{m['VTI, no Q compensation'][2]:.1f} to {m['VTI + Q compensation'][2]:.1f} m",
            "anisotropy is what matters in 1D; the image improvement needs 2D/3D imaging")
    text = ("# Discrepancy register of `examples/reproduce_paper.py`\n\n"
            "Generated by the script on the synthetic well; do not edit by hand.\n\n"
            + reg.to_markdown())
    (ROOT / "docs" / "discrepancy_register.md").write_text(text)
    print("\nAcceptance criteria:")
    for c in reg.criteria:
        print(f"   {c.name}: {c.value:.3g} {c.unit} ({c.sense} {c.threshold:g}) "
              f"{'pass' if c.passed else 'FAIL'}")
    print(f"\nregister written to docs/discrepancy_register.md ({len(reg.versions)} versions, "
          f"{len(reg.discrepancies)} discrepancies); total {time.time() - t_start:.0f} s")
    figures(zo, sf, sc, sec)


if __name__ == "__main__":
    main()
