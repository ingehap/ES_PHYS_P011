"""Work package 5: eps and delta from a walkaway VSP, and the gamma -> (eps, delta) calibration.

The article inverts the walkaway VSP for eps and delta at six levels between 640 and
860 m (slowness-polarization method of Leaney & Hornby, 2007; source exclusion zone
200 m, angles of about 30-60 degrees at the target), stabilized by sonic Vs and gamma,
and then calibrates the sonic gamma curve with these estimates to get continuous eps
and delta curves, also in the strongly anisotropic overburden.

1. A synthetic walkaway (exact qP rays through the 10 m Backus average of the
   synthetic well; picks with 0.2 ms and polarizations with 1.5 degree noise) and the
   inversion at six levels, five ways: the article's (exact, slowness + polarization,
   30-60 degrees, priors on Vp0 from the ZO-VSP and on Vs0 from the dipole log), without
   polarization, with the weak-anisotropy formula, with all angles, and with the raw
   (dispersed, sonic-scale) dipole Vs0 as prior.
2. The consistency checks: a second azimuth (VTI), and the polarization check against a
   source-side static of 5 ms per km of offset (lateral heterogeneity).
3. The near-angle trade-off: the delta error and the eps-delta correlation against the
   smallest angle used.
4. The calibration: sonic gamma (WP4 processing) upscaled to the 40 m sub-array
   (Backus), a Bayesian straight line for eps and for delta with leave-one-out
   predictions, and the predicted curves against the seismic-scale truth inside and
   outside the calibrated interval.  Also: what happens when gamma is also used as a
   prior in the inversion (circularity).

Writes ``docs/figures/wp5_*.png`` and prints the numbers quoted in the README.

    python examples/wp5_walkaway.py
"""

import dataclasses
from pathlib import Path

import numpy as np

from iwt import sonic as S
from iwt.calibration import bayes_linear, upscale_gamma
from iwt.plotting import unit_tops
from iwt.qmodel import phase_velocity
from iwt.synthetic import caliper_log, stoneley_log, wisting_like_well
from iwt.vsp import walkaway as W

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
OFFSETS = np.concatenate([-np.arange(2500.0, 199.0, -25.0), np.arange(200.0, 2501.0, 25.0)])
DEPTHS = np.arange(600.0, 901.0, 10.0)
LEVELS = (650.0, 690.0, 720.0, 770.0, 810.0, 850.0)
SUBARRAY = 40.0                       # 5 receivers at 10 m
SIGMA_T = 0.2e-3
SIGMA_PSI = np.radians(1.5)
ANGLES = (30.0, 60.0)
STATIC = 5e-6                         # s per m of signed offset
PRIOR_SD = (0.01, 0.05)               # relative errors of the Vp0 (ZO-VSP) and Vs0 (dipole) priors
ZONES = {"overburden 470-640 m": (470.0, 640.0), "calibrated 640-860 m": (640.0, 860.0),
         "below 860-1670 m": (860.0, 1670.0)}


def window(z, level):
    return (z >= level - SUBARRAY / 2) & (z <= level + SUBARRAY / 2)


def sonic_gamma(model, logs):
    """WP4 processing (step F): tube-wave C66 with tool, caliper, dispersion and a mud
    slowness calibrated in the Stø; dipole C44 matched to the Stoneley resolution."""
    z = logs.depth
    radius, caliper = caliper_log(model, z)
    mud = S.Mud(1540.0, 1250.0)
    tool = S.Borehole(0.108, 0.046)
    st = stoneley_log(model, z, mud, radius, tool.tool_radius, frequency=500.0, aperture=1.8)
    c44 = S.running_average(z, logs.rho * logs.vs ** 2, 1.8, harmonic=True)
    rho = S.running_average(z, logs.rho, 1.8)
    vs = np.sqrt(c44 / rho)
    qs = 0.5 * model.qp[model.layer_index(z)]
    f = logs.meta["f_sonic_s"]
    ref = (z >= 672.0) & (z < 702.0)
    sf = S.calibrate_mud_slowness(logs.vp[ref], vs[ref], rho[ref], st.slowness[ref], 1250.0,
                                  tool, 0.0, 500.0, caliper[ref], shear_frequency=f, qs=qs[ref])
    r = S.sonic_anisotropy(logs.vp, vs, rho, st.slowness, S.Mud(1.0 / sf, 1250.0), tool, 500.0,
                           caliper, shear_frequency=f, qs=qs)
    ok = np.isfinite(r.c66) & np.isfinite(r.c44)
    c44f, c66f = (np.interp(z, z[ok], c[ok]) for c in (r.c44, r.c66))
    return r.gamma, upscale_gamma(z, c44f, c66f, SUBARRAY)


def vs0_priors(model, logs, level):
    """Vs0 prior means from the dipole log over the sub-array: raw (time average at the
    sonic frequency) and corrected (Backus C44, dispersed to the seismic frequency)."""
    z = logs.depth
    w = window(z, level)
    raw = 1.0 / np.mean(1.0 / logs.vs[w])
    c44 = 1.0 / np.mean(1.0 / (logs.rho[w] * logs.vs[w] ** 2))
    vb = np.sqrt(c44 / np.mean(logs.rho[w]))
    qs = 0.5 * model.qp[model.layer_index(level)]
    return raw, float(phase_velocity(vb, qs, model.f_ref, logs.meta["f_sonic_s"]))


def fmt(post, k):
    m, s = post[k]
    return f"{m:+.3f}±{s:.3f}"


def main():
    model, logs = wisting_like_well()
    seabed = model.interfaces[1]
    # the wave-scale model: Backus over the sub-array length (~ a wavelength at 50 Hz)
    seis = model.backus(SUBARRAY, respect=[seabed])
    truth = {lv: seis.at(lv) for lv in LEVELS}
    rng = np.random.default_rng(2016)

    clean = W.synthetic_walkaway(seis, OFFSETS, DEPTHS)
    # two azimuths of a VTI medium have the same kinematics: reuse the rays, new noise
    two = dataclasses.replace(
        clean, azimuths=np.array([0.0, 90.0]),
        time=np.concatenate([clean.time] * 2) + SIGMA_T * rng.standard_normal((2,) + clean.time.shape[1:]),
        psi=np.concatenate([clean.psi] * 2) + SIGMA_PSI * rng.standard_normal((2,) + clean.psi.shape[1:]))
    static = dataclasses.replace(two, time=two.time + STATIC * OFFSETS[None, :, None])

    def measure(survey, level, azimuth=0):
        return W.measure_slowness(survey, level, azimuth=azimuth, time_noise=SIGMA_T,
                                  psi_noise=SIGMA_PSI)

    def angles(d, lo=ANGLES[0], hi=ANGLES[1]):
        th = np.degrees(d.theta)
        return d.select((th >= lo) & (th <= hi))

    print("1. Walkaway inversion at six levels (40 m sub-arrays, 30-60 degrees unless noted)")
    print("   synthetic and truth: 40 m running Backus average of the well model\n")
    res = {}
    for lv in LEVELS:
        t = truth[lv]
        d_all = measure(two, lv)
        d = angles(d_all)
        vs_raw, vs_cor = vs0_priors(model, logs, lv)
        vp_prior = t["vp0"] * (1.0 + PRIOR_SD[0] * rng.standard_normal())
        prior = {"vp0": (vp_prior, PRIOR_SD[0] * t["vp0"]), "vs0": (vs_cor, PRIOR_SD[1] * vs_cor)}
        prior_raw = {"vp0": prior["vp0"], "vs0": (vs_raw, PRIOR_SD[1] * vs_raw)}
        runs = {
            "A exact, slowness + polarization": W.invert_level(d, prior),
            "B exact, slowness only": W.invert_level(d, prior, use_polarization=False),
            "C weak anisotropy (Thomsen)": W.invert_level(d, prior, method="weak"),
            "D as A, all angles": W.invert_level(d_all, prior),
            "E as A, raw dipole Vs0 prior": W.invert_level(d, prior_raw),
        }
        noise_free = W.invert_level(angles(measure(clean, lv)), prior)
        az = W.azimuthal_consistency([runs["A exact, slowness + polarization"],
                                      W.invert_level(angles(measure(two, lv, 1)), prior)])
        pol = W.polarization_check(d, runs["B exact, slowness only"])
        d_st = angles(measure(static, lv))
        b_st = W.invert_level(d_st, prior, use_polarization=False)
        pol_st = W.polarization_check(d_st, b_st)
        res[lv] = dict(runs=runs, prior=prior, d=d, d_all=d_all, noise_free=noise_free,
                       static=b_st, pol=pol, pol_static=pol_st, az=az)
        name = seis.label[seis.layer_index(lv)]
        print(f"  {lv:.0f} m ({name}, {d.p.size} sources in 30-60 degrees, "
              f"{np.degrees(d_all.theta.min()):.0f}-{np.degrees(d_all.theta.max()):.0f} available): "
              f"true eps {t['eps']:+.3f}, delta {t['delta']:+.3f}, gamma {t['gamma']:.3f}; "
              f"Vs0 true {t['vs0']:.0f}, dipole raw {vs_raw:.0f}, corrected {vs_cor:.0f} m/s")
        print(f"    {'noise-free data, as A':34s} eps {fmt(noise_free, 'eps')}  delta "
              f"{fmt(noise_free, 'delta')}")
        for k, p in runs.items():
            print(f"    {k:34s} eps {fmt(p, 'eps')}  delta {fmt(p, 'delta')}  "
                  f"z ({(p['eps'][0] - t['eps']) / p['eps'][1]:+.1f}, "
                  f"{(p['delta'][0] - t['delta']) / p['delta'][1]:+.1f})  "
                  f"data weight Vp0 {p.data_weight[0]:.2f} Vs0 {p.data_weight[1]:.2f}  "
                  f"chi2/n {p.chi2 / p.n_data:.2f}")
        print(f"    checks: azimuth p = {az[2]:.2f}; polarization check chi2/n {pol[0]:.2f} "
              f"(p = {pol[1]:.2f}); with a {STATIC * 1e6:.0f} ms/km source static: eps "
              f"{fmt(b_st, 'eps')} delta {fmt(b_st, 'delta')}, polarization check chi2/n "
              f"{pol_st[0]:.2f} (p = {pol_st[1]:.1e})")

    print("\n   Summary over the six levels (estimate - truth) / std:")
    for k in res[LEVELS[0]]["runs"]:
        z = np.array([[(res[lv]["runs"][k][n][0] - truth[lv][n]) / res[lv]["runs"][k][n][1]
                       for n in ("eps", "delta")] for lv in LEVELS])
        print(f"    {k:34s} eps mean {z[:, 0].mean():+.2f} rms {np.sqrt((z[:, 0] ** 2).mean()):.2f}; "
              f"delta mean {z[:, 1].mean():+.2f} rms {np.sqrt((z[:, 1] ** 2).mean()):.2f}")

    print("\n2. Near-angle trade-off at 770 m (slowness + polarization, priors as A, angles up to 60):")
    d770, pr770 = res[770.0]["d_all"], res[770.0]["prior"]
    for lo in (20.0, 25.0, 30.0, 35.0, 40.0):
        p = W.invert_level(angles(d770, lo, 60.0), pr770)
        print(f"    from {lo:2.0f} deg: delta std {p['delta'][1]:.4f}, corr(eps, delta) "
              f"{p.corr[2, 3]:+.2f}, corr(Vp0, delta) {p.corr[0, 3]:+.2f}, Vp0 data weight "
              f"{p.data_weight[0]:.2f}")

    print("\n3. Heterogeneity within the sub-array: noise-free data from the 10 m Backus model "
          "(ray theory: q averages the sub-array, the polarization is local), as A")
    fine = W.synthetic_walkaway(model.backus(10.0, respect=[seabed]), OFFSETS, DEPTHS)
    for lv in LEVELS:
        d = angles(measure(fine, lv))
        pa = W.invert_level(d, res[lv]["prior"])
        pb = W.invert_level(d, res[lv]["prior"], use_polarization=False)
        t = truth[lv]
        print(f"  {lv:.0f} m: eps {pa['eps'][0]:+.3f} (slowness only {pb['eps'][0]:+.3f}), "
              f"delta {pa['delta'][0]:+.3f} ({pb['delta'][0]:+.3f}); truth {t['eps']:+.3f}, "
              f"{t['delta']:+.3f}; chi2/n {pa.chi2 / pa.n_data:.2f}; polarization check p "
              f"{W.polarization_check(d, pb)[1]:.1e}")

    print("\n4. Calibration of the sonic gamma with the walkaway eps and delta (method A)")
    z = logs.depth
    g_sonic, g_up = sonic_gamma(model, logs)
    true_up = seis.at(z)
    gx = np.array([np.interp(lv, z, g_up) for lv in LEVELS])
    cals = {}
    for n in ("eps", "delta"):
        y = np.array([res[lv]["runs"]["A exact, slowness + polarization"][n][0] for lv in LEVELS])
        s = np.array([res[lv]["runs"]["A exact, slowness + polarization"][n][1] for lv in LEVELS])
        cal = bayes_linear(gx, y, s, prior_mean=(0.0, 0.5), prior_std=(0.1, 2.0))
        cals[n] = cal
        sd = np.sqrt(np.diag(cal.coef_cov))
        print(f"  {n:5s} = {cal.coef[0]:+.3f}(±{sd[0]:.3f}) + {cal.coef[1]:.2f}(±{sd[1]:.2f}) gamma, "
              f"extra scatter {cal.scatter:.3f}; leave-one-out z "
              + " ".join(f"{v:+.1f}" for v in cal.loo_z))
    print(f"  sonic gamma upscaled to {SUBARRAY:.0f} m at the levels: "
          + " ".join(f"{v:.3f}" for v in gx) + "; true: "
          + " ".join(f"{truth[lv]['gamma']:.3f}" for lv in LEVELS))
    print("  Predicted curves against the truth (40 m Backus): RMS error, mean predictive std, "
          "share of the truth inside 2 std")
    pred = {}
    for n, cal in cals.items():
        m, s = cal.predict(g_up)
        pred[n] = (m, s)
        for zone, (a, b) in ZONES.items():
            w = (z >= a) & (z < b) & np.isfinite(m)
            e = m[w] - true_up[n][w]
            print(f"    {n:5s} {zone:22s} RMS {np.sqrt(np.mean(e ** 2)):.3f}, std "
                  f"{np.mean(s[w]):.3f}, inside 2 std {np.mean(np.abs(e) < 2 * s[w]):.0%}")
    # circularity: gamma also used as the eps prior in the inversion
    print("  Circularity: eps prior N(gamma_sonic, 0.05) in the inversion as well")
    y_c, s_c, dw = [], [], []
    for lv, g in zip(LEVELS, gx):
        r = res[lv]
        p = W.invert_level(r["d"], dict(r["prior"], eps=(g, 0.05)))
        y_c.append(p["eps"][0])
        s_c.append(p["eps"][1])
        dw.append(p.data_weight[2])
    cal_c = bayes_linear(gx, np.array(y_c), np.array(s_c), prior_mean=(0.0, 0.5),
                         prior_std=(0.1, 2.0))
    print("    data weight of eps per level: " + " ".join(f"{v:.2f}" for v in dw)
          + f"; calibrated eps slope {cal_c.coef[1]:.2f} (without the gamma prior "
          f"{cals['eps'].coef[1]:.2f}), predictive std at gamma 0.3: "
          f"{cal_c.predict(0.3)[1]:.3f} vs {cals['eps'].predict(0.3)[1]:.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from iwt.plotting import PALETTE, depth_profiles
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    import matplotlib.pyplot as plt

    from iwt.raytrace import qp_phase_velocity, qp_polarization, velocity_stiffness
    from iwt.vsp.walkaway import PARAMETERS

    FIG.mkdir(parents=True, exist_ok=True)
    col = PALETTE["series"]

    # figure 1: data and fits at 770 m, and the eps-delta posteriors
    lv = 770.0
    r = res[lv]
    d = r["d_all"]
    th = np.linspace(0.0, np.radians(85.0), 200)
    t = truth[lv]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw=dict(width_ratios=(1, 1, 1.35)))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax = axes[0]
    ax.errorbar(np.degrees(d.theta), d.velocity, yerr=d.velocity ** 2 * np.hypot(
        d.sigma_p * np.sin(d.theta), d.sigma_q * np.cos(d.theta)), fmt=".", ms=3,
        color=PALETTE["muted"], elinewidth=0.5, label="measured (p, q)")
    a_true = velocity_stiffness(t["vp0"], t["vs0"], t["eps"], t["delta"])
    ax.plot(np.degrees(th), qp_phase_velocity(th, *a_true), color=PALETTE["ink"], lw=1.2,
            label="truth (40 m Backus)")
    for k, (name, c, ls) in enumerate((("A exact, slowness + polarization", col[0], "-"),
                                       ("C weak anisotropy (Thomsen)", col[1], "--"))):
        x = r["runs"][name].mean
        v = (qp_phase_velocity(th, *velocity_stiffness(*x)) if "exact" in name else
             x[0] * (1 + x[3] * np.sin(th) ** 2 * np.cos(th) ** 2 + x[2] * np.sin(th) ** 4))
        ax.plot(np.degrees(th), v, color=c, ls=ls, lw=1.2, label=name.split(" ", 1)[1])
    ax.axvspan(*ANGLES, color=PALETTE["grid"], alpha=0.6, zorder=0, label="article's window")
    ax.set_xlabel("phase angle (deg)")
    ax.set_ylabel("phase velocity (m/s)")
    ax.set_title(f"qP phase velocity at {lv:.0f} m", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.errorbar(np.degrees(d.theta), np.degrees(d.psi - d.theta), yerr=np.degrees(d.sigma_psi),
                fmt=".", ms=3, color=PALETTE["muted"], elinewidth=0.5, label="measured")
    for x, lab, c, ls in ((np.array([t["vp0"], t["vs0"], t["eps"], t["delta"]]), "truth",
                           PALETTE["ink"], "-"),
                          (r["runs"]["A exact, slowness + polarization"].mean, "A", col[0], "-"),
                          (r["runs"]["E as A, raw dipole Vs0 prior"].mean,
                           "E (raw dipole Vs0 prior)", col[2], "--")):
        a = velocity_stiffness(*x)
        v = qp_phase_velocity(th, *a)
        ax.plot(np.degrees(th), np.degrees(qp_polarization(np.sin(th) / v, np.cos(th) / v, *a) - th),
                color=c, ls=ls, lw=1.2, label=lab)
    ax.axvspan(*ANGLES, color=PALETTE["grid"], alpha=0.6, zorder=0)
    ax.set_xlabel("phase angle (deg)")
    ax.set_ylabel("polarization - phase angle (deg)")
    ax.set_title("qP polarization", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    i, j = PARAMETERS.index("eps"), PARAMETERS.index("delta")
    phi = np.linspace(0, 2 * np.pi, 100)
    for k, (name, c) in enumerate(zip(r["runs"], list(col) + [PALETTE["ink2"]])):
        p = r["runs"][name]
        cov = p.cov[np.ix_([i, j], [i, j])]
        w, v = np.linalg.eigh(cov)
        e = (v * np.sqrt(w * 5.991)) @ np.stack([np.cos(phi), np.sin(phi)])   # 95 %
        ax.plot(p.mean[i] + e[0], p.mean[j] + e[1], color=c, lw=1.0,
                ls=("-", "--", "-.", ":", "-")[k], label=name)
        ax.plot(p.mean[i], p.mean[j], "o", ms=3, color=c)
    ax.plot(t["eps"], t["delta"], "k*", ms=9, label="truth")
    ax.set_xlabel("eps")
    ax.set_ylabel("delta")
    ax.set_title("95 % posterior regions", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(color=PALETTE["grid"], lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "wp5_walkaway.png", dpi=150)

    # figure 2: gamma, eps and delta curves with the walkaway estimates and the calibration
    tops = [(tt, n) for tt, n in unit_tops(model)[1:] if tt > 440.0]
    fig = depth_profiles({
        "gamma": ("gamma", {
            f"true, {SUBARRAY:.0f} m Backus": (true_up["gamma"], z),
            f"sonic (WP4), upscaled to {SUBARRAY:.0f} m": (g_up, z)}),
        "eps": ("eps", {
            f"true, {SUBARRAY:.0f} m Backus": (true_up["eps"], z),
            "calibrated from sonic gamma": (pred["eps"][0], z),
            "± 2 std (predictive)": (pred["eps"][0] + 2 * pred["eps"][1], z)}),
        "delta": ("delta", {
            f"true, {SUBARRAY:.0f} m Backus": (true_up["delta"], z),
            "calibrated from sonic gamma": (pred["delta"][0], z),
            "± 2 std (predictive)": (pred["delta"][0] + 2 * pred["delta"][1], z)}),
    }, top=460.0, bottom=1000.0, tops=tops, title="Walkaway eps and delta (method A, 2 std) "
       "and the calibrated sonic gamma", width=3.4)
    for ax, n in zip(fig.axes[1:3], ("eps", "delta")):
        m, s = pred[n]
        ax.plot(m - 2 * s, z, color=col[2], ls=":", lw=1.3)
        e = np.array([res[lv]["runs"]["A exact, slowness + polarization"][n] for lv in LEVELS])
        ax.errorbar(e[:, 0], LEVELS, xerr=2 * e[:, 1], yerr=SUBARRAY / 2, fmt="o", ms=4,
                    color=PALETTE["ink"], elinewidth=0.9, capsize=2, zorder=5)
        ax.axhspan(640.0, 860.0, color=PALETTE["grid"], alpha=0.5, zorder=0)
    fig.axes[0].set_xlim(-0.05, 0.45)
    fig.axes[1].set_xlim(-0.05, 0.4)
    fig.axes[2].set_xlim(-0.15, 0.3)
    fig.savefig(FIG / "wp5_calibration.png", dpi=150)
    print("\nfigures written to", FIG)


if __name__ == "__main__":
    main()
