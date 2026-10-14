"""Work package 6: ray tracing and Q-compensated depth imaging, and the Fig. 3 experiment.

The article (Fig. 3) compares the reservoir AVA after three depth migrations of the
3D data with the AVA of elastic synthetics: isotropic PSDM with Q compensation and
anisotropic PSDM without it are "OK", anisotropic PSDM with Q compensation is the
best.  This example repeats the experiment on the synthetic well:

1. Kinematics at the Stø top: the exact layered-VTI reflection times against the
   Alkhalifah & Tsvankin (1995) nonhyperbolic moveout, and the time and angle errors
   of two isotropic velocity models (the vertical Vp0, which ties the well, and the
   NMO velocity Vp0 sqrt(1 + 2 delta), which flattens near-offset gathers).
2. Data: point-source CMP gathers (explosion at 6 m, hydrophones at 8 m) from the
   reflectivity method, Born primaries without free surface, with the model's Q
   ("observed") and without Q ("elastic synthetic", the reference).  The seabed
   reflection is muted.  It crosses the Stø reflection at about 1150 m of offset
   (54 degrees); the AVA uses offsets up to 850 m (44 degrees), where the two are
   at least 85 ms apart.
3. Migrations: VTI and isotropic models (20 m Backus averages of the well model),
   with and without ray-based Q compensation (Cavalca et al., 2013), plus phase-only
   compensation and a Q model 30 % too low.  Each is scored on the Stø-top AVA
   against the reference: intercept and gradient over 20-44 degrees and the RMS
   misfit of the AVA curve.
4. The gain limit: AVA misfit against the gain limit for data with added noise.

Writes ``docs/figures/wp6_*.png`` and prints the numbers quoted in the README.

    python examples/wp6_imaging.py
"""

import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from iwt import avo, paper
from iwt import imaging as I
from iwt.model import ElasticModel1D
from iwt.modeling.convolution import ricker
from iwt.modeling.reflection import exact_pp
from iwt.modeling.reflectivity import Reflectivity
from iwt.raytrace import reflection_table
from iwt.synthetic import wisting_like_well

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
OFFSETS = np.arange(100.0, 1501.0, 50.0)
DT, NT = 0.001, 1250
Z_S, Z_R = 6.0, 8.0
F_MAX = 100.0
HIGH_CUT, GAIN = 80.0, 10.0
FIT = (20.0, 44.0)
X_AVA = 850.0                     # offsets whose Stø reflection is clear of the seabed reflection
Z_STO = paper.STO_TOP_TVDSS


def born_gathers(model, workers=4):
    """Born primaries, no free surface, no converted waves: with and without Q."""
    m = model.subset(0.0, 730.0)
    out = {}
    for name, att in (("observed", True), ("elastic", False)):
        t0 = time.time()
        r = Reflectivity(m, attenuation=att, free_surface=False, internal_multiples=False,
                         converted_waves=False, transmission_loss=False)
        out[name] = r.record(OFFSETS, Z_S, [Z_R], DT, NT, ricker(30.0, DT), "pressure",
                             f_max=F_MAX, workers=workers).data[0]
        print(f"  {name} gather: {time.time() - t0:.0f} s")
    return out


def seabed_mute(data, model, offsets=OFFSETS, after=0.035, ramp=0.01):
    """Zero everything before the seabed reflection plus ``after`` (s), with a cosine ramp."""
    t_sb = reflection_table(model, [model.interfaces[1] - 0.01], offsets, Z_S, Z_R).time[0]
    t = np.arange(data.shape[1]) * DT
    w = np.clip((t[None, :] - (t_sb[:, None] + after)) / ramp, 0.0, 1.0)
    return data * (0.5 - 0.5 * np.cos(np.pi * w))


def isotropic(m: ElasticModel1D, nmo: bool) -> ElasticModel1D:
    iso = m.copy()
    if nmo:
        iso.vp0 = m.vp0 * np.sqrt(1.0 + 2.0 * m.delta)
    iso.eps[:] = 0.0
    iso.delta[:] = 0.0
    return iso


def amplitude_scale(upper, lower):
    """Migrated amplitude per unit reflection coefficient for this source, wavelet and
    processing: one interface below the sea, no Q (the true-amplitude check)."""
    m = ElasticModel1D([0.0, 417.5, Z_STO, 900.0], vp0=[1480.0, upper[0], lower[0]],
                       vs0=[0.0, upper[1], lower[1]], rho=[1025.0, upper[2], lower[2]],
                       eps=[0.0, upper[3], lower[3]], delta=[0.0, upper[4], lower[4]])
    x = OFFSETS[OFFSETS <= X_AVA]
    rec = Reflectivity(m, attenuation=False, free_surface=False, internal_multiples=False,
                       converted_waves=False, transmission_loss=False).record(
        x, Z_S, [Z_R], DT, NT, ricker(30.0, DT), "pressure", f_max=F_MAX, workers=4)
    z = np.arange(Z_STO - 30.0, Z_STO + 30.0, 0.5)
    img = I.migrate_cmp(seabed_mute(rec.data[0], m, x), DT, x, m, z, Z_S, Z_R, high_cut=HIGH_CUT)
    pk = avo.pick_event(z, img.data, img.theta, Z_STO, 6.0)
    theta = img.theta[np.searchsorted(z, Z_STO) - 1]       # incidence angle, just above
    return pk.amplitude / exact_pp(theta, upper, lower).real, theta


def main():
    model, _ = wisting_like_well()
    seabed = model.interfaces[1]
    vti = model.backus(20.0, respect=[seabed])
    iso_nmo, iso_vert = isotropic(vti, True), isotropic(vti, False)
    low_q = vti.copy()
    low_q.qp = 0.7 * vti.qp

    print("1. Kinematics of the Stø-top reflection (source 6 m, receiver 8 m)")
    xs = np.arange(0.0, 1501.0, 25.0)
    tab = reflection_table(vti, [Z_STO], xs, Z_S, Z_R)
    T, ok = tab.time[0], np.isfinite(tab.time[0])
    t0 = T[0]

    def at(par, x):
        v, eta = par
        return np.sqrt(t0 ** 2 + x ** 2 / v ** 2
                       - 2 * eta * x ** 4 / (v ** 2 * (t0 ** 2 * v ** 2 + (1 + 2 * eta) * x ** 2)))

    for ratio in (1.0, 1.5, 2.0):
        m = ok & (xs <= ratio * Z_STO)
        sol = least_squares(lambda p: at(p, xs[m]) - T[m], [2000.0, 0.1])
        print(f"  Alkhalifah-Tsvankin fit to x/z = {ratio:.1f}: Vnmo {sol.x[0]:.0f} m/s, "
              f"eta {sol.x[1]:.3f}, largest time error {1e3 * np.abs(sol.fun).max():.2f} ms")
    for name, mm in (("isotropic, NMO velocity", iso_nmo), ("isotropic, vertical Vp0", iso_vert)):
        ti = reflection_table(mm, [float(mm.depth_at_time(vti.vertical_time(Z_STO)))], xs, Z_S, Z_R)
        rows = []
        for ang in (20.0, 30.0, 45.0):
            j = int(np.nanargmin(np.abs(np.degrees(tab.theta[0]) - ang)))
            rows.append(f"{ang:.0f} deg (x {xs[j]:.0f} m): time {1e3 * (ti.time[0, j] - T[j]):+.1f} ms, "
                        f"angle {np.degrees(ti.theta[0, j] - tab.theta[0, j]):+.1f} deg")
        z0 = float(mm.depth_at_time(vti.vertical_time(Z_STO)))
        print(f"  {name}: Stø imaged at {z0:.1f} m (true {Z_STO:.0f}); " + "; ".join(rows))

    print("\n2. Born CMP gathers by the reflectivity method (30 Hz Ricker, f <= 100 Hz)")
    raw = born_gathers(model)
    data = {k: seabed_mute(v, vti) for k, v in raw.items()}
    upper, lower = model.at(Z_STO - 0.25), model.at(Z_STO + 0.25)
    up = tuple(float(upper[k]) for k in ("vp0", "vs0", "rho", "eps", "delta"))
    lo = tuple(float(lower[k]) for k in ("vp0", "vs0", "rho", "eps", "delta"))
    ratio, th = amplitude_scale(up, lo)
    sel = (np.degrees(th) >= FIT[0]) & (np.degrees(th) <= FIT[1])
    scale = float(np.median(ratio[sel]))
    print(f"  true-amplitude check (single Fuglen/Stø interface, no Q): migrated amplitude / "
          f"exact R constant to {np.ptp(ratio[sel]) / scale:.1%} over {FIT[0]:.0f}-{FIT[1]:.0f} deg")

    print(f"\n3. Migrations scored on the Stø-top AVA, offsets <= {X_AVA:.0f} m "
          f"(gain limit {GAIN:.0f}, high cut {HIGH_CUT:.0f} Hz)")
    variants = [
        ("reference: elastic data, VTI", "elastic", vti, None),
        ("VTI, no Q compensation", "observed", vti, None),
        ("VTI + Q compensation", "observed", vti, "amplitude"),
        ("VTI + phase-only Q", "observed", vti, "phase"),
        ("VTI + Q compensation, Q 30 % low", "observed", low_q, "amplitude"),
        ("isotropic (NMO v), no Q compensation", "observed", iso_nmo, None),
        ("isotropic (NMO v) + Q compensation", "observed", iso_nmo, "amplitude"),
        ("isotropic (vertical Vp0) + Q compensation", "observed", iso_vert, "amplitude"),
    ]
    sel = OFFSETS <= X_AVA
    res = {}
    t_sto = vti.vertical_time(Z_STO)
    for name, d, mm, qc in variants:
        z0 = float(mm.depth_at_time(t_sto))
        z = np.arange(z0 - 70.0, z0 + 50.0, 0.5)
        full = I.migrate_cmp(data[d], DT, OFFSETS, mm, z, Z_S, Z_R, q_compensation=qc,
                             gain_limit=GAIN, high_cut=HIGH_CUT)
        pk = avo.pick_event(z, full.data[:, sel], full.theta[:, sel], z0, 6.0)
        res[name] = (full, pk, z0)
    ref = res[variants[0][0]][1]
    fr = avo.intercept_gradient(ref.theta, ref.amplitude, FIT)
    grid = np.radians(np.arange(FIT[0], FIT[1] + 0.01, 1.0))
    a_ref = np.interp(grid, ref.theta, ref.amplitude)
    print(f"  {'variant':42s} {'Stø depth':>9s} {'moveout':>8s} {'A/A_ref':>8s} {'B/B_ref':>8s} {'misfit':>7s}")
    score = {}
    for name, (full, pk, z0) in res.items():
        fit = avo.intercept_gradient(pk.theta, pk.amplitude, FIT)
        okp = np.isfinite(pk.theta) & np.isfinite(pk.amplitude)
        a = np.interp(grid, pk.theta[okp], pk.amplitude[okp], left=np.nan, right=np.nan)
        mis = np.sqrt(np.nanmean((a - a_ref) ** 2) / np.mean(a_ref ** 2))
        j20, j45 = (int(np.nanargmin(np.abs(np.degrees(pk.theta) - v))) for v in FIT)
        score[name] = (fit, mis)
        print(f"  {name:42s} {pk.depth[0]:9.1f} {pk.depth[j45] - pk.depth[j20]:+8.1f} "
              f"{fit.intercept / fr.intercept:8.2f} {fit.gradient / fr.gradient:8.2f} {mis:7.3f}")
    r_int = exact_pp(grid, up, lo).real * scale
    fit_int = avo.intercept_gradient(grid, r_int, FIT)
    print(f"  single-interface AVA (exact R of Fuglen over Stø, same scale): A/A_ref "
          f"{fit_int.intercept / fr.intercept:.2f}, B/B_ref {fit_int.gradient / fr.gradient:.2f}: "
          f"the Stø-top event is tuned with the stringers 11 and 24 m above it")

    print("\n4. Gain limit against noise (VTI + Q compensation; misfit to the reference AVA)")
    rng = np.random.default_rng(1)
    d = data["observed"][sel]
    f = np.fft.rfftfreq(NT, DT)
    win = (np.arange(NT) * DT > 0.7) & (np.arange(NT) * DT < 0.95)
    z = np.arange(Z_STO - 30.0, Z_STO + 30.0, 0.5)
    tab = reflection_table(vti, z, OFFSETS[sel], Z_S, Z_R)
    gains = (2.0, 3.0, 5.0, 10.0, 30.0, 100.0)
    print("  noise / signal RMS   gain limit: " + "  ".join(f"{g:>5.0f}" for g in gains))
    gain_rows = {}
    for level in (0.0, 0.05, 0.2):
        noise = rng.standard_normal(d.shape)
        spec = np.fft.rfft(noise)
        spec[..., f > F_MAX] = 0.0
        noise = np.fft.irfft(spec, n=NT)
        noise *= level * np.sqrt(np.mean(d[:, win] ** 2)) / noise.std()
        row = []
        for g in gains:
            img = I.migrate_cmp(d + noise, DT, OFFSETS[sel], vti, z, Z_S, Z_R,
                                q_compensation="amplitude", gain_limit=g, high_cut=HIGH_CUT,
                                table=tab)
            pk = avo.pick_event(z, img.data, img.theta, Z_STO - 1.8, 6.0)
            m = pk.theta >= np.radians(FIT[0])
            row.append(np.sqrt(np.mean((pk.amplitude[m] - np.interp(pk.theta[m], ref.theta,
                                                                    ref.amplitude)) ** 2)
                               / np.mean(ref.amplitude[ref.theta >= np.radians(FIT[0])] ** 2)))
        gain_rows[level] = row
        print(f"  {level:18.0%}                " + "  ".join(f"{v:5.3f}" for v in row))

    try:
        import matplotlib

        matplotlib.use("Agg")
        from iwt.plotting import PALETTE, gather_panels
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    show = ("reference: elastic data, VTI", "VTI, no Q compensation", "VTI + Q compensation",
            "isotropic (NMO v) + Q compensation")
    zc = np.arange(600.0, 720.0, 0.5)
    panels = {}
    for name in show:
        full, _, z0 = res[name]
        panels[name.replace(" compensation", "")] = np.array(
            [np.interp(zc, full.depth, full.data[:, i]) for i in range(OFFSETS.size)])
    fig = gather_panels(zc, OFFSETS, panels, ylabel="depth (m)",
                        title="Migrated CMP gathers (true amplitude), Stø top at 672 m",
                        marks={"Stø top": Z_STO, "Fuglen": 631.0})
    fig.savefig(FIG / "wp6_gathers.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax = axes[0]
    styles = {"reference: elastic data, VTI": (PALETTE["ink"], "-", 2.0),
              "VTI + Q compensation": (PALETTE["series"][0], "-", 1.3),
              "VTI, no Q compensation": (PALETTE["series"][1], "--", 1.3),
              "isotropic (NMO v) + Q compensation": (PALETTE["series"][2], "-.", 1.3),
              "VTI + Q compensation, Q 30 % low": (PALETTE["series"][3], ":", 1.3)}
    for name, (c, ls, lw) in styles.items():
        pk = res[name][1]
        ax.plot(np.degrees(pk.theta), pk.amplitude / scale, color=c, ls=ls, lw=lw,
                marker="o", ms=2.5, label=name)
    ax.plot(np.degrees(grid), r_int / scale, color=PALETTE["muted"], lw=1.0,
            label="exact R, single Fuglen/Stø interface")
    ax.axvspan(*FIT, color=PALETTE["grid"], alpha=0.5, zorder=0)
    ax.set_xlabel("reflection angle (deg)")
    ax.set_ylabel("Stø-top amplitude / R scale")
    ax.set_title("Stø-top AVA after migration (cf. Fig. 3)", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax = axes[1]
    for k, (level, row) in enumerate(gain_rows.items()):
        ax.semilogx(gains, row, color=PALETTE["series"][k], marker="o", ms=3,
                    label=f"noise {level:.0%} of signal RMS")
    ax.set_xlabel("gain limit")
    ax.set_ylabel("AVA misfit (relative RMS)")
    ax.set_title(f"Gain limit (VTI + Q compensation, {HIGH_CUT:.0f} Hz high cut)", fontsize=9,
                 loc="left")
    ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(color=PALETTE["grid"], lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "wp6_ava.png", dpi=150)
    print("\nfigures written to", FIG)


if __name__ == "__main__":
    main()
