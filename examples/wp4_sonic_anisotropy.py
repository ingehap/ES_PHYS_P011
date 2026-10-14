"""Work package 4: Thomsen's gamma from the sonic logs of the vertical synthetic well.

C44 comes from the dipole shear log, C66 from the Stoneley log through the tube-wave
relation of Norris (1990), and gamma = (C66 - C44) / (2 C44) (Walsh et al., 2007).

1. Stoneley dispersion in the units of the well: the exact borehole mode in the TI
   formation, against the two shortcuts (eps = delta = 0, and an isotropic formation
   with shear modulus C66).  The slow overburden shales make the Stoneley wave leaky.
2. A processing ladder, each step adding one correction, against the true
   sonic-scale gamma: tool term, caliper, dispersion, the dipole's frequency and
   resolution, and the mud slowness (nominal, calibrated in the Stø, or true).
3. The gamma uncertainty budget by linear propagation, checked by Monte Carlo.

The Stoneley log is "measured" at 500 Hz with a rigid 46 mm tool in mud of
1540 m/s and 1250 kg/m^3; processing assumes 1500 m/s unless calibrated.

Writes ``docs/figures/wp4_*.png`` and prints the numbers quoted in the README.

    python examples/wp4_sonic_anisotropy.py
"""

from pathlib import Path

import numpy as np

from iwt import sonic as S
from iwt.model import thomsen_to_cij
from iwt.plotting import unit_tops
from iwt.synthetic import caliper_log, stoneley_log, wisting_like_well

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
MUD_TRUE = S.Mud(1540.0, 1250.0)
MUD_NOMINAL = S.Mud(1500.0, 1250.0)
BIT = 0.108
TOOL = S.Borehole(BIT, 0.046)
F_ST = 500.0
APERTURE = 1.8
UNITS = [(470.0, 600.0, "Lower Cretaceous shale"), (600.0, 631.0, "Hekkingen"),
         (631.0, 672.0, "Fuglen"), (672.0, 702.0, "Stø"), (702.0, 740.0, "Nordmela"),
         (740.0, 800.0, "Fruholmen"), (800.0, 1670.0, "Snadd")]
SIGMA = {  # 1-sigma input errors for the budget
    "s_stoneley": 0.003, "s_shear": 0.01, "rho": 15.0, "mud_slowness": 0.02,
    "mud_density": 0.01, "radius": 0.001, "tool_radius": 0.001,
}


def per_unit(z, x, fn=np.nanmean):
    return np.array([fn(x[(z >= a) & (z < b)]) for a, b, _ in UNITS])


def dispersion_curves(model):
    """Stoneley dispersion for unit averages: exact TI, eps = delta = 0, equivalent isotropic."""
    f = np.geomspace(50.0, 3000.0, 40)
    blk = model.block([z for z, _ in unit_tops(model)[1:]] + [694.5])
    out = {}
    print(f"1. Stoneley dispersion (unit Backus averages, rigid tool, mud {MUD_TRUE.velocity:.0f} m/s):"
          " slowness relative to the tube wave at 500 Hz / 1 kHz, and the radiation 1/Q at 500 Hz")
    for name in ("Lower Cretaceous shale", "Hekkingen", "Stø", "Fruholmen"):
        i = list(blk.label).index(name if name != "Stø" else "Stø (oil)")
        vp0, vs0, rho = blk.vp0[i], blk.vs0[i], blk.rho[i]
        c = thomsen_to_cij(vp0, vs0, rho, blk.eps[i], blk.delta[i], blk.gamma[i])
        cij = (c.c11, c.c13, c.c33, c.c44, c.c66)
        st = S.tube_wave_slowness(c.c66, MUD_TRUE.slowness, MUD_TRUE.density, BIT, TOOL.tool_radius)
        args = (MUD_TRUE.slowness, MUD_TRUE.density, BIT, TOOL.tool_radius)
        exact = S.stoneley_slowness(f, cij, rho, *args, full=True)
        ti0 = S.stoneley_slowness(f, S.ti_stiffness(vp0, vs0, rho, c.c66), rho, *args)
        iso = S.stoneley_slowness(f, S.ti_stiffness(vp0, np.sqrt(c.c66 / rho), rho, c.c66), rho, *args)
        out[name] = dict(f=f, exact=exact.real / st, ti0=ti0 / st, iso=iso / st,
                         inv_q=2 * exact.imag / exact.real)
        k500, k1k = np.argmin(abs(f - 500)), np.argmin(abs(f - 1000))
        leaky = "leaky" if st < 1.0 / vs0 else "trapped"
        print(f"  {name:24s} gamma {blk.gamma[i]:.2f}, {leaky:7s}: exact {exact.real[k500] / st:.4f} / "
              f"{exact.real[k1k] / st:.4f}; eps=delta=0 {ti0[k500] / st:.4f}; equivalent isotropic "
              f"{iso[k500] / st:.4f}; 1/Q {2 * exact.imag[k500] / exact.real[k500]:.4f}")
    return out


def main():
    model, logs = wisting_like_well()
    z = logs.depth
    radius, caliper = caliper_log(model, z)
    st = stoneley_log(model, z, MUD_TRUE, radius, TOOL.tool_radius, frequency=F_ST,
                      aperture=APERTURE)
    truth = model.backus(APERTURE, respect=[model.interfaces[1]]).at(z)["gamma"]
    curves = dispersion_curves(model)

    # resolution matching and the dipole's frequency (Qs = Qp / 2 assumed)
    c44_m = S.running_average(z, logs.rho * logs.vs ** 2, APERTURE, harmonic=True)
    rho_m = S.running_average(z, logs.rho, APERTURE)
    vs_m = np.sqrt(c44_m / rho_m)
    qs = 0.5 * model.qp[model.layer_index(z)]
    f_dip = logs.meta["f_sonic_s"]
    base = (logs.vp, logs.vs, logs.rho, st.slowness)
    matched = (logs.vp, vs_m, rho_m, st.slowness)
    ref = (z >= 672.0) & (z < 702.0)                     # the Stø, assumed isotropic
    sf_cal = S.calibrate_mud_slowness(logs.vp[ref], vs_m[ref], rho_m[ref], st.slowness[ref],
                                      MUD_NOMINAL.density, TOOL, 0.0, F_ST, caliper[ref],
                                      shear_frequency=f_dip, qs=qs[ref])
    mud_cal = S.Mud(1.0 / sf_cal, MUD_NOMINAL.density)
    ladder = {
        "A tube wave, no tool, nominal mud, bit size":
            S.sonic_anisotropy(*base, MUD_NOMINAL, S.Borehole(BIT)),
        "B + tool term (Norris, 1990)": S.sonic_anisotropy(*base, MUD_NOMINAL, TOOL),
        "C + caliper": S.sonic_anisotropy(*base, MUD_NOMINAL, TOOL, radius=caliper),
        "D + dispersion (exact TI mode, eps = delta = 0)":
            S.sonic_anisotropy(*base, MUD_NOMINAL, TOOL, F_ST, caliper),
        "D' (as D, equivalent isotropic formation)":
            S.sonic_anisotropy(*base, MUD_NOMINAL, TOOL, F_ST, caliper, equivalent_isotropic=True),
        "E + dipole C44 at 500 Hz, matched resolution":
            S.sonic_anisotropy(*matched, MUD_NOMINAL, TOOL, F_ST, caliper,
                               shear_frequency=f_dip, qs=qs),
        f"F + mud calibrated in the Stø ({mud_cal.velocity:.0f} m/s)":
            S.sonic_anisotropy(*matched, mud_cal, TOOL, F_ST, caliper, shear_frequency=f_dip, qs=qs),
        "G as E with the true mud (1540 m/s)":
            S.sonic_anisotropy(*matched, MUD_TRUE, TOOL, F_ST, caliper, shear_frequency=f_dip, qs=qs),
    }
    print("\n2. Gamma error (estimate - truth) per unit, mean; RMS over the well; samples without a solution")
    print("  " + " " * 48 + " ".join(f"{n[:9]:>9s}" for _, _, n in UNITS))
    print("  " + f"{'true gamma':48s}" + " ".join(f"{x:9.3f}" for x in per_unit(z, truth)))
    for name, r in ladder.items():
        e = r.gamma - truth
        print(f"  {name:48s}" + " ".join(f"{x:+9.3f}" for x in per_unit(z, e))
              + f"   RMS {np.sqrt(np.nanmean(e ** 2)):.3f}, NaN {int(np.isnan(e).sum())}")
    final = list(ladder.values())[6]

    # 3. uncertainty budget of the final processing (tube-wave relation at the corrected slowness)
    s_tube = st.slowness / final.dispersion
    J = S.gamma_jacobian(s_tube, 1.0 / vs_m, rho_m, mud_cal.slowness, mud_cal.density, caliper,
                         TOOL.tool_radius, 0.0)
    sig = dict(SIGMA)
    sig["s_stoneley"] = SIGMA["s_stoneley"] * s_tube
    sig["s_shear"] = SIGMA["s_shear"] / vs_m
    sig["mud_slowness"] = SIGMA["mud_slowness"] * mud_cal.slowness
    sig["mud_density"] = SIGMA["mud_density"] * mud_cal.density
    total, contrib = S.gamma_uncertainty(J, sig)
    print("\n3. Gamma standard deviation per unit (linear propagation) and its largest terms")
    print("  " + " " * 30 + " ".join(f"{n[:9]:>9s}" for _, _, n in UNITS))
    print("  " + f"{'total':30s}" + " ".join(f"{x:9.3f}" for x in per_unit(z, total, np.nanmedian)))
    for k in ("mud_slowness", "s_stoneley", "s_shear", "radius", "tool_radius", "mud_density", "rho"):
        print("  " + f"{k:30s}" + " ".join(f"{x:9.3f}" for x in per_unit(z, contrib[k], np.nanmedian)))
    rng = np.random.default_rng(1)
    print("  Monte Carlo check (4000 draws) at the unit midpoints, linear / sampled std:")
    for a, b, name in UNITS:
        i = np.searchsorted(z, 0.5 * (a + b))
        x = dict(s=s_tube[i], ss=1 / vs_m[i], rho=rho_m[i], sf=mud_cal.slowness,
                 rf=mud_cal.density, b=caliper[i], a=TOOL.tool_radius)
        n = 4000
        d = {k: x[k] + sd * rng.standard_normal(n) for k, sd in
             (("s", sig["s_stoneley"][i]), ("ss", sig["s_shear"][i]), ("rho", sig["rho"]),
              ("sf", sig["mud_slowness"]), ("rf", sig["mud_density"]), ("b", sig["radius"]),
              ("a", sig["tool_radius"]))}
        g = S.gamma_from_moduli(S.c66_from_tube_wave(d["s"], d["sf"], d["rf"], d["b"], d["a"]),
                                d["rho"] / d["ss"] ** 2)
        print(f"    {name:24s} {total[i]:.3f} / {np.nanstd(g):.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from iwt.plotting import PALETTE, depth_profiles
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    tops = [(t, n) for t, n in unit_tops(model)[1:] if t > 440.0]
    us = 1e6
    order = list(ladder)

    def smooth(x):
        x = np.where(np.isfinite(x), x, np.nan)
        ok = np.isfinite(x)
        return np.interp(z, z[ok], S.running_average(z[ok], x[ok], 5.0))
    fig = depth_profiles({
        "logs": ("slowness (us/m)", {
            "monopole P (12 kHz)": (us / logs.vp, z),
            "dipole shear (2 kHz)": (us / logs.vs, z),
            "Stoneley (500 Hz)": (us * st.slowness, z),
            "mud (nominal)": (np.full(z.size, us * MUD_NOMINAL.slowness), z)}),
        "gamma (5 m running mean)": ("gamma", {
            "true (sonic scale)": (smooth(truth), z),
            "A: tube wave, nominal mud, no tool": (smooth(ladder[order[0]].gamma), z),
            "F: all corrections, calibrated mud": (smooth(final.gamma), z),
            "G: all corrections, true mud": (smooth(ladder[order[7]].gamma), z)}),
        "gamma error budget (1 sigma)": ("standard deviation", {
            "total": (total, z),
            "mud slowness (2 %)": (contrib["mud_slowness"], z),
            "Stoneley (0.3 %) + dipole (1 %)":
                (np.hypot(contrib["s_stoneley"], contrib["s_shear"]), z),
            "radius + tool (1 mm each)": (np.hypot(contrib["radius"], contrib["tool_radius"]), z)}),
    }, top=460.0, bottom=1670.0, tops=tops, title="Sonic gamma in the vertical synthetic well "
       "(Stoneley at 500 Hz, 46 mm tool; thin-bed spikes clipped)", width=3.4)
    fig.axes[1].set_xlim(-0.2, 0.6)
    fig.axes[2].set_xlim(0.0, 0.12)
    fig.savefig(FIG / "wp4_gamma.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fig.patch.set_facecolor(PALETTE["surface"])
    styles = ("-", "--", "-.", ":")
    for k, (name, c) in enumerate(curves.items()):
        col = PALETTE["series"][k]
        axes[0].plot(c["f"], c["exact"], color=col, linestyle="-", linewidth=1.4, label=f"{name}",
                     marker="osD^"[k], markevery=6, markersize=3.5)
        axes[0].plot(c["f"], c["iso"], color=col, linestyle=":", linewidth=1.2)
        axes[1].plot(c["f"], c["inv_q"], color=col, linestyle=styles[k], linewidth=1.4, label=name)
    axes[0].plot([], [], color=PALETTE["ink2"], linestyle=":", label="dotted: equivalent isotropic")
    axes[0].set_ylabel("Stoneley slowness / tube-wave slowness", color=PALETTE["ink2"], fontsize=8)
    axes[1].set_ylabel("radiation attenuation 1/Q", color=PALETTE["ink2"], fontsize=8)
    for ax, t in zip(axes, ("Dispersion: exact TI mode (solid)", "Leakage into the formation")):
        ax.set_xscale("log")
        ax.set_xlabel("frequency (Hz)", color=PALETTE["ink2"], fontsize=8)
        ax.set_title(t, fontsize=8.5, color=PALETTE["ink"], loc="left")
        ax.tick_params(colors=PALETTE["muted"], labelsize=7, length=2)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
        ax.set_facecolor(PALETTE["surface"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["axis"])
        leg = ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        for txt in leg.get_texts():
            txt.set_color(PALETTE["ink2"])
    fig.suptitle("Stoneley wave in the units of the synthetic well (unit averages, 46 mm rigid tool)",
                 fontsize=9.5, color=PALETTE["ink"], x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "wp4_dispersion.png", dpi=150)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
