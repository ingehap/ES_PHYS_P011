"""Work package 1: the synthetic Wisting-like well.

Builds the true model and the logs, prints the numbers quoted in the README and
writes ``docs/figures/wp1_*.png``.

    python examples/wp1_synthetic_well.py
"""

from pathlib import Path

import numpy as np

from iwt import paper
from iwt.plotting import unit_tops
from iwt.synthetic import OWC_TVDSS, wisting_like_well

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"

#: Backus window: about a third of the shortest P wavelength in the overburden
#: at 80 Hz (2300 m/s / 80 Hz / 3 = 9.6 m) rounded up, and a long window of
#: about one 30-Hz wavelength for comparison.
WINDOWS = (10.0, 80.0)


def main():
    model, logs = wisting_like_well()
    print(f"model: {model.n_layers} layers, {model.interfaces[0]:.1f}-{model.interfaces[-1]:.1f} m TVDSS,"
          f" f_ref = {model.f_ref:g} Hz")

    print("\nUnit tops (m TVDSS) and vertical two-way time (ms, 30 Hz):")
    for z, name in sorted(unit_tops(model) + [(OWC_TVDSS, "OWC")]):
        print(f"  {name:24s} {z:7.1f}  {1e3 * float(model.twt(z)):7.1f}")
    print(f"  Stø top - seabed = {paper.STO_TOP_TVDSS - paper.WATER_DEPTH:.1f} m"
          f" (article: about {paper.RESERVOIR_BELOW_SEABED:g} m)")

    # sonic versus seismic traveltime ("drift") from the top of the log
    z0 = logs.depth[0]
    print(f"\nOne-way time from {z0:g} m: seismic (true, 30 Hz) minus integrated sonic (12 kHz):")
    for z in (600.0, paper.STO_TOP_TVDSS, 800.0, 1200.0, 1670.0):
        sel = (logs.depth >= z0) & (logs.depth <= z)
        s, d = 1.0 / logs.vp[sel], logs.depth[sel]
        t_sonic = np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(d))
        t_true = float(model.vertical_time(z) - model.vertical_time(z0))
        print(f"  to {z:6.1f} m: {1e3 * (t_true - t_sonic):5.2f} ms"
              f" ({100 * (t_true - t_sonic) / t_true:.2f} %)")

    # blocked (unit-average) model
    tops = [z for z, _ in unit_tops(model)[1:]] + [OWC_TVDSS]
    blk = model.block(tops)
    print("\nUnit averages (Backus): vp0 vs0 rho eps delta gamma Qp")
    for i in range(1, blk.n_layers):
        print(f"  {blk.label[i]:24s} {blk.vp0[i]:6.0f} {blk.vs0[i]:6.0f} {blk.rho[i]:6.0f}"
              f" {blk.eps[i]:6.3f} {blk.delta[i]:6.3f} {blk.gamma[i]:6.3f} {blk.qp[i]:6.1f}")

    # layer-induced anisotropy: running Backus averages that stay inside each unit
    print("\nIntrinsic versus running-Backus anisotropy (mean over the interval;"
          " windows do not cross unit tops):")
    ups = {w: model.backus(w, respect=tops) for w in WINDOWS}
    for name, (a, b) in {"Lower Cretaceous": (450.0, 595.0), "Fuglen": (632.0, 671.0),
                         "Stø": (673.0, 701.0), "Snadd": (810.0, 1670.0)}.items():
        sel = (model.mid > a) & (model.mid < b)
        line = f"  {name:17s} eps {model.eps[sel].mean():.3f} gamma {model.gamma[sel].mean():.3f}"
        for w, u in ups.items():
            line += (f" | {w:g} m: eps {u.eps[sel].mean():.3f} delta {u.delta[sel].mean():.3f}"
                     f" gamma {u.gamma[sel].mean():.3f}")
        print(line)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from iwt.plotting import backus_display, well_display
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig = well_display(model, logs, top=400.0, bottom=900.0,
                       title="Synthetic Wisting-like well: true model and logs (400-900 m)")
    fig.savefig(FIG / "wp1_well.png", dpi=150)
    fig = backus_display(model, ups[WINDOWS[0]], top=420.0, bottom=900.0,
                         label=f"Backus, {WINDOWS[0]:g} m window",
                         title="Intrinsic and layer-induced anisotropy (windows stay inside units)")
    fig.savefig(FIG / "wp1_backus.png", dpi=150)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
