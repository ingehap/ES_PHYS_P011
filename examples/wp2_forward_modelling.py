"""Work package 2: forward modelling of the synthetic Wisting-like well.

1. AVO at top Stø and top Fruholmen: exact isotropic (Zoeppritz), exact VTI and
   the Aki-Richards, Shuey and Rüger approximations, on unit (Backus) averages.
2. A check that the reflectivity method and the convolution synthetic agree at
   normal incidence (primaries, transmission loss, Q).
3. Transmission loss and short-period multiples at normal incidence, and
   streamer gathers of the full response against the Born (single-scattering)
   primaries: how much multiple energy reaches the reservoir window.
4. A zero-offset VSP (vertical component).

Writes ``docs/figures/wp2_*.png`` and prints the numbers quoted in the README.

    python examples/wp2_forward_modelling.py [--workers 4]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from iwt import paper
from iwt.modeling import reflection
from iwt.modeling.convolution import normal_incidence, ricker
from iwt.modeling.reflectivity import (
    Reflectivity,
    spectrum_to_trace,
    time_axis_spectrum,
    wavelet_spectrum,
)
from iwt.plotting import unit_tops
from iwt.synthetic import OWC_TVDSS, wisting_like_model

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"


def unit_average(model, name):
    """Backus average ``(vp0, vs0, rho, eps, delta)`` of the unit labelled ``name``."""
    tops = [z for z, _ in unit_tops(model)[1:]] + [OWC_TVDSS]
    blk = model.block(tops)
    i = list(blk.label).index(name)
    return blk.vp0[i], blk.vs0[i], blk.rho[i], blk.eps[i], blk.delta[i]


def avo_section(model):
    theta = np.radians(np.arange(0.0, 50.5, 1.0))
    pairs = {"top Stø (Fuglen over oil sand)": ("Fuglen", "Stø (oil)"),
             "top Fruholmen (Nordmela over Fruholmen)": ("Nordmela", "Fruholmen")}
    panels = {}
    print("\n1. AVO on unit (Backus) averages: R at 0 / 20 / 40 deg")
    for title, (a, b) in pairs.items():
        up, lo = unit_average(model, a), unit_average(model, b)
        curves = {
            "exact, isotropic (Zoeppritz)": reflection.zoeppritz_pp(theta, *up[:3], *lo[:3]).real,
            "exact, VTI": reflection.exact_pp(theta, up, lo).real,
            "Rüger (1997), VTI": reflection.ruger(theta, up, lo),
            "Aki-Richards, isotropic": reflection.aki_richards(theta, *up[:3], *lo[:3]),
        }
        panels[title] = curves
        k = [0, 20, 40]
        print(f"  {title}:  Δε = {lo[3] - up[3]:+.3f}, Δδ = {lo[4] - up[4]:+.3f}")
        for lab, r in curves.items():
            print(f"    {lab:30s} " + "  ".join(f"{r[i]:+.4f}" for i in k))
        ex = curves["exact, VTI"]
        for lab in ("Rüger (1997), VTI", "exact, isotropic (Zoeppritz)"):
            err = np.abs(curves[lab] - ex)
            print(f"    max |{lab} - exact VTI|: {err[:31].max():.4f} (0-30 deg), "
                  f"{err.max():.4f} (0-50 deg)")
    return np.degrees(theta), panels


def normal_incidence_check(model):
    print("\n2. Reflectivity (p = 0, primaries, no free surface) versus convolution")
    om, spec = normal_incidence(model, 0.002, 1024, None, top=6.0, attenuation=True,
                                transmission_loss=True, f_max=90.0, return_spectrum=True)
    refl = Reflectivity(model, free_surface=False, internal_multiples=False)
    up = refl.plane_wave(om, [0.0], 6.0, [6.0])["up_p"][0, :, 0]
    print(f"  {model.n_layers} layers, {om.size} frequencies: max |difference| = "
          f"{np.max(np.abs(up - spec)):.1e} (max |spectrum| {np.max(np.abs(spec)):.2f})")


def transmission_budget(model):
    """Normal incidence: primaries with and without transmission loss versus the full response."""
    m = model.subset(0.0, 1000.0)
    dt, nt = 0.002, 640
    wav = ricker(30.0, dt)
    _, om, eps, _ = time_axis_spectrum(dt, nt, 90.0)
    W = wavelet_spectrum(*wav, om)
    t = np.arange(nt) * dt
    traces = {}
    for name, kw in (("full response", {}),
                     ("primaries with transmission loss", dict(internal_multiples=False)),
                     ("Born primaries (no transmission loss)",
                      dict(internal_multiples=False, transmission_loss=False))):
        up = Reflectivity(m, free_surface=False, **kw).plane_wave(om, [0.0], 6.0, [6.0])["up_p"][0, :, 0]
        traces[name] = spectrum_to_trace(up * W, dt, nt, eps, om, -wav[0][0])
    z = m.rho * m.vp0
    r = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    k = m.layer_index(paper.STO_TOP_TVDSS)
    t_sto = float(m.twt(paper.STO_TOP_TVDSS) - m.twt(6.0))
    w = (t > t_sto - 0.01) & (t < t_sto + 0.05)
    print("\n3a. Normal incidence, top Stø to Nordmela window (Q on, no free surface)")
    print(f"  two-way transmission (1 - R^2) product through {k - 1} interfaces above top Stø:"
          f" {np.prod(1.0 - r[:k - 1] ** 2):.2f}")
    born = np.sqrt(np.mean(traces["Born primaries (no transmission loss)"][w] ** 2))
    for name, tr in traces.items():
        print(f"  RMS {name:38s} {np.sqrt(np.mean(tr[w] ** 2)) / born:.2f} x Born")


def surface_gathers(model, workers):
    dt, nt = 0.002, 640
    wav = ricker(30.0, dt)
    offsets = np.arange(100.0, 2001.0, 50.0)
    m = model.subset(0.0, 1000.0)
    recs = {}
    for name, kw in (("full", {}), ("born", dict(internal_multiples=False,
                                                 transmission_loss=False))):
        t0 = time.time()
        r = Reflectivity(m, attenuation=True, free_surface=False, **kw)
        recs[name] = r.record(offsets, 6.0, [8.0], dt, nt, wav, "pressure", f_max=90.0,
                              workers=workers)
        print(f"  {name} gather: {time.time() - t0:.0f} s")
    t = recs["full"].t
    full, born = recs["full"].data[0], recs["born"].data[0]
    diff = full - born
    t_sto = float(m.twt(paper.STO_TOP_TVDSS))
    t_fru = float(m.twt(740.0))
    print("\n3b. Full response versus Born primaries in the reservoir window (streamer)")
    # Beyond about 600 m the seabed reflection, with its large water moveout, crosses
    # the reservoir window in these raw (not moveout-corrected) gathers, so only the
    # near offsets measure the reservoir window cleanly.
    for lo_, hi_, lab in ((100, 600, "near 100-600 m"),):
        ratios, lags = [], []
        for i in np.flatnonzero((offsets >= lo_) & (offsets <= hi_)):
            x = offsets[i]
            a = np.sqrt(t_sto**2 + (x / 2000.0) ** 2)      # approximate moveout, vrms ~ 2000 m/s
            b = np.sqrt(t_fru**2 + (x / 2000.0) ** 2)
            w = (t >= a - 0.01) & (t <= b + 0.01)
            ratios.append(np.sqrt(np.mean(diff[i, w] ** 2) / np.mean(born[i, w] ** 2)))
            c = np.correlate(full[i, w], born[i, w], mode="full")
            lags.append((np.argmax(c) - (w.sum() - 1)) * dt)
        print(f"  {lab}: RMS(full - Born) / RMS(Born) = {np.mean(ratios):.2f};"
              f" full lags Born by {1e3 * np.median(lags):.0f} ms (cross-correlation)")
    return t, offsets, full, born, diff, t_sto, t_fru


def zovsp(model, workers):
    dt, nt = 0.002, 640
    wav = ricker(30.0, dt)
    depths = np.arange(450.0, 900.1, 5.0)
    m = model.subset(0.0, 1000.0)
    t0 = time.time()
    rec = Reflectivity(m, attenuation=True, free_surface=True).record(
        [50.0], 6.0, depths, dt, nt, wav, "uz", f_max=90.0, workers=workers)
    print(f"\n4. Zero-offset VSP (source 50 m offset, 91 levels): {time.time() - t0:.0f} s")
    uz = rec.data[:, 0, :]
    # first-break check: first peak of |uz| against the straight-ray vertical time
    picks = []
    for k, z in enumerate(depths):
        t_v = float(m.vertical_time(z) - m.vertical_time(6.0))
        w = (rec.t > t_v - 0.03) & (rec.t < t_v + 0.03)
        picks.append(rec.t[w][np.argmax(np.abs(uz[k, w]))] - t_v)
    picks = np.array(picks)
    print(f"  first-arrival peak minus vertical 30 Hz traveltime: median {1e3 * np.median(picks):+.1f} ms,"
          f" range {1e3 * picks.min():+.1f} to {1e3 * picks.max():+.1f} ms")
    return rec.t, depths, uz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    model = wisting_like_model()

    theta, panels = avo_section(model)
    normal_incidence_check(model)
    transmission_budget(model)
    t, offsets, full, born, diff, t_sto, t_fru = surface_gathers(model, args.workers)
    tv, depths, uz = zovsp(model, args.workers)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from iwt.plotting import avo_display, gather_panels
    except ImportError:
        print("\nmatplotlib not installed: no figures")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    avo_display(theta, panels, title="Exact and approximate PP reflection coefficients"
                ).savefig(FIG / "wp2_avo.png", dpi=150)
    # clip on the reservoir reflections (the seabed reflection saturates)
    near = offsets <= 1200.0
    win = (t >= 0.70) & (t <= 1.0)
    scale = np.percentile(np.abs(born[near][:, win]), 99.5)
    gather_panels(t, offsets[near], {"full response": full[near] / scale,
                                     "Born primaries": born[near] / scale,
                                     "full - Born (multiples, stratigraphic filtering)":
                                         diff[near] / scale},
                  tmin=0.6, tmax=1.0, clip=1.0,
                  marks={"top Stø (x=0)": t_sto, "top Fruholmen (x=0)": t_fru},
                  title="Streamer gathers (pressure, Q on, no free surface; steep event ="
                        " direct wave in the sea)").savefig(FIG / "wp2_multiples.png", dpi=150)
    uz_n = uz / np.max(np.abs(uz))
    gather_panels(tv, depths, {"vertical displacement (normalized, clipped at 0.2)": uz_n},
                  tmin=0.0, tmax=1.1, clip=0.2, xlabel="receiver depth (m TVDSS)",
                  title="Zero-offset VSP (source offset 50 m, free surface on)"
                  ).savefig(FIG / "wp2_zovsp.png", dpi=150)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
