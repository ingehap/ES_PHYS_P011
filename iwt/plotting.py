"""Figures for the synthetic well and the 1D model (needs matplotlib).

Colours follow a fixed categorical order (blue, orange, aqua, yellow), which was
checked for colour-vision-deficiency separation.  Every series is also told apart
by line style and named in a legend, so colour never carries identity alone.
Seismic amplitudes use a diverging blue - grey - red map with the same clip in
every panel of a figure, so panels can be compared directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .model import ElasticModel1D

__all__ = ["unit_tops", "well_display", "backus_display", "avo_display", "gather_panels",
           "PALETTE"]

#: Categorical slots in fixed order, plus chart ink and chrome.
PALETTE = {
    "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
_STYLES = ("-", "--", "-.", ":")
#: Diverging pair for signed amplitudes (negative blue, positive red, grey midpoint).
_DIVERGING = ("#1c5cab", "#f0efec", "#c23b3a")


def _plt():
    import matplotlib.pyplot as plt  # deferred: optional dependency

    return plt


def unit_tops(model: ElasticModel1D):
    """``[(depth, name), ...]`` where the unit changes (stringers and fluid legs merged)."""
    names = [str(lab).replace(" stringer", "").replace(" (oil)", "") for lab in model.label]
    out = [(model.interfaces[0], names[0])]
    for i in range(1, len(names)):
        if names[i] != names[i - 1] and names[i] != out[-1][1]:
            out.append((model.interfaces[i], names[i]))
    return out


def _steps(model: ElasticModel1D, values):
    y = np.repeat(model.interfaces, 2)[1:-1]
    x = np.repeat(np.asarray(values, dtype=float), 2)
    return x, y


def _style(ax, xlabel: str):
    ax.set_xlabel(xlabel, color=PALETTE["ink2"], fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(colors=PALETTE["muted"], labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["axis"])
        spine.set_linewidth(0.6)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.set_facecolor(PALETTE["surface"])


def _legend(ax):
    # below the track, so it never covers a curve
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.005), fontsize=6.5,
                    frameon=False, borderaxespad=0.0, handlelength=2.2)
    for t in leg.get_texts():
        t.set_color(PALETTE["ink2"])


def _tops(axes, model, top, bottom, label_ax):
    for z, name in unit_tops(model):
        if not (top < z < bottom):
            continue
        for ax in axes:
            ax.axhline(z, color=PALETTE["axis"], linewidth=0.6, zorder=0)
        label_ax.annotate(name, (1.02, z), xycoords=("axes fraction", "data"),
                          fontsize=6.5, color=PALETTE["ink2"], va="bottom",
                          annotation_clip=False)


def well_display(model: ElasticModel1D, logs=None, top: Optional[float] = None,
                 bottom: Optional[float] = None, title: Optional[str] = None):
    """Five tracks: Vp0, Vs0, density, Thomsen parameters and Q, against depth.

    ``logs`` (an :class:`iwt.synthetic.WellLogs`) are overlaid on the first three
    tracks.  Returns the matplotlib figure.
    """
    plt = _plt()
    top = model.interfaces[0] if top is None else top
    bottom = model.interfaces[-1] if bottom is None else bottom
    blue, orange, aqua = PALETTE["series"][:3]
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 7.0), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(PALETTE["surface"])
    solid = ~model.is_fluid

    def track(ax, values, scale, xlabel, log_values=None, log_label=None):
        x, y = _steps(model, np.where(solid, values, np.nan) * scale)
        ax.plot(x, y, color=blue, linewidth=0.9, label="true model (30 Hz)")
        if log_values is not None:
            ax.plot(log_values * scale, logs.depth, color=orange, linewidth=0.7,
                    linestyle="--", label=log_label)
            _legend(ax)
        _style(ax, xlabel)

    lv = logs is not None
    track(axes[0], model.vp0, 1e-3, "Vp0 (km/s)", logs.vp if lv else None,
          f"sonic ({logs.meta.get('f_sonic_p', 0) / 1e3:g} kHz)" if lv else None)
    track(axes[1], model.vs0, 1e-3, "Vs0 (km/s)", logs.vs if lv else None,
          f"dipole ({logs.meta.get('f_sonic_s', 0) / 1e3:g} kHz)" if lv else None)
    track(axes[2], model.rho, 1e-3, "density (g/cm³)", logs.rho if lv else None,
          "density log" if lv else None)

    ax = axes[3]
    for k, (name, sym) in enumerate((("eps", "ε"), ("delta", "δ"), ("gamma", "γ"))):
        x, y = _steps(model, np.where(solid, getattr(model, name), np.nan))
        ax.plot(x, y, color=PALETTE["series"][k], linestyle=_STYLES[k], linewidth=0.9,
                label=sym)
    _legend(ax)
    _style(ax, "Thomsen parameters")

    ax = axes[4]
    for k, name in enumerate(("qp", "qs")):
        x, y = _steps(model, np.where(solid, getattr(model, name), np.nan))
        ax.plot(x, y, color=PALETTE["series"][k], linestyle=_STYLES[k], linewidth=0.9,
                label=name.capitalize())
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.set_xticklabels(["10", "20", "50", "100", "200"])
    ax.xaxis.set_minor_formatter(_plt().NullFormatter())
    _legend(ax)
    _style(ax, "Q")

    axes[0].set_ylim(bottom, top)
    axes[0].set_ylabel("depth (m TVDSS)", color=PALETTE["ink2"], fontsize=8)
    _tops(axes, model, top, bottom, axes[-1])
    if title:
        fig.suptitle(title, fontsize=9.5, color=PALETTE["ink"], x=0.06, ha="left", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.89, bottom=0.08)
    return fig


def backus_display(model: ElasticModel1D, upscaled: ElasticModel1D,
                   top: Optional[float] = None, bottom: Optional[float] = None,
                   label: str = "Backus average", title: Optional[str] = None):
    """Intrinsic versus upscaled Thomsen parameters (three tracks)."""
    plt = _plt()
    top = model.interfaces[0] if top is None else top
    bottom = model.interfaces[-1] if bottom is None else bottom
    blue, orange = PALETTE["series"][:2]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 6.5), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(PALETTE["surface"])
    solid = ~model.is_fluid
    for ax, (name, sym) in zip(axes, (("eps", "ε"), ("delta", "δ"), ("gamma", "γ"))):
        x, y = _steps(model, np.where(solid, getattr(model, name), np.nan))
        ax.plot(x, y, color=blue, linewidth=0.8, label="intrinsic (0.5 m)")
        x, y = _steps(upscaled, np.where(solid, getattr(upscaled, name), np.nan))
        ax.plot(x, y, color=orange, linewidth=1.3, linestyle="--", label=label)
        _legend(ax)
        _style(ax, sym)
    axes[0].set_ylim(bottom, top)
    axes[0].set_ylabel("depth (m TVDSS)", color=PALETTE["ink2"], fontsize=8)
    _tops(axes, model, top, bottom, axes[-1])
    if title:
        fig.suptitle(title, fontsize=9.5, color=PALETTE["ink"], x=0.08, ha="left", y=0.995)
    fig.subplots_adjust(left=0.1, right=0.82, top=0.89, bottom=0.08)
    return fig


def avo_display(theta_deg, panels: dict, title: Optional[str] = None,
                ylabel: str = "PP reflection coefficient"):
    """AVO curves: ``panels`` maps a panel title to ``{label: R(theta)}`` (up to 4 curves)."""
    plt = _plt()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    fig.patch.set_facecolor(PALETTE["surface"])
    for ax, (name, curves) in zip(axes[0], panels.items()):
        for k, (lab, r) in enumerate(curves.items()):
            ax.plot(theta_deg, np.real(r), color=PALETTE["series"][k], linestyle=_STYLES[k],
                    linewidth=1.4, label=lab)
        ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.6)
        ax.set_title(name, fontsize=8.5, color=PALETTE["ink"], loc="left")
        ax.set_xlabel("incidence angle (degrees)", color=PALETTE["ink2"], fontsize=8)
        ax.tick_params(colors=PALETTE["muted"], labelsize=7, length=2)
        for spine in ax.spines.values():
            spine.set_color(PALETTE["axis"])
            spine.set_linewidth(0.6)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
        ax.set_facecolor(PALETTE["surface"])
        leg = ax.legend(fontsize=6.5, frameon=False, loc="best")
        for t in leg.get_texts():
            t.set_color(PALETTE["ink2"])
    axes[0][0].set_ylabel(ylabel, color=PALETTE["ink2"], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9.5, color=PALETTE["ink"], x=0.02, ha="left")
    fig.tight_layout()
    return fig


def gather_panels(t, x, panels: dict, tmin: Optional[float] = None,
                  tmax: Optional[float] = None, clip: Optional[float] = None,
                  xlabel: str = "offset (m)", ylabel: str = "time (s)",
                  title: Optional[str] = None, marks: Optional[dict] = None):
    """Side-by-side seismic panels ``{title: data[n_x, n_t]}`` with one shared clip.

    ``clip`` defaults to the 99th percentile of |amplitude| in the first panel.
    ``marks`` maps a label to a y value drawn as a thin line across every panel.
    """
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("iwt_div", _DIVERGING)
    t = np.asarray(t)
    tmin = t[0] if tmin is None else tmin
    tmax = t[-1] if tmax is None else tmax
    sel = (t >= tmin) & (t <= tmax)
    first = np.asarray(next(iter(panels.values())))[:, sel]
    clip = np.percentile(np.abs(first), 99) if clip is None else clip
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.6, 5.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(PALETTE["surface"])
    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    for ax, (name, data) in zip(axes[0], panels.items()):
        d = np.asarray(data)[:, sel].T
        im = ax.imshow(d, aspect="auto", cmap=cmap, vmin=-clip, vmax=clip,
                       extent=[x[0] - dx / 2, x[-1] + dx / 2, t[sel][-1], t[sel][0]],
                       interpolation="bilinear")
        ax.set_title(name, fontsize=8.5, color=PALETTE["ink"], loc="left")
        ax.set_xlabel(xlabel, color=PALETTE["ink2"], fontsize=8)
        ax.tick_params(colors=PALETTE["muted"], labelsize=7, length=2)
        for spine in ax.spines.values():
            spine.set_color(PALETTE["axis"])
            spine.set_linewidth(0.6)
        for lab, y in (marks or {}).items():
            ax.axhline(y, color=PALETTE["ink2"], linewidth=0.6, linestyle="--")
            if ax is axes[0][-1]:
                ax.annotate(lab, (1.02, y), xycoords=("axes fraction", "data"), fontsize=6.5,
                            color=PALETTE["ink2"], va="center", annotation_clip=False)
    axes[0][0].set_ylabel(ylabel, color=PALETTE["ink2"], fontsize=8)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.13 if marks else 0.02)
    cb.ax.tick_params(colors=PALETTE["muted"], labelsize=6.5)
    cb.outline.set_edgecolor(PALETTE["axis"])
    cb.set_label("amplitude (same clip in every panel)", color=PALETTE["ink2"], fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9.5, color=PALETTE["ink"], x=0.02, ha="left")
    return fig
