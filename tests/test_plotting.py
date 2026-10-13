import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from iwt.plotting import backus_display, unit_tops, well_display  # noqa: E402
from iwt.synthetic import wisting_like_well  # noqa: E402


@pytest.fixture(scope="module")
def well():
    return wisting_like_well()


def test_unit_tops_merge_stringers_and_fluid_legs(well):
    model, _ = well
    names = [n for _, n in unit_tops(model)]
    assert names[:3] == ["Sea water", "Quaternary", "Lower Cretaceous shale"]
    assert names.count("Stø") == 1
    assert not any("stringer" in n for n in names)
    depths = [z for z, _ in unit_tops(model)]
    assert np.all(np.diff(depths) > 0)


def test_figures_render(well):
    model, logs = well
    fig = well_display(model, logs, top=400.0, bottom=900.0, title="t")
    assert len(fig.axes) == 5
    up = model.backus(10.0, respect=[z for z, _ in unit_tops(model)[1:]])
    fig2 = backus_display(model, up, top=420.0, bottom=900.0)
    assert len(fig2.axes) == 3
    import matplotlib.pyplot as plt

    plt.close("all")


def test_wp2_figures_render():
    from iwt.plotting import avo_display, gather_panels

    theta = np.linspace(0.0, 40.0, 9)
    fig = avo_display(theta, {"a": {"x": theta * 0.001, "y": -theta * 0.001},
                              "b": {"x": theta * 0.0}}, title="t")
    assert len(fig.axes) == 2
    t = np.arange(100) * 0.004
    x = np.arange(5) * 100.0
    d = np.random.default_rng(0).standard_normal((5, 100))
    fig2 = gather_panels(t, x, {"one": d, "two": 0.5 * d}, tmin=0.1, tmax=0.3,
                         marks={"m": 0.2}, title="g")
    assert len(fig2.axes) == 3          # two panels and the colour bar
    import matplotlib.pyplot as plt

    plt.close("all")
