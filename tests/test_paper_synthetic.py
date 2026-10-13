import numpy as np
import pytest

from iwt import paper, synthetic
from iwt.qmodel import phase_velocity
from iwt.synthetic import synthetic_logs, wisting_like_model, wisting_like_well


# -- paper constants -------------------------------------------------------------


def test_paper_constants_are_consistent():
    assert paper.MODEL_SAMPLING == 0.5
    assert paper.WAVSP_N_ESTIMATES == 6
    lo, hi = paper.WAVSP_DEPTH_RANGE
    assert lo < hi
    assert paper.WAVSP_TARGET_ANGLES == (30.0, 60.0)
    assert set(paper.TABLE_1) == {"Vp", "Vs", "density", "epsilon", "delta", "gamma", "Qp"}
    assert paper.FIG3_MIGRATION_RANKING[-1] == ("anisotropic PSDM with Q compensation", "BEST")
    # external facts agree with the article's "some 250 m below the seabed"
    assert paper.KB_ELEVATION == pytest.approx(40.0)
    assert paper.STO_TOP_TVDSS == pytest.approx(672.0)
    assert abs(paper.STO_TOP_TVDSS - paper.WATER_DEPTH - paper.RESERVOIR_BELOW_SEABED) < 10.0


# -- the synthetic model -----------------------------------------------------------


@pytest.fixture(scope="module")
def well():
    return wisting_like_well()


def test_model_structure(well):
    m, _ = well
    m.validate()
    assert m.interfaces[0] == 0.0
    assert m.interfaces[-1] == pytest.approx(paper.TD_TVDSS)
    assert m.interfaces[1] == pytest.approx(paper.WATER_DEPTH)
    assert m.is_fluid[0] and not m.is_fluid[1:].any()
    np.testing.assert_allclose(np.diff(m.interfaces[1:]), paper.MODEL_SAMPLING)
    assert m.at(paper.STO_TOP_TVDSS + 0.1)["label"] == "Stø (oil)"
    assert m.at(paper.STO_TOP_TVDSS - 0.1)["label"].startswith("Fuglen")
    assert m.at(synthetic.OWC_TVDSS + 0.1)["label"] == "Stø"
    column = synthetic.OWC_TVDSS - paper.STO_TOP_TVDSS
    assert paper.OIL_COLUMN_GROSS[0] <= column <= paper.OIL_COLUMN_GROSS[1]


def test_model_is_deterministic_and_seeded():
    a = wisting_like_model(seed=1)
    b = wisting_like_model(seed=1)
    c = wisting_like_model(seed=2)
    np.testing.assert_array_equal(a.vp0, b.vp0)
    assert not np.array_equal(a.vp0, c.vp0)


def test_blocky_model_matches_unit_values():
    m = wisting_like_model(heterogeneity=False)
    for u in synthetic.WISTING_LIKE_UNITS[1:]:
        if u.grad != 0.0:
            continue
        i = m.layer_index(u.top + 0.1)
        assert m.vp0[i] == pytest.approx(u.vp0)
        assert m.vs0[i] == pytest.approx(u.vs0)
        assert m.eps[i] == u.eps and m.qp[i] == u.qp


def test_oil_leg_is_gassmann_consistent():
    oil, water = synthetic._STO_OIL, synthetic._STO_WATER
    assert oil.vp0 < water.vp0
    assert oil.rho < water.rho
    assert oil.rho * oil.vs0**2 == pytest.approx(water.rho * water.vs0**2)


def test_oil_and_water_legs_share_one_heterogeneity_field():
    m = wisting_like_model()
    oil = m.label == "Stø (oil)"
    water = m.label == "Stø"
    rel_oil = m.vp0[oil] / synthetic._STO_OIL.vp0 - 1.0
    rel_water = m.vp0[water] / synthetic._STO_WATER.vp0 - 1.0
    # the AR(1) field is continuous across the contact
    assert abs(rel_oil[-1] - rel_water[0]) < 4 * synthetic._STO_WATER.sigma


def test_stringers_present(well):
    m, _ = well
    for s in synthetic.STRINGERS:
        i = m.layer_index(s.top + 0.25)
        assert m.vp0[i] == pytest.approx(s.vp0)
        assert m.label[i].endswith("stringer")


# -- logs -----------------------------------------------------------------------


def test_noise_free_logs_follow_dispersion():
    m = wisting_like_model(heterogeneity=False)
    logs = synthetic_logs(m, seed=None)
    # in a homogeneous interval (Hekkingen, no gradient) the log is the dispersed
    # true velocity
    z = 615.0
    i = m.layer_index(z)
    k = np.argmin(np.abs(logs.depth - z))
    assert logs.vp[k] == pytest.approx(
        float(phase_velocity(m.vp0[i], m.qp[i], 12e3, m.f_ref)), rel=1e-9)
    assert logs.vs[k] == pytest.approx(
        float(phase_velocity(m.vs0[i], m.qs[i], 2e3, m.f_ref)), rel=1e-9)
    assert logs.rho[k] == pytest.approx(m.rho[i])
    # dispersion makes every log velocity exceed the aperture-averaged true one
    v_true = 1.0 / m.window_mean(1.0 / m.vp0, logs.depth, 0.6)
    assert np.all(logs.vp > v_true)


def test_logs_interval_units_and_noise(well):
    m, logs = well
    assert logs.depth[0] == pytest.approx(470.0)
    assert logs.depth[-1] == pytest.approx(1670.0)
    c = logs.las_curves()
    assert c["DEPT"][0] == pytest.approx(470.0 + paper.KB_ELEVATION)
    # 3000 m/s is about 101.6 us/ft
    assert 1e6 * 0.3048 / 3000.0 == pytest.approx(101.6, rel=1e-3)
    assert np.all((c["DTCO"] > 60.0) & (c["DTCO"] < 200.0))
    assert np.all((c["RHOB"] > 1.8) & (c["RHOB"] < 2.8))
    clean = synthetic_logs(m, seed=None)
    rel = logs.vp / clean.vp - 1.0
    assert 0.003 < rel.std() < 0.007
    with pytest.raises(ValueError):
        synthetic_logs(m, top=400.0)


def test_sonic_drift_has_the_sign_of_dispersion(well):
    m, logs = well
    sel = (logs.depth >= 470.0) & (logs.depth <= paper.STO_TOP_TVDSS)
    slow, z = 1.0 / logs.vp[sel], logs.depth[sel]
    t_sonic = np.sum(0.5 * (slow[1:] + slow[:-1]) * np.diff(z))
    t_true = m.vertical_time(paper.STO_TOP_TVDSS) - m.vertical_time(470.0)
    # sonic (kHz) is faster than seismic (30 Hz): the integrated sonic time is short
    assert 1.5e-3 < t_true - t_sonic < 6e-3
