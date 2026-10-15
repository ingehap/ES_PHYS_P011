"""Tests of the file adapters (iwt.io), the ZO-VSP processing chain (iwt.vsp.processing)
and the command line (iwt.cli)."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from iwt import cli
from iwt import io as IO
from iwt.model import ElasticModel1D
from iwt.modeling.convolution import ricker
from iwt.synthetic import WellLogs, wisting_like_well
from iwt.vsp import zovsp as Z
from iwt.vsp.processing import process_zovsp

# -- SEG-Y ----------------------------------------------------------------------------


def test_ibm_floats():
    # the textbook example: -118.625 = 0xC276A000
    assert int(IO.ieee_to_ibm([-118.625])[0]) == 0xC276A000
    np.testing.assert_array_equal(IO.ibm_to_ieee([0xC276A000, 0x41100000, 0x00000000, 0xC1100000]),
                                  [-118.625, 1.0, 0.0, -1.0])
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000) * 10.0 ** rng.uniform(-30, 30, 2000)
    y = IO.ibm_to_ieee(IO.ieee_to_ibm(x))
    assert np.max(np.abs(y / x - 1.0)) <= 2.0 ** -21          # hexadecimal normalization
    assert int(IO.ieee_to_ibm([1e-80])[0]) == 0                # underflow to zero
    with pytest.raises(OverflowError):
        IO.ieee_to_ibm([1e77])
    with pytest.raises(ValueError):
        IO.ieee_to_ibm([np.nan])


@pytest.mark.parametrize("fmt", [1, 5])
@pytest.mark.parametrize("byteorder", [">", "<"])
def test_segy_round_trip_and_byte_layout(tmp_path, fmt, byteorder):
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 251)).astype(np.float32)
    depth = np.array([450.0, 460.25, 470.5, 480.75, 1650.01])
    p = tmp_path / "t.sgy"
    IO.write_segy(p, data, 0.002, dict(receiver_elevation=-depth, offset=[0, 50, 100, 150, 200],
                                       source_x=np.full(5, 612345.67)),
                  text="LINE ONE\nLINE TWO", t0=0.1, fmt=fmt, byteorder=byteorder)
    raw = p.read_bytes()
    assert len(raw) == 3600 + 5 * (240 + 251 * 4)
    o = byteorder
    word = lambda pos, kind: struct.unpack(o + kind, raw[pos - 1:pos - 1 + struct.calcsize(kind)])[0]  # noqa: E731
    assert (word(3217, "h"), word(3221, "h"), word(3225, "h"), word(3503, "h")) == (2000, 251, fmt, 1)
    assert raw[3500] == (1 if o == ">" else 2)                  # revision (major byte)
    if o == "<":
        assert raw[3296:3300] == struct.pack("<I", 0x01020304)
    second = 3600 + 240 + 251 * 4                               # second trace header
    assert word(second + 37, "i") == 50                         # offset, bytes 37-40
    assert word(second + 69, "h") == -100                       # elevation scalar: centimetres
    assert word(second + 41, "i") == -46025                     # receiver elevation in cm
    assert word(second + 109, "h") == 100                       # delay, ms
    assert raw[:3].decode("cp037") == "C 1"                     # EBCDIC textual header
    sg = IO.read_segy(p)
    tol = 2.0 ** -21 if fmt == 1 else 0.0
    np.testing.assert_allclose(sg.data, data, rtol=tol, atol=0)
    assert (sg.dt, sg.t0) == (0.002, 0.1)
    np.testing.assert_allclose(sg.headers["receiver_elevation"], -depth)
    np.testing.assert_allclose(sg.headers["source_x"], 612345.67)
    np.testing.assert_array_equal(sg.headers["trace_sequence_line"], np.arange(1, 6))
    assert sg.text.splitlines()[:2] == ["C 1 LINE ONE", "C 2 LINE TWO"]
    assert sg.text.splitlines()[-1] == "C40 END TEXTUAL HEADER"


def test_segy_against_segyio(tmp_path):
    segyio = pytest.importorskip("segyio")
    rng = np.random.default_rng(2)
    data = rng.standard_normal((4, 120)).astype(np.float32)
    # ours -> segyio
    p = tmp_path / "ours.sgy"
    IO.write_segy(p, data, 0.004, dict(offset=[10, 20, 30, 40], receiver_elevation=[-1.5] * 4),
                  fmt=1)
    with segyio.open(str(p), ignore_geometry=True) as f:
        np.testing.assert_allclose(f.trace.raw[:], data, rtol=2.0 ** -21)
        assert segyio.tools.dt(f) == 4000.0
        assert list(f.attributes(segyio.TraceField.offset)[:]) == [10, 20, 30, 40]
        assert f.header[3][segyio.TraceField.ReceiverGroupElevation] == -150
        assert f.header[3][segyio.TraceField.ElevationScalar] == -100
    # segyio -> ours (IBM and IEEE)
    for fmt in (1, 5):
        q = tmp_path / f"segyio{fmt}.sgy"
        spec = segyio.spec()
        spec.format, spec.samples, spec.tracecount = fmt, np.arange(120) * 4.0, 4
        with segyio.create(str(q), spec) as f:
            for i in range(4):
                f.header[i] = {segyio.TraceField.offset: 100 * (i + 1),
                               segyio.TraceField.SourceDepth: 600,
                               segyio.TraceField.ElevationScalar: -100,
                               segyio.TraceField.TRACE_SAMPLE_INTERVAL: 4000}
                f.trace[i] = data[i]
            f.bin.update(hdt=4000, format=fmt)
        sg = IO.read_segy(q)
        np.testing.assert_allclose(sg.data, data, rtol=2.0 ** -21 if fmt == 1 else 0)
        np.testing.assert_array_equal(sg.headers["offset"], [100, 200, 300, 400])
        np.testing.assert_allclose(sg.headers["source_depth"], 6.0)
        assert sg.dt == 0.004


def test_vsp_and_gather_files(tmp_path):
    rng = np.random.default_rng(3)
    vsp = Z.VSP(np.arange(400) * 0.001, np.array([470.0, 450.0, 460.5]),
                rng.standard_normal((3, 400)), source_depth=6.0, source_offset=55.0)
    IO.write_zovsp(tmp_path / "v.sgy", vsp)
    r = IO.read_zovsp(tmp_path / "v.sgy")
    np.testing.assert_allclose(r.depths, [450.0, 460.5, 470.0])          # sorted by depth
    np.testing.assert_allclose(r.data, vsp.data[[1, 2, 0]], rtol=1e-6)
    assert (r.source_depth, r.source_offset, r.dt) == (6.0, 55.0, 0.001)
    # depths below the rig floor stored as positive words, converted with the KB
    IO.write_segy(tmp_path / "md.sgy", vsp.data, 0.001,
                  dict(receiver_datum_elevation=vsp.depths + 40.0))
    r = IO.read_zovsp(tmp_path / "md.sgy", depth_field="receiver_datum_elevation", depth_sign=1.0,
                      depth_shift=-40.0, source_depth=6.0)
    np.testing.assert_allclose(r.depths, [450.0, 460.5, 470.0])
    g = IO.Gather(0.2 + np.arange(300) * 0.002, np.array([300.0, 100.0, 200.0]),
                  rng.standard_normal((3, 300)), source_depth=6.0, receiver_depth=8.0)
    IO.write_gather(tmp_path / "g.sgy", g)
    h = IO.read_gather(tmp_path / "g.sgy")
    np.testing.assert_allclose(h.offsets, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(h.t, g.t)
    assert (h.source_depth, h.receiver_depth) == (6.0, 8.0)


# -- logs, models, tables -------------------------------------------------------------

LAS_METRIC = """~VERSION INFORMATION
 VERS.                 2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.                  NO :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
 STRT.M           1670.000 :
 STOP.M           1669.250 :
 STEP.M             -0.250 :
 NULL.           -999.2500 :
 WELL.           TEST WELL :   WELL
~CURVE INFORMATION
 DEPT.M                    :   DEPTH
 DT  .US/M                 :   SONIC TRANSIT TIME
 DTS .US/M                 :   SHEAR TRANSIT TIME
 RHOB.K/M3                 :   BULK DENSITY
~A  DEPTH     DT       DTS      RHOB
1670.000   400.000   800.000  2300.000
1669.750   250.000   500.000  2500.000
1669.500   400.000  -999.250  2300.000
1669.250   500.000  1000.000  2100.000
"""

LAS_IMPERIAL = """~VERSION INFORMATION
 VERS.                 2.0 :
 WRAP.                  NO :
~WELL INFORMATION
 STRT.FT          1000.000 :
 STOP.FT          1001.000 :
 STEP.FT             1.000 :
 NULL.           -999.2500 :
 EKB .FT           131.234 :   KELLY BUSHING
~CURVE INFORMATION
 DEPT.FT                   :
 DTCO.US/F                 :
 DTSM.US/F                 :
 RHOB.G/CC                 :
~A
1000.000   100.000   200.000   2.300
1001.000   120.000   240.000   2.400
"""


def test_read_las_units_and_depths(tmp_path):
    pytest.importorskip("lasio")
    p = tmp_path / "metric.las"
    p.write_text(LAS_METRIC)
    with pytest.raises(ValueError, match="kelly-bushing"):
        IO.read_las(p)
    logs = IO.read_las(p, kb=20.0)
    np.testing.assert_allclose(logs.depth, [1649.25, 1649.75, 1650.0])   # sorted; null row dropped
    np.testing.assert_allclose(logs.vp, [2000.0, 4000.0, 2500.0])        # 1e6 / (us/m)
    np.testing.assert_allclose(logs.vs, [1000.0, 2000.0, 1250.0])
    np.testing.assert_allclose(logs.rho, [2100.0, 2500.0, 2300.0])
    assert logs.meta["dropped_samples"] == 1 and logs.meta["curves"]["vs"] == "DTS"
    q = tmp_path / "imperial.las"
    q.write_text(LAS_IMPERIAL)
    logs = IO.read_las(q, step=0.1524)                   # KB from the EKB item (ft)
    np.testing.assert_allclose(logs.depth[[0, -1]], [1000 * 0.3048 - 40.0, 1001 * 0.3048 - 40.0],
                               atol=1e-4)
    np.testing.assert_allclose(logs.vp[[0, -1]], [3048.0, 2540.0])
    np.testing.assert_allclose(logs.rho[[0, -1]], [2300.0, 2400.0])
    assert logs.depth.size == 3                          # resampled every 0.1524 m
    np.testing.assert_allclose(1.0 / logs.vp[1], 0.5 * (1 / 3048.0 + 1 / 2540.0))   # slowness


def test_las_round_trip(tmp_path):
    pytest.importorskip("lasio")
    import lasio

    _, logs = wisting_like_well()
    IO.write_las(tmp_path / "w.las", logs, kb=40.0)
    r = IO.read_las(tmp_path / "w.las")                  # TVDSS curve
    np.testing.assert_allclose(r.depth, logs.depth)
    np.testing.assert_allclose(r.vp, logs.vp, rtol=1e-6)
    np.testing.assert_allclose(r.vs, logs.vs, rtol=1e-6)
    np.testing.assert_allclose(r.rho, logs.rho, atol=0.01)
    las = lasio.read(str(tmp_path / "w.las"))
    las.delete_curve("TVDSS")
    with open(tmp_path / "md.las", "w") as fh:
        las.write(fh)
    r = IO.read_las(tmp_path / "md.las")                 # MD - EKB
    assert r.meta["kb"] == 40.0
    np.testing.assert_allclose(r.depth, logs.depth, atol=1e-6)


def test_csv_logs_models_and_tables(tmp_path):
    _, logs = wisting_like_well()
    IO.write_logs_csv(tmp_path / "l.csv", logs)
    r = IO.read_logs(tmp_path / "l.csv")
    np.testing.assert_allclose(r.vp, logs.vp, rtol=1e-9)
    assert r.meta["f_sonic_p"] == 12e3
    m = ElasticModel1D([0.0, 417.5, 470.0, 480.0], vp0=[1470.0, 2100.0, 2400.0],
                       vs0=[0.0, 800.0, 1000.0], rho=[1025.0, 2000.0, 2250.0],
                       eps=[0.0, 0.0, 0.2], delta=[0.0, 0.0, 0.1], qp=[np.inf, 45.0, 60.0],
                       label=["Sea water", "gap, from VSP", "Lower Cretaceous"], f_ref=25.0)
    IO.write_model_csv(tmp_path / "m.csv", m)
    r = IO.read_model_csv(tmp_path / "m.csv")
    for k in ("interfaces", "vp0", "vs0", "rho", "eps", "delta", "gamma", "qp", "qs"):
        np.testing.assert_array_equal(getattr(r, k), getattr(m, k))
    assert list(r.label) == list(m.label) and r.f_ref == 25.0
    IO.write_time_depth(tmp_path / "td.csv", [450.0, 460.0], [0.3, 0.305], velocity=[2000.0, 2100.0])
    z, t = IO.read_time_depth(tmp_path / "td.csv")
    np.testing.assert_allclose(z, [450.0, 460.0])
    np.testing.assert_allclose(t, [0.3, 0.305])


# -- ZO-VSP processing ----------------------------------------------------------------


def test_process_zovsp_recovers_times_and_intrinsic_q():
    m = ElasticModel1D([0.0, 400.0, 600.0, 650.0, 800.0, 1000.0],
                       vp0=[1470.0, 2300.0, 3000.0, 2500.0, 3200.0],
                       vs0=[0.0, 1000.0, 1500.0, 1150.0, 1700.0],
                       rho=[1025.0, 2200.0, 2400.0, 2250.0, 2450.0],
                       qp=[np.inf, 40.0, 40.0, 80.0, 80.0], qs=[np.inf, 20.0, 20.0, 40.0, 40.0])
    depths = np.arange(450.0, 950.1, 10.0)
    dt = 0.001
    vsp = Z.synthetic_zovsp(m, depths, 6.0, dt, 900, ricker(40.0, dt), f_max=120.0)
    strat = m.copy()
    strat.qp[:] = np.inf
    strat.qs[:] = np.inf
    r = process_zovsp(vsp, 1470.0, q_blocks=[650.0], strat_model=strat, corridor_peak=30.0)
    err = r.times - m.vertical_time(depths)
    assert abs(np.mean(err)) < 1e-3 and np.ptp(err) < 0.7e-3   # bias of the reference pick
    assert r.q_kind == "intrinsic"
    np.testing.assert_allclose(r.q.q, [40.0, 80.0], rtol=0.06)
    assert np.max(np.abs(r.dt_star_strat)) < 0.5e-3      # four interfaces: little layering loss
    assert process_zovsp(vsp, 1470.0, q_blocks=[650.0]).q_kind == "apparent"
    assert r.corridor.shape == vsp.t.shape and np.isfinite(r.corridor).all()
    d = r.to_dict()
    assert json.loads(json.dumps(d))["q"]["kind"] == "intrinsic"


# -- command line ---------------------------------------------------------------------


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "demo"
    assert cli.main(["demo", str(out), "--quick"]) == 0
    logs = out / ("logs.las" if (out / "logs.las").exists() else "logs.csv")
    assert cli.main(["vsp", str(out / "zovsp.sgy"), "--datum-velocity", "1470", "--q-blocks",
                     "600", "740", "800", "--logs", str(logs), "--water-depth", "417.5",
                     "--corridor", str(out / "corridor.sgy"), "--td", str(out / "td.csv"),
                     "-o", str(out / "vsp.json")]) == 0
    v = json.loads((out / "vsp.json").read_text())
    assert v["q"]["kind"] == "intrinsic" and len(v["q"]["top"]) == 4
    model, _ = wisting_like_well()
    np.testing.assert_allclose(v["vertical_time"], model.vertical_time(v["depths"]), atol=1.5e-3)
    assert cli.main(["model", str(logs), "--water-depth", "417.5", "--vsp", str(out / "vsp.json"),
                     "-o", str(out / "model.csv")]) == 0
    text = capsys.readouterr().out
    assert "gap velocity" in text and "intrinsic Q" in text
    m = IO.read_model_csv(out / "model.csv")
    assert m.interfaces[1] == 417.5 and np.isfinite(m.qp[2:]).all()
    assert cli.main(["tie", str(out / "model.csv"), str(out / "corridor.sgy"),
                     "-o", str(out / "tie_c.json")]) == 0
    assert json.loads((out / "tie_c.json").read_text())["metrics"]["correlation"] > 0.85
    assert cli.main(["tie", str(out / "model.csv"), str(out / "seismic.sgy"), "--attenuation",
                     "-o", str(out / "tie_s.json")]) == 0
    tie = json.loads((out / "tie_s.json").read_text())["metrics"]
    assert tie["correlation"] > 0.9 and abs(tie["shift"]) < 0.003
    assert cli.main(["tie", str(out / "model.csv"), str(out / "seismic.sgy"), "--attenuation",
                     "--wavelet", "estimate"]) == 0
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_cli_rejects_bad_input(tmp_path):
    logs = WellLogs(np.array([470.0, 470.5]), np.array([2400.0, 2410.0]), np.array([1000.0, 1001.0]),
                    np.array([2200.0, 2210.0]), dict(f_sonic_p=12e3, f_sonic_s=2e3))
    IO.write_logs_csv(tmp_path / "l.csv", logs)
    with pytest.raises(SystemExit):                      # --water-depth is required
        cli.main(["model", str(tmp_path / "l.csv"), "-o", str(tmp_path / "m.csv")])
    (tmp_path / "bad.sgy").write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError):
        IO.read_segy(tmp_path / "bad.sgy")


def test_python_m_iwt():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-m", "iwt", "--version"], capture_output=True, text=True,
                         env=env, cwd=root, check=True)
    import iwt

    assert out.stdout.strip() == f"iwt {iwt.__version__}"
