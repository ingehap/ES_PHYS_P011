"""The integrated loop: versioned earth models, acceptance criteria and the discrepancy register.

The article's workflow (Fig. 1) refines one earth model with every data set "until
the predictions match the measurements, or until the remaining discrepancies are
explained".  That stopping rule is qualitative.  Here the loop is made auditable:

* :class:`Register` keeps every model version with what changed and why;
* each discrepancy is an entry with the observation, the hypothesis, the test,
  its result and the decision, and the model version it produced (if any);
* acceptance criteria are fixed in advance (plan, section 5) and evaluated on
  the final version.

:func:`model_from_logs` builds the 0.5 m elastic model from wireline logs, the
starting point of the loop: sonic and dipole velocities moved from the tool
frequencies to the seismic frequency with a given Q (Kjartansson, 1979), optional
anisotropy and Q curves, the sea, and a gap layer between the seabed and the top
of the logs.

References
----------
Kjartansson, E., 1979, Constant Q-wave propagation and attenuation: JGR, 84(B9),
4737-4748. https://doi.org/10.1029/JB084iB09p04737
Lewis, O. J., et al., 2016, Integrated processing of borehole and surface seismic
measurements for enhanced well tie and AVO analysis: 78th EAGE Conference &
Exhibition. https://doi.org/10.3997/2214-4609.201601553
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .model import ElasticModel1D
from .qmodel import phase_velocity

__all__ = ["model_from_logs", "ModelVersion", "Discrepancy", "Criterion", "Register"]


def _at(values, n):
    v = np.asarray(values, dtype=float)
    return np.broadcast_to(v, (n,)).copy() if v.ndim == 0 else v


def model_from_logs(logs, water_depth: float, water_velocity: float, water_density: float,
                    qp=np.inf, qs=np.inf, eps=0.0, delta=0.0, gamma=0.0,
                    f_seismic: float = 30.0, disperse_vs: bool = True,
                    gap_length: float = 10.0, gap_vp: Optional[float] = None) -> ElasticModel1D:
    """Elastic model at the log sampling from sonic, dipole and density logs.

    ``qp``, ``qs``, ``eps``, ``delta`` and ``gamma`` are scalars or curves at
    the log depths.  The sonic Vp is moved from the monopole frequency and the
    dipole Vs from the dipole frequency (``logs.meta``) to ``f_seismic`` with
    the given Q (no change where Q is infinite); ``disperse_vs=False`` keeps the
    dipole Vs as logged.  Layers lie between log samples (slowness and density
    averaged).  The gap between the seabed and the top log takes the mean of
    the first ``gap_length`` m of logs, isotropic, with the first Q values;
    ``gap_vp`` replaces its P velocity (for example from a VSP, the logs being
    blind there).
    """
    d = np.asarray(logs.depth, dtype=float)
    n = d.size
    qp, qs = _at(qp, n), _at(qs, n)
    vp = phase_velocity(np.asarray(logs.vp, float), qp, f_seismic, logs.meta["f_sonic_p"])
    vs = np.asarray(logs.vs, float)
    if disperse_vs:
        vs = phase_velocity(vs, qs, f_seismic, logs.meta["f_sonic_s"])
    mid = lambda x: 0.5 * (x[1:] + x[:-1])          # noqa: E731
    s_p, s_s, rho = mid(1.0 / vp), mid(1.0 / vs), mid(np.asarray(logs.rho, float))
    curves = {k: mid(_at(v, n)) for k, v in (("eps", eps), ("delta", delta), ("gamma", gamma),
                                               ("qp", qp), ("qs", qs))}
    top = d < d[0] + gap_length
    gap = (np.mean(vp[top]) if gap_vp is None else float(gap_vp), np.mean(vs[top]),
           np.mean(np.asarray(logs.rho)[top]))
    return ElasticModel1D(
        np.concatenate([[0.0, water_depth], d]),
        vp0=np.concatenate([[water_velocity, gap[0]], 1.0 / s_p]),
        vs0=np.concatenate([[0.0, gap[1]], 1.0 / s_s]),
        rho=np.concatenate([[water_density, gap[2]], rho]),
        eps=np.concatenate([[0.0, 0.0], curves["eps"]]),
        delta=np.concatenate([[0.0, 0.0], curves["delta"]]),
        gamma=np.concatenate([[0.0, 0.0], curves["gamma"]]),
        qp=np.concatenate([[np.inf, qp[0]], curves["qp"]]),
        qs=np.concatenate([[np.inf, qs[0]], curves["qs"]]),
        f_ref=f_seismic)


@dataclass
class ModelVersion:
    """One state of the earth model and the reason for it."""

    number: int
    name: str
    model: ElasticModel1D
    changes: str


@dataclass
class Discrepancy:
    """An entry of the discrepancy register."""

    observation: str
    hypothesis: str
    test: str
    result: str
    decision: str
    version: Optional[int] = None


@dataclass
class Criterion:
    """An acceptance criterion fixed before tuning: ``value`` must satisfy ``sense threshold``."""

    name: str
    value: float
    threshold: float
    sense: str = "<="
    unit: str = ""

    @property
    def passed(self) -> bool:
        if self.sense == "<=":
            return bool(self.value <= self.threshold)
        if self.sense == ">=":
            return bool(self.value >= self.threshold)
        raise ValueError("sense must be '<=' or '>='")


@dataclass
class Register:
    """Model versions, discrepancies and acceptance criteria of one run of the loop."""

    versions: List[ModelVersion] = field(default_factory=list)
    discrepancies: List[Discrepancy] = field(default_factory=list)
    criteria: List[Criterion] = field(default_factory=list)

    def new_version(self, name: str, model: ElasticModel1D, changes: str) -> ModelVersion:
        v = ModelVersion(len(self.versions), name, model, changes)
        self.versions.append(v)
        return v

    @property
    def current(self) -> ModelVersion:
        return self.versions[-1]

    def log(self, observation: str, hypothesis: str, test: str, result: str, decision: str,
            version: Optional[int] = None) -> Discrepancy:
        entry = Discrepancy(observation, hypothesis, test, result, decision, version)
        self.discrepancies.append(entry)
        return entry

    def check(self, name: str, value: float, threshold: float, sense: str = "<=",
              unit: str = "") -> Criterion:
        c = Criterion(name, float(value), float(threshold), sense, unit)
        c.passed                                   # validates ``sense``
        self.criteria.append(c)
        return c

    def to_markdown(self) -> str:
        """The register as Markdown: versions, discrepancies, acceptance criteria."""
        def row(*cells):
            return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"

        out = ["## Model versions", "", "| # | name | changes |", "|---|---|---|"]
        out += [row(f"v{v.number}", v.name, v.changes) for v in self.versions]
        out += ["", "## Discrepancies", "",
                "| # | observation | hypothesis | test | result | decision | version |",
                "|---|---|---|---|---|---|---|"]
        for i, d in enumerate(self.discrepancies, start=1):
            out.append(row(i, d.observation, d.hypothesis, d.test, d.result, d.decision,
                           "–" if d.version is None else f"v{d.version}"))
        out += ["", "## Acceptance criteria (fixed before tuning)", "",
                "| criterion | value | threshold | passed |", "|---|---|---|---|"]
        for c in self.criteria:
            out.append(row(c.name, f"{c.value:.3g} {c.unit}".strip(),
                           f"{c.sense} {c.threshold:g} {c.unit}".strip(),
                           "yes" if c.passed else "**no**"))
        return "\n".join(out) + "\n"
