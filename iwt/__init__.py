"""iwt: integrated borehole and surface seismic well tie.

A Python implementation of the workflow of Lewis et al. (2016), "Integrated
processing of borehole and surface seismic measurements for enhanced well tie and
AVO analysis", 78th EAGE Conference & Exhibition,
https://doi.org/10.3997/2214-4609.201601553, and of the methods of its references.
See ``docs/IMPLEMENTATION_PLAN.md`` for the roadmap.

Work package 1 provides the foundations: the published numbers
(:mod:`iwt.paper`), the 1D elastic VTI model with attenuation
(:mod:`iwt.model`), constant-Q dispersion (:mod:`iwt.qmodel`), Gassmann fluid
substitution (:mod:`iwt.rockphysics`) and a synthetic Wisting-like well
(:mod:`iwt.synthetic`).  Work package 2 adds forward modelling
(:mod:`iwt.modeling`), work package 3 ZO-VSP processing and Q
(:mod:`iwt.vsp`) and the sonic-to-seismic calibration (:mod:`iwt.calibration`),
work package 4 Thomsen's gamma from the sonic logs (:mod:`iwt.sonic`), and work
package 5 exact qP ray tracing (:mod:`iwt.raytrace`), the walkaway VSP inversion
(:mod:`iwt.vsp.walkaway`) and the gamma -> (eps, delta) calibration.
Plotting (:mod:`iwt.plotting`) needs matplotlib.
"""

from . import calibration, paper, qmodel, raytrace, rockphysics, sonic
from .model import (
    ElasticModel1D,
    Stiffness,
    backus_average,
    cij_to_thomsen,
    delta_min,
    is_stable,
    phase_velocities,
    thomsen_to_cij,
)
from .synthetic import WellLogs, synthetic_logs, wisting_like_model, wisting_like_well

__version__ = "0.5.0"

__all__ = [
    "calibration",
    "paper",
    "qmodel",
    "raytrace",
    "rockphysics",
    "sonic",
    "ElasticModel1D",
    "Stiffness",
    "backus_average",
    "cij_to_thomsen",
    "delta_min",
    "is_stable",
    "phase_velocities",
    "thomsen_to_cij",
    "WellLogs",
    "synthetic_logs",
    "wisting_like_model",
    "wisting_like_well",
    "__version__",
]
