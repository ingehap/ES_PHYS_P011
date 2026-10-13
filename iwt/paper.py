"""Numbers published in Lewis et al. (2016), with their provenance.

Reference
---------
Lewis, O. J., Smith, P., Hohner, M., Kvist-Lassen, T., Dahlhaus, L., Stuberg, L.,
Granli, J. R., Veire, H., & Berger, P., 2016, Integrated processing of borehole and
surface seismic measurements for enhanced well tie and AVO analysis: 78th EAGE
Conference & Exhibition, Vienna, paper Th SRS3 04, 1-5.
https://doi.org/10.3997/2214-4609.201601553  (PDF in ``References/``.)

The first block holds values stated in the article, each with its page, table or
figure.  The second block holds facts about the Hanssen well taken from public
sources outside the article; each states its source and whether it is quoted or
derived.  The article gives no velocities, densities, anisotropy or Q values, so
every elastic number used for the synthetic well lives in :mod:`iwt.synthetic`
and is an assumption, not a measurement.

Depths are in metres.  "TVDSS" is true vertical depth below mean sea level,
positive down; "RKB" is depth below the rotary kelly bushing.
"""

# ---------------------------------------------------------------------------
# Stated in the article
# ---------------------------------------------------------------------------

#: p. 1 (Summary) "a shallow reservoir lying at as little as 250 m below the sea
#: bed"; p. 2 (Introduction) "some 250 m below the seabed".
RESERVOIR_BELOW_SEABED = 250.0

#: p. 1: "At the time of this study three exploration wells had been drilled".
#: Hanssen was the third (p. 2).
N_EXPLORATION_WELLS = 3

#: p. 2: the third exploration well, Hanssen, "was drilled in mid-2014".
HANSSEN_DRILLED = "mid-2014"

#: p. 3: the 1D model "was first populated, at 0.5-m depth sampling, with velocity
#: and density information from the wireline measurements".
MODEL_SAMPLING = 0.5

#: p. 3: "available reflection angles from the 3D seismic data covered only angles
#: beyond 20 deg, violating the zero-offset assumption".
SURFACE_3D_MIN_ANGLE = 20.0

#: p. 3: "the 200-m exclusion zone around the drilling platform, in which the
#: source vessel could not operate".
EXCLUSION_ZONE_RADIUS = 200.0

#: p. 3: "the available reflection angle range at target depth was limited to
#: about 30 deg to 60 deg" (walkaway VSP).
WAVSP_TARGET_ANGLES = (30.0, 60.0)

#: p. 3: "The WA-VSP provided six discrete anisotropy estimates between depths of
#: 640 and 860 m".  The depth reference (MD or TVD, RKB or sea level) is not
#: stated.
WAVSP_N_ESTIMATES = 6
WAVSP_DEPTH_RANGE = (640.0, 860.0)

#: Fig. 1 and p. 2: the data domains combined in the integrated workflow.
DATA_DOMAINS = (
    "petrophysics/geology",
    "sonic scanner",
    "zero-offset VSP",
    "walkaway VSP",
    "surface seismic",
)

#: Table 1 (p. 3): elastic property -> (source, calibration, validation).
#: ``None`` marks an empty cell.
TABLE_1 = {
    "Vp": ("wireline sonic compressional", "zero offset VSP",
           "synthetic modelling and surface seismic"),
    "Vs": ("wireline sonic shear", "rock physics (by OMV)",
           "synthetic modelling and AVO response of surface seismic and WAVSP"),
    "density": ("wireline density", None,
                "synthetic modelling and surface seismic"),
    "epsilon": ("WAVSP", "sonic derived gamma and surface seismic",
                "synthetic modelling, surface seismic and WAVSP"),
    "delta": ("WAVSP", "sonic derived gamma and surface seismic",
              "synthetic modelling, surface seismic and WAVSP"),
    "gamma": ("wireline sonic", "WAVSP and surface seismic",
              "synthetic modelling and AVO response of surface seismic and WAVSP"),
    "Qp": ("zero offset VSP", "surface seismic",
           "synthetic modelling and AVO response of surface seismic and WAVSP"),
}

#: Fig. 3 (p. 4): "AVA gradient tie to synthetic" for three depth migrations of
#: the 3D data at the Sto reflection.
FIG3_MIGRATION_RANKING = (
    ("isotropic PSDM with Q compensation", "OK"),
    ("anisotropic PSDM, no Q compensation", "OK"),
    ("anisotropic PSDM with Q compensation", "BEST"),
)

#: Fig. 4 (p. 5): the two reflections whose measured and predicted AVA are
#: compared with the measured and the modified (rock-physics) Vs.
FIG4_EVENTS = ("Stø", "Fruholmen")

#: Figs. 3 and 4: the AVA curves are plotted to 50-60 deg reflection angle.
FIG3_FIG4_MAX_ANGLE = 50.0

# ---------------------------------------------------------------------------
# Public facts about well 7324/7-2 (Hanssen) from outside the article
# ---------------------------------------------------------------------------
# These were retrieved by web search on 2026-10-13.  The primary pages
# (omv.com and factpages.sodir.no) were not reachable from the build
# environment, so the values are as quoted by the search results.

#: OMV press release, 2014, "OMV finds hydrocarbons at Hanssen well in Norway":
#: water depth 417.5 m.
WATER_DEPTH = 417.5

#: OMV press release, 2014: the well reached a vertical depth of 1679 m below
#: the sea surface, in the Middle Triassic Snadd Formation.
TD_TVDSS = 1679.0

#: Sodir factpage, wellbore 7324/7-2, wellbore history: TD at 1730 m MD,
#: 1719 m TVD (RKB).
TD_MD_RKB = 1730.0
TD_TVD_RKB = 1719.0

#: Derived, not stated: the kelly-bushing elevation implied by the two TD
#: values above (1719 - 1679 = 40 m).
KB_ELEVATION = TD_TVD_RKB - TD_TVDSS

#: Sodir factpage, wellbore 7324/7-2: "the primary target Sto Formation was
#: encountered at 712 m" and was oil bearing (taken as RKB).
STO_TOP_RKB = 712.0

#: Derived: Sto top below sea level, assuming a vertical well above the
#: reservoir (712 - 40 = 672 m TVDSS, i.e. 254.5 m below the seabed, which
#: agrees with the article's "some 250 m").
STO_TOP_TVDSS = STO_TOP_RKB - KB_ELEVATION

#: OMV press release, 2014: a gross oil column of 20-25 m in the Sto Formation.
OIL_COLUMN_GROSS = (20.0, 25.0)

#: OMV press release, 2014: main targets Sto, Nordmela and Fruholmen formations
#: (Middle Jurassic to Late Triassic), secondary target Snadd.
TARGET_FORMATIONS = ("Stø", "Nordmela", "Fruholmen")

#: OMV press release / Sodir: production licence and drilling year.
LICENCE = "PL 537"
WELLBORE = "7324/7-2"
