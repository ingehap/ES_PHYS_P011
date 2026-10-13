# Implementation plan: integrated borehole and surface seismic well tie (Lewis et al., 2016)

Status: **plan only, nothing implemented yet.** This document says what will be built, in what
order, how each part will be checked, and where the method can be improved. It follows the
conventions of the earlier repositories in this series (for example `ES_PHYS_P010`): a Python
package, a `paper.py` that records where each published number comes from, figures redrawn on
synthetic data, a test suite, a critical assessment, and references checked against the
literature.

## 1. The article

> Lewis, O. J., Smith, P., Hohner, M., Kvist-Lassen, T., Dahlhaus, L., Stuberg, L., Granli,
> J. R., Veire, H., & Berger, P. (2016). Integrated processing of borehole and surface seismic
> measurements for enhanced well tie and AVO analysis. *78th EAGE Conference & Exhibition*,
> Vienna, paper Th SRS3 04, 1–5. https://doi.org/10.3997/2214-4609.201601553
> (PDF in [`References/`](../References/).)

**The setting.** Wisting (OMV, southwest Barents Sea) is a shallow oil reservoir, as little as
250 m below the seabed. The first two wells were hard to tie to the surface seismic. At the third
well, Hanssen (7324/7-2, drilled 2014), OMV acquired the following, which complemented the
existing 3D survey and two orthogonal high-resolution 2D site-survey lines:

- wireline logs, including sonic;
- a zero-offset VSP (ZO-VSP);
- a multi-azimuth walkaway VSP (WA-VSP).

**The method** (Fig. 1 of the article):

1. Build one 1D elastic model at the well: VTI anisotropy (Vp, Vs, ρ, ε, δ, γ) and Q, sampled
   every 0.5 m.
2. Predict the borehole and surface seismic responses from that model by elastic
   finite-difference modelling.
3. Explain or remove each discrepancy between prediction and measurement by editing,
   reprocessing or updating the model.
4. Repeat until all data sets agree.

Table 1 of the article lists which data set supplies, calibrates and validates each property:

| Property | Source | Calibrated by | Validated by |
|---|---|---|---|
| Vp | wireline sonic | ZO-VSP | synthetics, surface seismic |
| Vs | wireline dipole shear | rock physics (OMV) | synthetics, AVO of surface seismic and WA-VSP |
| ρ | wireline density | – | synthetics, surface seismic |
| ε, δ | WA-VSP | sonic γ, surface seismic | synthetics, surface seismic, WA-VSP |
| γ | wireline sonic | WA-VSP, surface seismic | synthetics, AVO of surface seismic and WA-VSP |
| Qp | ZO-VSP | surface seismic | synthetics, AVO of surface seismic and WA-VSP |

**Key findings:**

- Overburden VTI anisotropy and Q were both needed in depth imaging. Anisotropic PSDM with Q
  compensation gave the best AVA-gradient tie (Fig. 3).
- Interbed multiples from the overburden contaminate the reservoir reflections.
- A rock-physics Vs in the reservoir matched the AVA better than the measured dipole Vs (Fig. 4).
- Anisotropic Q-PSDM revealed an oil–water contact that the legacy image did not show (Fig. 5).

**The three cited references, and what each contributes to the code:**

| Reference | Role in the workflow | Module |
|---|---|---|
| Leaney & Hornby (2007), https://doi.org/10.3997/2214-4609.201401677 | ε and δ by slowness–polarization inversion of the WA-VSP | `iwt.vsp.walkaway` |
| Walsh et al. (2007), https://doi.org/10.1190/1.2792435 | continuous γ from sonic: C44 from dipole shear, C66 from Stoneley | `iwt.sonic` |
| Cavalca, Fletcher & Riedel (2013), https://doi.org/10.1190/segam2013-1130.1 | ray-based t\* Q-compensation inside Kirchhoff depth migration | `iwt.imaging` |

## 2. Assumptions

1. **"LFP article" means the Lewis et al. (2016) paper**, the only article stored in the repo.
   If a different article was meant, only sections 1 and 7 change.
2. **"Implement" means a tested Python package**, laid out like `ES_PHYS_P010`. It reproduces
   the method and the *structure* of the figures, not the measured values.
3. **"Its references" means implementing the methods** of the three cited papers to the extent
   the workflow needs them. Their own case studies (sub-salt Gulf of Mexico WA-VSP, Barnett
   shale sonic, the BP 2004 model) are not reproduced.
4. **The Wisting/Hanssen data are proprietary.** The default data set is therefore a synthetic
   "Wisting-like" well with known true properties, built from published facts:
   - water depth 417.5 m;
   - top reservoir about 250 m below the seabed, i.e. about 670 m TVDSS;
   - a 20–25 m gross oil column in the Stø Fm, with Fruholmen below;
   - TD 1679 m TVD in the Snadd Fm;
   - WA-VSP anisotropy levels at 640–860 m.

   Sources: [OMV press release, 2014](https://www.omv.com/en/media/press-releases/2014/omv-finds-hydrocarbons-at-hanssen-well-in-norway)
   and the article. The article does not say whether its 640–860 m is MD or TVD, or which datum
   it uses. Here these depths are taken as TVDSS, which is plausible but unconfirmed.

   Data from released Norwegian wells can be ordered through the Diskos Public Portal
   ([Sodir: access to released data](https://www.sodir.no/en/facts/data-and-analyses/release-of-data/access-to-released-data/);
   [well 7324/7-2 factpage](https://factpages.sodir.no/no/wellbore/PageView/With/Wdss/7450)).
   Whether the Hanssen logs and VSPs are released has **not** been checked. LAS and SEG-Y
   adapters are planned so that real data can replace the synthetic set.
5. **Dependencies differ from `ES_PHYS_P010`.** The core uses **numpy + scipy**, not numpy only.
   The work needs robust nonlinear least squares, eigen-solvers and spectral estimation, and
   rewriting those in numpy is not worth it. matplotlib, lasio and segyio stay optional.
6. **Python ≥ 3.9.** The package is named `iwt` ("integrated well tie"). The name is easy to
   change.

## 3. Target architecture

```
iwt/
  paper.py          every number in the article with page/figure provenance (+ external facts, sourced)
  model.py          ElasticModel1D (z, Vp0, Vs0, rho, eps, delta, gamma, Qp, Qs); Thomsen <-> Cij;
                    stability checks; blocking; Backus (1962) upscaling; time-depth
  synthetic.py      Wisting-like true model and the "measured" data sets derived from it (logs at
                    sonic frequency, ZO-VSP, multi-azimuth WA-VSP with 200 m exclusion zone,
                    3D gathers with angles > 20 deg, high-resolution 2D gathers), with noise
  modeling/
    convolution.py  1D convolutional synthetics; Zoeppritz, Aki-Richards, Shuey (1985), Rueger (1997)
    reflectivity.py layered-medium reflectivity method (Kennett & Kerry 1979; VTI after Fryer &
                    Frazer 1984) with complex velocities for Q; switches for free surface, internal
                    multiples and converted waves; surface and buried-receiver (VSP) geometries
    fd2d.py         optional: 2D staggered-grid elastic VTI finite differences (Virieux 1986) as an
                    independent cross-check
  vsp/
    zovsp.py        first-break picking, vertical-time correction, interval velocity (regularized),
                    up/down separation, deconvolution, corridor stack
    attenuation.py  Q by spectral ratio (Hauge 1981) and centroid shift (Quan & Harris 1997);
                    interval Q by regularized inversion; intrinsic vs stratigraphic split
    walkaway.py     horizontal slowness p, vertical slowness q, polarization angle; exact-Christoffel
                    VTI inversion per depth level with priors (Leaney & Hornby 2007; Gaiser 1990;
                    Miller & Spencer 1994); posterior covariance and resolution
  sonic.py          C33, C44, C66 (Stoneley; Norris 1990 tube-wave speed) -> gamma (Walsh et al. 2007)
  calibration.py    sonic-to-seismic drift: classic knee-point drift correction and a physical
                    alternative (Kjartansson 1979 dispersion + Backus + anisotropy); gamma -> eps, delta
  raytrace.py       exact VTI phase/group velocities; 1D layered ray tracing; traveltime, t*, angle
  imaging.py        1D prestack depth migration of CMP gathers (isotropic / VTI) with t* amplitude and
                    phase Q-compensation and a gain limit (Cavalca et al. 2013; Wang 2006)
  avo.py            AVA extraction, intercept/gradient fits, Vs scenarios, sensitivity (Jacobian)
  welltie.py        wavelet estimation (Walden & White 1998), correlation, time shift, phase
  multiples.py      interbed-multiple contamination: full minus primaries-only reflectivity
  workflow.py       the integrated loop: model -> predict -> misfit per domain -> discrepancy register
  io.py, cli.py     LAS (lasio), SEG-Y (segyio), CSV; `iwt` command
  plotting.py       figure analogues
examples/reproduce_paper.py   writes docs/figures/* and prints the paper-vs-implementation table
tests/                        pytest; analytic and round-trip checks (section 5)
```

## 4. Work packages

Each package is one PR, as in `ES_PHYS_P010`. Every PR ships with tests and a README section and
leaves the earlier results bit-identical.

### WP1: Foundations

**Builds** `paper.py`, `model.py`, `synthetic.py` (true model plus logs), `pyproject.toml`, the
README skeleton and a GitHub Actions workflow running `pytest` and `pyflakes`. The CI workflow
is an improvement over `ES_PHYS_P010`, which has none.

**`paper.py` records:**

- 0.5 m sampling (p. 3);
- reservoir 250 m below the seabed (p. 1);
- 3D angles only beyond 20° (p. 3);
- 200 m exclusion zone and 30–60° available at target (p. 3);
- six WA-VSP anisotropy estimates at 640–860 m (p. 3);
- the Table 1 mapping;
- the Fig. 3 migration ranking.

**Checks:**

- Thomsen ↔ Cij round trip;
- stiffness positive-definite;
- Backus averaging of isotropic layers gives γ ≥ 0 and ε ≥ 0 (horizontal velocity at least the
  vertical);
- Backus of identical layers is the identity.

### WP2: Forward modelling

**Builds** convolutional synthetics with AVO approximations, and the reflectivity method
(isotropic first, then VTI) with constant Q.

**Why the reflectivity method instead of the article's elastic FD:** for a 1D model it is exact,
fast and free of grid dispersion. It can also switch internal multiples on and off, which
`multiples.py` needs. 2D FD is kept as an optional cross-check.

**Checks:**

- a single interface matches the exact Zoeppritz coefficients;
- Rüger reduces to Aki–Richards when ε = δ = 0;
- energy is conserved at an elastic interface;
- Q dispersion matches Kjartansson (1979);
- source–receiver reciprocity holds;
- a reflectivity seismogram matches FD in a small model (if `fd2d` is built).

### WP3: ZO-VSP processing, Q and the sonic calibration

**Builds:**

- picking and time–depth;
- regularized interval velocity (plain differencing amplifies picking noise);
- up/down separation;
- deconvolution and the corridor stack;
- Q by spectral ratio and by centroid shift;
- the drift curve.

**Improvement over the article: one consistent Q model.** The Q that the ZO-VSP measures should
also predict the sonic-to-seismic velocity dispersion (Kjartansson, 1979; Stewart et al., 1984).
The plan therefore provides:

- the classic empirical drift correction;
- a *physical* one: dispersion from the ZO-VSP Q, plus Backus upscaling and an anisotropy
  term;
- a report of the residual between the two.

An empirical drift correction absorbs Q and anisotropy effects. Applying Q again in imaging then
counts them twice.

**Improvement: separate intrinsic from stratigraphic attenuation.** The apparent VSP Q also
contains short-path multiple ("stratigraphic") filtering (O'Doherty & Anstey, 1971; Banik et al.,
1985). The 0.5 m log model predicts that part by reflectivity, so intrinsic Q can be estimated as
1/Q_int = 1/Q_app − 1/Q_strat.

**Checks:**

- recover a known Q (true Q ± tolerance) from a synthetic ZO-VSP, with and without noise;
- recover a known drift from a known Q and frequency pair;
- the corridor stack correlates with the true reflectivity.

### WP4: Sonic anisotropy (Walsh et al., 2007)

**Builds** the vertical-well case:

- C33 = ρVp² and C44 = ρVs² (dipole);
- C66 from the low-frequency Stoneley slowness, using the tube-wave relation with Norris (1990)
  corrections for tool and eccentricity;
- γ = (C66 − C44)/(2C44).

**Uncertainty propagation** covers mud slowness, mud density and the tool term. The γ error from
a mud-slowness error is usually the dominant term.

The deviated-well extension (Horne et al., 2012) is noted but not built unless real
deviated-well data appear.

**Checks:**

- forward Stoneley slowness from known C66, then inversion, returns the original;
- γ = 0 in an isotropic formation;
- sensitivity derivatives against finite differences.

### WP5: Walkaway VSP anisotropy (Leaney & Hornby, 2007)

**Measurements:**

- p = ∂t/∂x_source at each receiver;
- q from the moveout across receivers;
- polarization angle from 3C hodograms (principal components).

**Inversion:**

- invert per depth level for (Vp0, Vs0, ε, δ) with the *exact* VTI dispersion relation, not
  the weak-anisotropy form;
- use priors on Vs0 and γ from sonic, as the article does to stabilize missing near angles;
- report the posterior covariance, which shows the ε–δ trade-off that the 30–60° aperture
  causes.

**Multi-azimuth check:** compare the azimuths to detect azimuthal anisotropy, which a VTI fit
would hide (Miller & Spencer, 1994).

**Lateral heterogeneity:** horizontal slowness at depth equals surface slowness only in a
laterally homogeneous medium (Gaiser, 1990). The plan includes a test that flags violations of
this assumption (Grechka et al., arXiv 2019).

**γ → (ε, δ) calibration** (`calibration.py`):

- fit ε and δ against γ at the six levels with a *Bayesian* linear model and leave-one-level-out
  cross-validation;
- propagate the result to the continuous γ curve with uncertainty bands.

Six points cannot support more than about two parameters per curve. The plan makes that limit
visible rather than hiding it.

**Checks:**

- noise-free synthetic WA-VSP → true ε and δ within 0.005;
- noisy data → posterior covering the truth at the stated rate (coverage test);
- the ε–δ trade-off grows as the near-angle cut-off increases.

### WP6: Ray tracing and Q-compensated depth imaging (Cavalca et al., 2013)

**Builds:**

- exact VTI phase and group velocities;
- layered ray tracing giving traveltime, t\* = ∫dt/Q and incidence angle;
- 1D prestack depth migration of synthetic CMP gathers.

**Migration:**

- offset → angle mapping, and depth mapping, in isotropic or VTI models;
- t\* amplitude and Kjartansson phase compensation inside the migration operator;
- a user gain limit, because Cavalca et al. note noise amplification.

**Reproduces Fig. 3:** four imaging variants (isotropic or VTI × with or without Q) compared on
reservoir AVA gradient against the elastic synthetic, scored numerically.

**Checks:**

- the homogeneous VTI ray tracer matches the analytic group velocity;
- traveltimes match the Alkhalifah & Tsvankin (1995) nonhyperbolic moveout to within its known
  accuracy;
- Q compensation of a forward-attenuated spike recovers it up to the gain limit.

### WP7: Integration, AVO, multiples and the figures

**Builds:**

- the `workflow.py` loop;
- well-tie metrics (Fig. 2);
- interbed-multiple contamination from the full minus the primaries-only reflectivity;
- Vs scenarios at Stø and Fruholmen (Fig. 4);
- a 2D convolutional section with a flat oil–water contact under a dipping reservoir, comparing
  imaging with and without anisotropy and Q (Fig. 5 analogue);
- `examples/reproduce_paper.py`.

**Discrepancy register.** The loop writes one entry per discrepancy, which makes the article's
"until the discrepancies are explained" auditable. Each entry records:

- the observation;
- the hypothesis;
- the test;
- the decision;
- the model version it changed.

### WP8: Real data, CLI and documentation

**Builds:**

- LAS/SEG-Y adapters and the `iwt` command (`iwt model`, `iwt vsp`, `iwt tie`, `iwt demo`);
- the README critical assessment, assumptions, errata and the reference list.

## 5. Validation strategy

**Synthetic ground truth.** Every estimator is first run on the synthetic well with known truth.
The README then reports "recovered vs. true" for Vp drift, Q, γ, ε, δ and the AVA gradient. The
article cannot do this, because it has no ground truth.

**Quantitative acceptance criteria, defined before tuning.** The article reports its tie as
"largely achieved". Proposed defaults:

| Quantity | Default criterion |
|---|---|
| VSP first-break residual | ≤ 1 ms RMS |
| Corridor stack against synthetic | correlation ≥ 0.8 |
| Surface AVA gradient, 30–50° | within 10 % |
| Reservoir event | time shift ≤ 2 ms |

**Hold-out validation.** Calibrate with one family of data and validate on the other, for
example:

- fit on the ZO-VSP and WA-VSP, then validate on the surface AVA;
- or the reverse.

A model tuned on every data set at once is not validated by those same data sets.

**Identifiability study.** Compute the Jacobian of the reservoir AVA with respect to Vs, δ, ε, Q
and the multiple level. This shows which of the article's competing explanations the data can
actually distinguish (section 6, point 2).

**Ensembles.** Monte Carlo sampling over the WP4/WP5 posteriors gives uncertainty envelopes on the
synthetic AVA and on the tie.

## 6. Critical assessment and recommended improvements

1. **The loop is expert-driven and its stopping rule is qualitative.** "Until the predictions
   match the measurements, or until the remaining discrepancies are explained" has no misfit
   measure, no weights and no uncertainty. *Better:* treat the loop as a regularized joint
   inversion. Give each domain explicit data weights and noise models, keep the discrepancy
   register (WP7), and fix acceptance criteria before tuning (section 5).
2. **Several knobs can explain the same AVA gradient.** In Rüger's (1997) VTI approximation the
   gradient contains Δδ/2 next to the Vs contrast. Overburden Q and angle-mapping errors also
   tilt the measured gradient. Changing the reservoir Vs (Fig. 4) can therefore stand in for a δ
   contrast or an imaging error. *Better:* run the identifiability study, and test the Vs
   hypothesis directly. The candidates are:
   - flexural-mode dispersion in a slow formation;
   - Backus averaging of thin beds;
   - the δ contrast at the reservoir top.

   Shear modulus is almost insensitive to pore fluid, so a large dipole-versus-seismic Vs
   mismatch is unlikely to come from fluid effects.
3. **The γ → (ε, δ) calibration rests on six points over 640–860 m**, but is applied over a much
   larger depth range, including the strongly anisotropic shallow overburden. That is an
   extrapolation. *Better:*
   - use a Bayesian regression with cross-validation;
   - add rock-physics bounds on anisotropy in shales;
   - show uncertainty bands, which widen outside the calibration interval.
4. **The apparent Q from the ZO-VSP mixes intrinsic and stratigraphic attenuation.** Using the
   apparent Q for compensation over-corrects in cyclically layered sections (O'Doherty & Anstey,
   1971). *Better:* separate the two terms with the log-based reflectivity model (WP3).
5. **The sonic drift and Q must be consistent.** Both come from one causal attenuation mechanism
   (Kjartansson, 1979; Stewart et al., 1984). *Better:* derive the drift correction from the
   same Q that is later used in imaging. That avoids double-counting.
6. **The WA-VSP slowness method assumes lateral homogeneity.** Surface slowness is taken to equal
   slowness at depth (Gaiser, 1990). For a shallow target recorded with a 200 m exclusion zone,
   overburden heterogeneity directly biases ε and δ. *Better:* check for consistency across
   azimuths, and correct for weak lateral heterogeneity where needed (Grechka et al., arXiv
   2019).
7. **The article's Q-PSDM is real 3D/2D imaging; here it is reduced to 1D.** This keeps the Fig. 3
   experiment, but the Fig. 5 image improvement is only illustrated, not reproduced.
8. **Interbed multiples are identified but not removed.** *Better:* go from diagnosis to
   prediction, for example with Jakubowicz (1998) interbed prediction in 2D or 3D, or
   model-based subtraction guided by the 1D reflectivity model. Re-measure the AVA after
   removal.
9. **There is no blind test.** *Better:* validate the calibrated model at a second Wisting well
   that was not used in the calibration, for example Wisting Central 7324/8-1, if its data are
   released.
10. **A more modern alternative exists.** Elastic VTI full-waveform inversion of the WA-VSP,
    jointly with the surface AVA, would replace several sequential steps. It is out of scope
    here, but the forward modelling in WP2 is written so that it could serve as an FWI engine
    later.

## 7. Notes on the article (errata found so far)

- "PGS, formally Schlumberger" should read "formerly".
- "worfklow" (Conclusions), and "that that" (Results) are typos.
- Abbreviations are inconsistent: "WAVSP" (Summary, Table 1) against "WA-VSP" (text).
- The depth reference for "640 and 860 m" is not stated (see assumption 4).
- In Table 1, Qp's *calibration* source (surface seismic) is also one of its *validation*
  sources, so the validation is not independent. This supports critical-assessment point 1.

## 8. Open decisions

These are the user's decisions. The defaults in brackets apply if no answer comes:

1. **Real data.** Is there access to the Hanssen (7324/7-2) logs and VSPs, or another released
   Wisting well? [synthetic only; adapters in WP8]
2. **2D elastic FD.** Build it as a cross-check, or rely on the reflectivity method alone?
   [reflectivity; FD optional in WP2]
3. **Package name.** [`iwt`]

## 9. References

All DOIs were checked against the Scite index on 2026-10-13, except where noted.

**The article and its references**

* Cavalca, M., Fletcher, R., & Riedel, M. (2013). Q-compensation in complex media—Ray-based and wavefield extrapolation approaches. *SEG Technical Program Expanded Abstracts 2013*, 3831–3835. https://doi.org/10.1190/segam2013-1130.1
* Leaney, W. S., & Hornby, B. E. (2007). Depth-dependent anisotropy from sub-salt walkaway VSP data. *69th EAGE Conference & Exhibition*, Extended Abstracts. https://doi.org/10.3997/2214-4609.201401677
* Lewis, O. J., Smith, P., Hohner, M., Kvist-Lassen, T., Dahlhaus, L., Stuberg, L., Granli, J. R., Veire, H., & Berger, P. (2016). Integrated processing of borehole and surface seismic measurements for enhanced well tie and AVO analysis. *78th EAGE Conference & Exhibition*, 1–5. https://doi.org/10.3997/2214-4609.201601553
* Walsh, J., Sinha, B., Plona, T., & Ammerman, M. (2007). Derivation of anisotropy parameters in a shale using borehole sonic data. *SEG Technical Program Expanded Abstracts 2007*, 323–327. https://doi.org/10.1190/1.2792435

**Methods used by this plan**

* Alkhalifah, T., & Tsvankin, I. (1995). Velocity analysis for transversely isotropic media. *Geophysics*, 60(5), 1550–1566. https://doi.org/10.1190/1.1443888
* Backus, G. E. (1962). Long-wave elastic anisotropy produced by horizontal layering. *Journal of Geophysical Research*, 67(11), 4427–4440. https://doi.org/10.1029/JZ067i011p04427
* Banik, N. C., Lerche, I., & Shuey, R. T. (1985). Stratigraphic filtering, Part I: Derivation of the O'Doherty–Anstey formula. *Geophysics*, 50(12), 2768–2774. https://doi.org/10.1190/1.1441897
* Fryer, G. J., & Frazer, L. N. (1984). Seismic waves in stratified anisotropic media. *Geophysical Journal International*, 78(3), 691–710. https://doi.org/10.1111/j.1365-246X.1984.tb05065.x
* Gaiser, J. E. (1990). Transversely isotropic phase velocity analysis from slowness estimates. *Journal of Geophysical Research*, 95(B7), 11241–11254. https://doi.org/10.1029/JB095iB07p11241
* Grechka, V., Tsvankin, I., & Contreras, P. (2019). Inversion of walkaway VSP data in the presence of lateral velocity heterogeneity. arXiv:1901.02916. https://doi.org/10.48550/arXiv.1901.02916 (arXiv posting; the journal version has not been checked)
* Hauge, P. S. (1981). Measurements of attenuation from vertical seismic profiles. *Geophysics*, 46(11), 1548–1558. https://doi.org/10.1190/1.1441161
* Horne, S., Walsh, J., & Miller, D. (2012). Elastic anisotropy in the Haynesville Shale from dipole sonic data. *First Break*, 30(2). https://doi.org/10.3997/1365-2397.2011039
* Jakubowicz, H. (1998). Wave equation prediction and removal of interbed multiples. *SEG Technical Program Expanded Abstracts 1998*, 1527–1530. https://doi.org/10.1190/1.1820204
* Kennett, B. L. N., & Kerry, N. J. (1979). Seismic waves in a stratified half space. *Geophysical Journal International*, 57(3), 557–583. https://doi.org/10.1111/j.1365-246X.1979.tb06779.x
* Kjartansson, E. (1979). Constant Q-wave propagation and attenuation. *Journal of Geophysical Research*, 84(B9), 4737–4748. https://doi.org/10.1029/JB084iB09p04737
* Miller, D. E., & Spencer, C. (1994). An exact inversion for anisotropic moduli from phase slowness data. *Journal of Geophysical Research*, 99(B11), 21651–21657. https://doi.org/10.1029/94JB01848
* Norris, A. N. (1990). The speed of a tube wave. *Journal of the Acoustical Society of America*, 87(1), 414–417. https://doi.org/10.1121/1.399262
* O'Doherty, R. F., & Anstey, N. A. (1971). Reflections on amplitudes. *Geophysical Prospecting*, 19(3), 430–458. https://doi.org/10.1111/j.1365-2478.1971.tb00610.x
* Quan, Y., & Harris, J. M. (1997). Seismic attenuation tomography using the frequency shift method. *Geophysics*, 62(3), 895–905. https://doi.org/10.1190/1.1444197
* Rüger, A. (1997). P-wave reflection coefficients for transversely isotropic models with vertical and horizontal axis of symmetry. *Geophysics*, 62(3), 713–722. https://doi.org/10.1190/1.1444181
* Shuey, R. T. (1985). A simplification of the Zoeppritz equations. *Geophysics*, 50(4), 609–614. https://doi.org/10.1190/1.1441936
* Stewart, R. R., Huddleston, P. D., & Kan, T. K. (1984). Seismic versus sonic velocities: A vertical seismic profiling study. *Geophysics*, 49(8), 1153–1168. https://doi.org/10.1190/1.1441745
* Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*, 51(10), 1954–1966. https://doi.org/10.1190/1.1442051
* Tonn, R. (1991). The determination of the seismic quality factor Q from VSP data: A comparison of different computational methods. *Geophysical Prospecting*, 39(1), 1–27. https://doi.org/10.1111/j.1365-2478.1991.tb00298.x
* Virieux, J. (1986). P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901. https://doi.org/10.1190/1.1442147
* Walden, A. T., & White, R. E. (1998). Seismic wavelet estimation: A frequency domain solution to a geophysical noisy input-output problem. *IEEE Transactions on Geoscience and Remote Sensing*, 36(1), 287–297. https://doi.org/10.1109/36.655337
* Wang, Y. (2002). A stable and efficient approach of inverse Q filtering. *Geophysics*, 67(2), 657–663. https://doi.org/10.1190/1.1468627
* Wang, Y. (2006). Inverse Q-filter for seismic resolution enhancement. *Geophysics*, 71(3), V51–V60. https://doi.org/10.1190/1.2192912
* White, R. E., & Simm, R. (2003). Tutorial: Good practice in well ties. *First Break*, 21(10). https://doi.org/10.3997/1365-2397.21.10.25640

**Facts about the Hanssen well and data access (web sources)**

* OMV (2014). OMV finds hydrocarbons at Hanssen well in Norway (press release). https://www.omv.com/en/media/press-releases/2014/omv-finds-hydrocarbons-at-hanssen-well-in-norway
* Sokkeldirektoratet. Wellbore 7324/7-2 factpage. https://factpages.sodir.no/no/wellbore/PageView/With/Wdss/7450
* Sokkeldirektoratet. Access to released data. https://www.sodir.no/en/facts/data-and-analyses/release-of-data/access-to-released-data/
