# ES_PHYS_P011 – Integrated borehole and surface seismic well tie in Python

A Python implementation, in progress, of

> Lewis, O. J., Smith, P., Hohner, M., Kvist-Lassen, T., Dahlhaus, L., Stuberg, L., Granli, J. R.,
> Veire, H., & Berger, P. (2016). *Integrated processing of borehole and surface seismic
> measurements for enhanced well tie and AVO analysis.* 78th EAGE Conference & Exhibition, Vienna,
> paper Th SRS3 04. https://doi.org/10.3997/2214-4609.201601553 (PDF in [`References/`](References/))

and of the methods of its three references:

- WA-VSP anisotropy (Leaney & Hornby, 2007);
- sonic-derived γ (Walsh et al., 2007);
- ray-based Q-compensated depth imaging (Cavalca et al., 2013).

The roadmap, assumptions, critical assessment and full reference list are in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

**Status:** work package 1 (foundations) is implemented. WP2–WP8 (forward modelling, VSP
processing, sonic and walkaway anisotropy, Q-compensated imaging, the integrated loop, real-data
I/O) are still to do.

![Synthetic well](docs/figures/wp1_well.png)

## The method in one paragraph

Wisting (OMV, southwest Barents Sea) is an oil reservoir only about 250 m below the seabed. Its
first two wells would not tie to the surface seismic. At the third well, Hanssen (7324/7-2), OMV
acquired wireline sonic, a zero-offset VSP and a multi-azimuth walkaway VSP. All data were then
processed *together*:

1. **Build one model:** a single 1D elastic model with VTI anisotropy and Q, sampled every
   0.5 m, from surface to TD.
2. **Predict:** model the borehole and surface seismic responses from it.
3. **Explain each mismatch:** remove it by editing, reprocessing or updating the model.
4. **Repeat** until all data sets agree.

The tie needed two things in depth imaging: anisotropy and Q in the overburden, and a
rock-physics Vs in the reservoir. The work also showed that interbed multiples contaminate the
reservoir AVO, and the reprocessed image revealed an oil–water contact.

## Installation

```bash
pip install -e .            # core: numpy and scipy
pip install -e ".[all]"     # + matplotlib (figures) and lasio
pip install -e ".[dev]"     # + pytest and pyflakes
```

Python ≥ 3.9. CI (GitHub Actions) runs pyflakes and the tests on Python 3.9 and 3.13.

## Quick start

```python
import numpy as np
import iwt

model, logs = iwt.wisting_like_well()      # true 1D VTI + Q model and wireline logs
model.validate()                           # positive-definite stiffness, delta bound, Q > 0
model.twt(672.0)                           # two-way vertical time to top Stø (s)
model.at(650.0)                            # properties at a depth

c = model.cij()                            # C11, C13, C33, C44, C66 per layer (Pa)
blocked = model.block([600.0, 631.0, 672.0])         # Backus (1962) blocking
seismic_scale = model.backus(10.0, respect=[417.5])   # running Backus average, 10 m window
vp, vsv, vsh = iwt.phase_velocities(np.radians(30.0), 2400.0, 950.0, 0.20, 0.08, 0.24)  # exact VTI

logs.las_curves()                          # DEPT (m RKB), TVDSS, DTCO, DTSM (us/ft), RHOB (g/cm3)
```

## What WP1 provides

| Plan item | Code | Notes |
|---|---|---|
| numbers in the article, with provenance | `iwt.paper` | Table 1, angles, exclusion zone, the six WA-VSP levels, the Fig. 3 ranking; public well facts kept separate with their sources |
| 1D elastic VTI model with Q | `iwt.model.ElasticModel1D` | layers between interfaces; Thomsen parameters at reference frequency `f_ref` |
| Thomsen ↔ Cij (exact) | `thomsen_to_cij`, `cij_to_thomsen`, `delta_min` | Thomsen (1986); δ lower bound for real C13 |
| stability checks | `is_stable`, `ElasticModel1D.validate` | Voigt matrix positive definite; ideal fluid allowed |
| exact phase velocities | `phase_velocities` | qP, qSV, SH |
| Backus upscaling and blocking | `backus_average`, `.block`, `.backus` | Backus (1962); Q averaged so that vertical traveltime attenuation is preserved; windows never cross fluid–solid boundaries or depths you list |
| time–depth | `.vertical_time`, `.twt`, `.depth_at_time` | exact for the layered model |
| constant-Q dispersion | `iwt.qmodel` | Kjartansson (1979); needed to make sonic logs from the seismic-frequency model |
| Gassmann fluid substitution | `iwt.rockphysics` | derives the Stø oil leg from its water leg |
| synthetic Wisting-like well and logs | `iwt.synthetic` | see below |
| figures | `iwt.plotting`, `examples/wp1_synthetic_well.py` | `docs/figures/wp1_*.png` |

## The synthetic well

The Hanssen data are proprietary, and the article gives no velocities, densities, anisotropy or
Q values. `iwt.synthetic` therefore builds a well with the right *structure* and known true
properties.

**Public facts used:**

- water depth 417.5 m and TD 1679 m below sea level (OMV, 2014);
- top Stø 712 m RKB with RKB ≈ 40 m, i.e. 672 m TVDSS. That is 254.5 m below the seabed, against
  the article's "some 250 m" (Sodir factpage 7324/7-2);
- a 20–25 m gross oil column (OMV, 2014). Here the oil–water contact is at 694.5 m.

**Assumed:** every other top, every property value, the thin calcite-cemented stringers and the
fine-scale heterogeneity. The model is sampled at 0.5 m below a single water layer. Velocities
apply at 30 Hz.

| Unit | Top (m TVDSS) | Vp0 (m/s) | Vs0 (m/s) | ρ (kg/m³) | ε | δ | γ | Qp | Qs | Fluctuation σ / ℓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Sea water | 0 | 1470 | 0 | 1028 | 0 | 0 | 0 | ∞ | ∞ | – |
| Quaternary | 417.5 | 1750 | 450 | 1950 | 0.02 | 0.01 | 0.03 | 30 | 15 | 2 % / 1 m |
| Lower Cretaceous shale | 440 | 2400 (+2 %/100 m) | 950 | 2280 | 0.20 | 0.08 | 0.24 | 45 | 25 | 3 % / 1.5 m |
| Hekkingen | 600 | 2300 | 1050 | 2180 | 0.26 | 0.10 | 0.30 | 40 | 25 | 4 % / 1 m |
| Fuglen | 631 | 2800 | 1300 | 2450 | 0.12 | 0.05 | 0.14 | 60 | 35 | 4 % / 1 m |
| Stø (oil leg, Gassmann) | 672 | 2925 | 1667 | 2166 | 0.02 | 0.01 | 0.02 | 90 | 60 | 3 % / 1.5 m |
| Stø (water leg) | 694.5 | 3050 | 1650 | 2210 | 0.02 | 0.01 | 0.02 | 90 | 60 | 3 % / 1.5 m |
| Nordmela | 702 | 2950 | 1480 | 2350 | 0.08 | 0.03 | 0.09 | 70 | 45 | 5 % / 0.8 m |
| Fruholmen | 740 | 3350 | 1800 | 2420 | 0.05 | 0.02 | 0.06 | 80 | 50 | 4 % / 1.5 m |
| Snadd | 800 | 3150 (+1.5 %/100 m) | 1600 | 2450 | 0.10 | 0.04 | 0.12 | 70 | 40 | 4 % / 2 m |

**Stringers:** five isotropic calcite-cemented beds, 1–2 m thick (Vp 4300 m/s, Vs 2350 m/s,
ρ 2620 kg/m³). Three are in the Lower Cretaceous shale (505, 548 and 583 m) and two in the Fuglen
(648 and 661 m).

**Overburden:** the shales are strongly anisotropic with low Q. The article reports this (p. 3)
but gives no numbers.

**Stø oil leg:** derived from the water leg by Gassmann (1951) substitution, with:

- porosity 0.27;
- quartz K = 36.6 GPa (Mavko et al., 2009);
- brine 2.35 GPa / 1030 kg/m³;
- light oil 1.2 GPa / 840 kg/m³;
- Sw = 0.15.

Both legs share one fine-scale heterogeneity field, because they are one rock.

**Logs** are "measured" from the true model in a vertical well, so the monopole sonic reads Vp0
and the dipole reads Vs0:

- **Dispersion:** velocities are dispersed from 30 Hz to the tool frequency (12 kHz monopole,
  2 kHz dipole) by the constant-Q law.
- **Averaging:** slowness is averaged over the tool aperture (0.6 m P, 1.0 m S); density over
  0.45 m.
- **Noise:** 0.5 % on P slowness, 1 % on S slowness, 15 kg/m³ on density.
- **Interval:** 470–1670 m TVDSS.

## WP1 results (synthetic well; `python examples/wp1_synthetic_well.py`)

| Unit top | Depth (m TVDSS) | Vertical TWT (ms) |
|---|---|---|
| Seabed / Quaternary | 417.5 | 568.0 |
| Lower Cretaceous shale | 440.0 | 593.9 |
| Hekkingen | 600.0 | 724.1 |
| Fuglen | 631.0 | 750.8 |
| Stø | 672.0 | 779.7 |
| OWC | 694.5 | 795.2 |
| Nordmela | 702.0 | 800.1 |
| Fruholmen | 740.0 | 826.5 |
| Snadd | 800.0 | 862.2 |

**Dispersion alone produces a sonic-to-seismic drift.** Integrated sonic time (12 kHz) is shorter
than the true 30 Hz time from 470 m by:

| To | Drift | Share of the interval time |
|---|---|---|
| 600 m | 2.2 ms | 4.2 % |
| top Stø | 3.3 ms | 4.1 % |
| 1670 m | 11.3 ms | 3.0 % |

There is no anisotropy, depth or picking error in this calculation. With Q of 30–45 in the
overburden, constant-Q dispersion alone gives a drift of the size a "calibrate the sonic to the
VSP" step removes. This supports recommendation 5 of the plan: derive the drift correction from
the same Q that is later compensated in imaging, rather than correcting it empirically and then
applying Q again.

**Thin stringers change the anisotropy the seismic sees.** Running Backus averages kept inside
each unit, compared with the intrinsic 0.5 m values (interval means):

| Interval | Intrinsic ε / γ | 10 m Backus ε / δ / γ | 80 m Backus ε / δ / γ |
|---|---|---|---|
| Lower Cretaceous shale | 0.194 / 0.233 | 0.215 / 0.069 / 0.299 | 0.218 / 0.069 / 0.308 |
| Fuglen | 0.112 / 0.131 | 0.136 / 0.038 / 0.184 | 0.138 / 0.038 / 0.188 |
| Stø | 0.020 / 0.020 | 0.023 / 0.009 / 0.025 | 0.023 / 0.009 / 0.025 |
| Snadd | 0.100 / 0.120 | 0.103 / 0.039 / 0.126 | 0.104 / 0.039 / 0.128 |

Stiff stringers that make up only 3 % of the Lower Cretaceous raise the seismic-scale γ from
0.23 to 0.30. They also *lower* δ near each stringer (0.08 → about 0.07 on average). The article
calibrates the continuous sonic γ against walkaway ε and δ, which are seismic-scale estimates. A
γ measured at sonic scale and a δ measured at walkaway scale are not the same medium. The sonic
curves should therefore be Backus-upscaled to the walkaway scale *before* the γ → (ε, δ)
calibration (WP5). The article does not mention this step.

![Intrinsic and layer-induced anisotropy](docs/figures/wp1_backus.png)

## Assumptions

1. **"LFP article" means Lewis et al. (2016)**, the only article in [`References/`](References/).
2. **The model is a synthetic stand-in.** All elastic and attenuation values, and all formation
   tops other than the seabed, top Stø, the OWC and TD, are assumed (section "The synthetic
   well"). They are not Wisting measurements.
3. **Depths are TVDSS, positive down.** RKB ≈ 40 m is *derived* from two TD values (1719 m TVD
   RKB and 1679 m below sea level). Top Stø TVDSS assumes a vertical well. The public facts were
   taken from web-search quotations, because omv.com and factpages.sodir.no were not reachable
   from the build environment. The article does not state the datum of its 640–860 m walkaway
   levels.
4. **Velocities in the model apply at 30 Hz** (`f_ref`). Dispersion follows Kjartansson's (1979)
   constant-Q law between 30 Hz and the tool frequencies (12 kHz and 2 kHz, assumed).
5. **Thomsen's δ is inverted with C13 + C44 ≥ 0**, the physically usual root.
6. **Backus averaging assumes the long-wavelength limit.** The running average uses a box window.
   A window shorter than about a third of the shortest wavelength is the usual choice; 10 m is
   about a third of the wavelength at 2300 m/s and 80 Hz.
7. **Gassmann assumes** an isotropic, connected pore space at low frequency. Applying it to the
   reservoir sand only is a simplification.

## Notes found while implementing

- **The plan's WP1 check "Backus averaging of isotropic layers gives ε ≥ 0" was wrong.** The test
  suite found counter-examples, in about 10 % of random stacks. The correct statements are
  γ ≥ 0 and ε ≥ δ (η ≥ 0); ε can be negative when Vp contrasts are small and Vs contrasts large
  (Berryman et al., 1999). The plan and the tests now say so.

## Repository layout

```
iwt/
  paper.py        numbers in the article (page/table/figure) and public Hanssen facts (sourced)
  model.py        ElasticModel1D; Thomsen <-> Cij; stability; phase velocities; Backus; time-depth
  qmodel.py       constant-Q (Kjartansson) dispersion and attenuation
  rockphysics.py  Gassmann fluid substitution, Reuss fluid mixing
  synthetic.py    the Wisting-like true model and its wireline logs
  plotting.py     well and anisotropy displays (matplotlib)
examples/wp1_synthetic_well.py   prints the WP1 numbers and writes docs/figures/wp1_*.png
tests/            pytest suite (44 tests)
docs/             implementation plan and figures
References/       the article (PDF)
```

Run the tests with `pytest`. The plotting tests are skipped when matplotlib is missing.

## References

* Backus, G. E. (1962). Long-wave elastic anisotropy produced by horizontal layering. *Journal of Geophysical Research*, 67(11), 4427–4440. https://doi.org/10.1029/JZ067i011p04427
* Berryman, J. G., Grechka, V., & Berge, P. A. (1999). Analysis of Thomsen parameters for finely layered VTI media. *Geophysical Prospecting*, 47(6), 959–978. https://doi.org/10.1046/j.1365-2478.1999.00163.x
* Cavalca, M., Fletcher, R., & Riedel, M. (2013). Q-compensation in complex media—Ray-based and wavefield extrapolation approaches. *SEG Technical Program Expanded Abstracts 2013*, 3831–3835. https://doi.org/10.1190/segam2013-1130.1
* Gassmann, F. (1951). Über die Elastizität poröser Medien. *Vierteljahrsschrift der Naturforschenden Gesellschaft in Zürich*, 96, 1–23.
* Kjartansson, E. (1979). Constant Q-wave propagation and attenuation. *Journal of Geophysical Research*, 84(B9), 4737–4748. https://doi.org/10.1029/JB084iB09p04737
* Leaney, W. S., & Hornby, B. E. (2007). Depth-dependent anisotropy from sub-salt walkaway VSP data. *69th EAGE Conference & Exhibition*. https://doi.org/10.3997/2214-4609.201401677
* Lewis, O. J., et al. (2016). Integrated processing of borehole and surface seismic measurements for enhanced well tie and AVO analysis. *78th EAGE Conference & Exhibition*. https://doi.org/10.3997/2214-4609.201601553
* Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics Handbook* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511626753
* Stewart, R. R., Huddleston, P. D., & Kan, T. K. (1984). Seismic versus sonic velocities: A vertical seismic profiling study. *Geophysics*, 49(8), 1153–1168. https://doi.org/10.1190/1.1441745
* Thomsen, L. (1986). Weak elastic anisotropy. *Geophysics*, 51(10), 1954–1966. https://doi.org/10.1190/1.1442051
* Walsh, J., Sinha, B., Plona, T., & Ammerman, M. (2007). Derivation of anisotropy parameters in a shale using borehole sonic data. *SEG Technical Program Expanded Abstracts 2007*, 323–327. https://doi.org/10.1190/1.2792435
* OMV (2014). OMV finds hydrocarbons at Hanssen well in Norway (press release). https://www.omv.com/en/media/press-releases/2014/omv-finds-hydrocarbons-at-hanssen-well-in-norway
* Sokkeldirektoratet. Wellbore 7324/7-2 factpage. https://factpages.sodir.no/en/wellbore/pageview/exploration/all/7450

DOIs were checked against the Scite index. The plan lists further references for WP2–WP8.
