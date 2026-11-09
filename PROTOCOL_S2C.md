# PROTOCOL S2C (pre-registration; frozen before any Sentinel-1 composite was built or any S2C model fit)

Follows PROTOCOL_S2B, whose classifier failed the quality gate on FOREST recall (results/s2b, `eea8f16`: tree-covered recall FOREST 0.62,
PLANT 0.70, gate 0.70). Deviations are appended at the end with a date and reason, never edited in place.

## What is known when this is written (disclosed, because it is post hoc)
- S2 and S2B results were seen in aggregate (recalls, transfer check, raw and corrected shares). No per-feature importance or per-grid
  error map was looked at. This protocol is written once and run once.
- Sentinel-1 cost trials only (no labels touched): grid 03 2020, full-resolution VV+VH reads ~8.4 MB per band-scene; reading the first
  overview (20 m) of the same asset onto the Hansen grid with average resampling 1.7 MB per band-scene, 31 descending / 30 ascending items.
- Budget approved by the project owner on 2026-11-06 for this step. The S1 downloader has its own guard: **4.5 GB** of system-wide received
  bytes (cumulative over its runs). If it stops early, the analysis runs on the grid-periods done and says so.

## Data rules (Sentinel-1)
- STAC `sentinel-1-rtc` (Planetary Computer, anonymous SAS token), gamma0 terrain-corrected VV and VH, linear power.
- Periods as in S2 (`2020` and `ref` = label_year; one composite serves both when label_year is 2020).
- Scenes: the orbit direction with more items in the period (ties: descending); from those, 6 picked evenly spaced in date
  (indices round(linspace(0, n-1, 6)), all if n < 6). One direction only, so look geometry does not mix.
- Each scene's first overview (20 m) is warped straight onto the Hansen grid (EPSG:4326, 0.00025 deg) with average resampling; 0 = no data.
- Per cell over the chosen scenes, in dB (10 log10): median VV, median VH, median (VH - VV), temporal std of VV and of VH.
  A cell is S1-valid with >= 3 non-zero observations; otherwise its S1 features are NaN (LightGBM treats them as missing).
- The analysed population is unchanged: S2-valid cells, exactly as in S2/S2B.

## Change from S2B: features only
Feature set `s2c` = the 41 `s2b` features + 13 S1 features:
VV_med, VH_med, RATIO_med, VV_tstd, VH_tstd, and for VH_med and RATIO_med the window mean at radius 3 and 7 and window std at radius 3 and 7.
Why: C-band VH backscatter and its ratio to VV respond to canopy structure (palm fronds and regular stands vs multi-layer forest) and are
cloud-independent; the S2B failure was on recognising natural forest, which optical annual composites confuse with mature plantation.
Same folds (seed 0), LightGBM hyperparameters, sub-sampling, training population, correction and bootstrap.

## Decision rules (unchanged from PROTOCOL_S2)
- Transfer rule: 2020 application *unreliable* if balanced accuracy at 2020 on the 7 grids with label_year <= 2021 is more than 0.10 below
  the out-of-fold balanced accuracy at ref year on all grids.
- Quality gate: out-of-fold recall of FOREST and of PLANT on tree-covered cells both >= 0.70; otherwise corrected shares are not reported.
- Headline: only if the gate passes and the corrected PLANT share of Hansen 2021-24 loss has a 95 % grid-cluster interval excluding 0.20.
- Reported as well, descriptive: S2, S2B and S2C recalls side by side.
- Risk named in advance: Sentinel-1B failed in Dec 2021, so 2022+ periods have fewer scenes than 2020; the temporal check covers this only
  partly (its 7 grids have label_year <= 2021).

## Reading fixed in advance
- Gate passes: S2C supplies E2/E3; S2 and S2B stay in the record as failed attempts (three looks in total, all disclosed).
- Gate fails: annual optical plus C-band radar composites at 30 m do not separate forest from plantation well enough against this reference
  (overall accuracy 0.83 per its authors). No further feature protocol is planned; the raw shares stand as the answer, labelled raw.

## Outputs
`D:/Work/eudr-hansen-audit-refdata/s1/grid_XX_{2020,ref}.npz` + `manifest.json`; `results/s2c/landuse.json`, `decompose.json`,
`pred_2020.npz`, `run.log`.

## Deviations
