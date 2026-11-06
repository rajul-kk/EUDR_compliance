# PROTOCOL S2B (pre-registration; frozen before any S2B feature was computed or any S2B model fit)

Follows PROTOCOL_S2.md, whose classifier failed its quality gate (results/s2, commit `9af6fe9`: out-of-fold recall on tree-covered
cells FOREST 0.62, PLANT 0.63, gate 0.70). Deviations are appended at the end with a date and reason, never edited in place.

## What is known when this is written (disclosed, because it is post hoc)
- S2 results were seen in aggregate: the gate failure, the transfer check (passed), raw and corrected shares. No per-feature importance,
  per-grid error map or alternative feature set was looked at. This protocol is written once and run once.
- The decision rules below are copied from PROTOCOL_S2 unchanged, so a changed verdict can only come from the features.
- Sentinel-1 RTC was trialled for cost only (no labels touched): grid 03, 2020, 8 scenes x VV+VH = 135 MB system-wide (0.135 GB counted
  against the download budget; cumulative now ~7.77 GB of the approved 9 GB). Projected 2-5 GB for all grids even at a reduced overview.
  That exceeds the approved budget, so radar is **not** part of S2B (see "Next step if the gate fails").

## Change from S2: features only
Same composites (`D:/Work/eudr-hansen-audit-refdata/s2/`, no new download), same cells, folds (seed 0), LightGBM hyperparameters,
sub-sampling, training population, transfer rule, quality gate, correction, bootstrap and headline rule.

Feature set `s2b` (41 per cell; NaN-aware, invalid cells stay NaN in every feature, as in S2):
- 9 per-cell S2 features: B02 B03 B04 B08 B11 B12 NDVI NDMI NDVI_STD.
- 2 added indices: NBR = nd(B08, B12), NBR2 = nd(B11, B12).
- Spatial context for C in {NDVI, NDMI, NBR, B08, B11, NDVI_STD}: window mean at radius 1, 3, 7 cells (3x3, 7x7, 15x15; ~90, 210,
  450 m) and window standard deviation at radius 3 and 7 (6 x 5 = 30).

Why these: plantations are managed blocks, hundreds of metres across, with uniform canopy and sharp edges; natural forest is heterogeneous
at the same scale. Per-cell spectra at 30 m (S2) see a crown mosaic that looks alike in both. Context mean and spread at block scale is the
cheapest signal for that difference and needs no new data. NBR/NBR2 add SWIR moisture contrast that separates young stands and clearings.
Rejected: CNN on the 10 m composite (raw 10 m pixels were not kept; CPU only); seasonal composites (4x download, out of budget);
hyperparameter search (forbidden by S2 and here).

## Decision rules (unchanged from PROTOCOL_S2)
- Transfer rule: 2020 application labelled *unreliable* if balanced accuracy at 2020 on the 7 grids with label_year <= 2021 is more than
  0.10 below the out-of-fold balanced accuracy at ref year on all grids.
- Quality gate: out-of-fold recall of FOREST and of PLANT on tree-covered cells (y_H in {1,2}) both >= 0.70; otherwise E2/E3 corrected
  shares are not reported (raw shares only, labelled raw).
- Headline: only if the gate passes and the corrected PLANT share of Hansen 2021-24 loss has a 95 % grid-cluster interval excluding 0.20.
- Reported as well: S2B and S2 recall per class on tree-covered cells side by side (point estimates and their own intervals).
  Descriptive; it does not change any verdict.

## Reading fixed in advance
- Gate passes: S2B replaces S2 for E2/E3; S2 stays in the record as the failed first attempt.
- Gate fails: the conclusion is that annual 30 m Sentinel-2 composites, with or without block-scale context, cannot separate forest
  from plantation well enough against this reference (whose own overall accuracy is 0.83, so measured recall is capped below 1).
  Corrected shares stay unreported.

## Next step if the gate fails (not part of S2B; needs the project owner's budget approval)
Sentinel-1 RTC annual VV/VH statistics as extra features, as a separate protocol (S2C), with its own budget line.

## Outputs
`results/s2b/landuse.json`, `decompose.json`, `pred_2020.npz`, `run.log`; command logged in `run.log`.

## Deviations
