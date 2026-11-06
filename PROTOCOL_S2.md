# PROTOCOL S2 (pre-registration for the Sentinel-2 extension; frozen before any Sentinel-2 download or result)

Extends PROTOCOL.md; its definitions (Hansen label y_H, FOREST / PLANT / OTHER groups, flag thresholds) are used unchanged.
Design and rejected alternatives: docs/s2_design.md. Deviations are appended at the end with a date and reason, never edited in place.

## Data rules
- Grids: all 72 reference grids. Periods: `2020` = 2020-01-01..2020-12-31 (EUDR cut-off year) and `ref` = the grid's label_year (calendar year).
  For the one grid with label_year 2020 the same composite serves both.
- Scenes: STAC `sentinel-2-l2a`, tile cloud cover < 80 %, ranked by clear fraction inside the grid (SCL not in {0,1,3,8,9,10}); median of the best 6.
  DN - 1000 when processing baseline >= 04.00; reflectance = DN / 10000.
- A cell is valid in a period if it has >= 2 clear observations at >= 50 % of its 10 m pixels. Invalid cells are dropped and their share reported per grid.
- Download stops at 4.5 GB (system-wide received bytes since start, conservative). If it stops early, the analysis runs on the grids completed and says so.

## E1 Land-use classifier (measured quality)
- LightGBM 3-class (FOREST, PLANT, OTHER), features as in docs/s2_design.md, fixed hyperparameters: 400 trees, lr 0.05, 63 leaves,
  min_child_samples 200, class-balanced weights, seed 0. No tuning on any held-out grid.
- Training cells: annotated (cov >= 0.5), valid at ref period; sub-sampled to <= 20,000 cells per grid (seed 0) for memory.
- Grouped 6-fold CV by grid, folds balanced by island (seed 0). Every grid gets out-of-fold predictions.
- Reported: out-of-fold confusion matrix and per-class recall/precision on (a) all annotated cells and (b) tree-covered cells (y_H in {1,2}),
  with 95 % grid-cluster bootstrap intervals (2000 resamples).
- Temporal check: on the grids with label_year <= 2021 (7), out-of-fold balanced accuracy at the 2020 composite vs the reference.
  **Transfer rule (set in advance):** if that balanced accuracy is more than 0.10 below the out-of-fold balanced accuracy at ref year on all grids,
  the 2020 application is labelled *unreliable* and E2 is reported as exploratory only.
- Quality gate: if out-of-fold recall of FOREST or PLANT on tree-covered cells is < 0.70, E2 corrected shares are not reported (raw shares only, labelled as such).

## E2 Decomposition of Hansen 2021-24 loss by 2020 land use (primary)
- Population: cells with y_H = 2 that are valid in 2020, across all grids.
- Raw shares: fraction predicted (argmax, out-of-fold model) FOREST / PLANT / OTHER in 2020.
- Corrected shares: pi = solve(C^T, q), where q = raw predicted shares and C[i, j] = P(pred j | true i) on tree-covered cells at ref year (out-of-fold);
  negative components clipped to 0 and renormalised (reported if it happens). C is re-estimated inside each bootstrap resample.
- Interpretation fixed in advance: share PLANT-in-2020 = **upper part of the false-flag share** (replanting or harvest on land that was not forest at the cut-off);
  share FOREST-in-2020 = candidate deforestation; OTHER-in-2020 = cleared before the cut-off or non-forest, also not EUDR deforestation.
- Breakdowns: by island, and for the 7 <= 2021 grids, compared with the reference-based P(PLANT | loss) = 0.61 from docs/reference_labels.md (agreement check only).
- **Headline claim allowed only if** the corrected PLANT share's 95 % interval excludes 0.20. Otherwise reported as "not distinguishable from a small share".

## E3 Baseline commission dated to 2020
- Population: y_H = 1 cells valid in 2020. Report the share predicted PLANT in 2020, raw and corrected (same method). This dates the
  earlier 0.48 (2023-24 reference) to the cut-off year.

## E4 Optional, descriptive only
- Only if >= 1.5 GB of the download budget is left after E1-E3 data: the same 2020 composite and classifier applied to the user's
  94 Indonesian oil palm farms (1.1 km footprint as before) to give a farm-level share of flags driven by 2020 plantation (farm flagged at
  >= 6 loss cells while < 6 loss cells are predicted FOREST in 2020), with farm bootstrap. This is transported: no reference covers those farms.
- Malaysia: not run in this extension; no reference exists there, so it would be model output only. Recorded as future work.

## Not done
- No new noise-robust-loss experiments (the first audit's criteria stand). No hyperparameter search. No GPU.
- Every reported number comes from a logged command with a results file in `results/s2/`.

## Deviations
1. 2026-11-05, before any result (only download cost measured). A trial on grid 03 (2020) took 44 MB and 80 s. Ranking all 30 candidate scenes by SCL
   dominated the cost, which projects to ~6.3 GB for 143 grid-periods, above the 4.5 GB guard. Change: only the 12 items with the lowest tile cloud cover
   are ranked by in-grid clear fraction (items are pre-sorted by `eo:cloud_cover`); still the best 6 of those are composited. The grid 03 trial file
   (made with all 30 candidates) is deleted and rebuilt under the new rule so every grid follows one rule.
2. 2026-11-05, before any result. After deviation 1, trials on grids 03 and 04 (three grid-periods) still used 37-42 MB each, mostly band reads
   (~0.9 MB per band per scene), projecting to ~5.6 GB. Change: the composite uses the best **4** scenes (was 6). In the trials the 4th-best scene
   still had >= 0.83 in-grid clear fraction, and the >= 2 clear observations rule is unchanged. Trial files are rebuilt under the final rule.
3. 2026-11-06, before any result (no land-use model has been fit; only download cost seen). The first two download runs stopped at the 4.5 GB
   guard with 73 of 144 grid-periods done. The guard counts all traffic on the machine, so it over-counts: the median grid-period cost 44 MB, the
   mean 62 MB. Stopping there would leave a subset chosen by download order. Change, with the project owner's approval: the guard is raised to
   6 GB (`--budget-gb 6`). At ~44-62 MB per remaining grid-period this is expected to cover roughly 25-35 more, not all 71; finishing would need
   about 9 GB on the same counter, and any further raise is logged as its own deviation. The composite rule (deviations 1-2) is unchanged.
4. 2026-11-06, before any result. With the project owner's approval the guard is raised again, to 9 GB (`--budget-gb 9`), so that all 144
   grid-periods can finish and the analysed set does not depend on download order. The 6 GB run (deviation 3) is allowed to reach its own guard
   and is then resumed under the new limit, so the byte count stays continuous. Composite rule unchanged.
5. 2026-11-07, before any farm data was downloaded or any farm result computed. E4 (farm application) is run with the owner's approval,
   replacing its ">= 1.5 GB left" condition, and uses the S2C classifier because only S2C passed its quality gate (results/s2c). Rules:
   - Farms: the 94 Indonesian oil palm farms of `results/features.npz` (as in E4). Footprint: the farm image box (0.01 deg). Farms are
     grouped greedily (input order) into windows whose union box is <= 0.05 deg per side; each window gets a 0.00025 deg grid from its
     (west, north) corner, and a farm's cells are the window cells whose centres fall inside its box (overlapping farms share cells).
   - Hansen label: the farm's label raster resampled nearest onto the window grid, as in the first audit; 2 = loss.
   - 2020 composites: S2 rule as in S2 deviations 1-2 (12 candidates, best 4) and the S1 rule of PROTOCOL_S2C, period 2020 only.
     Own byte guard 2 GB (trial on one farm: 24 MB S2 + 22 MB S1; ~1.3 GB projected for 19 windows).
   - Classifier: S2C features and LightGBM settings, fit once on all 72 grids (annotated, valid at ref year, <= 20,000 cells per grid,
     seed 0). No CV, no tuning; its out-of-fold quality is the S2C result.
   - Reported (descriptive, transported; no reference covers these farms): per farm, Hansen loss cells and how many are predicted FOREST /
     PLANT / OTHER in 2020 (S2-valid cells only). A farm is Hansen-flagged at >= 6 loss cells; its flag is *plantation-driven* if fewer than
     6 of its loss cells are predicted FOREST in 2020. Share of flagged farms that are plantation-driven, with a farm bootstrap
     (2000 resamples, seed 0). Raw predictions only: the per-cell correction does not transfer to a farm-level threshold rule.
