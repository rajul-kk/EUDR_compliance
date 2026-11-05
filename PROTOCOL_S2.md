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
