# PROTOCOL (pre-registration, frozen before any noise/model result was computed)

Status at freezing: data inventory, scoop check and design.md exist; no transition matrix, model, or farm-level rate has been computed.
Any later deviation is appended under "Deviations" with a date and reason, never edited in place.

## Framing
Reference labels are weak for this question: (a) the only high-quality reference found (Wafiq et al., Zenodo 23874371) is a
2019/2020 land-use snapshot, over 72 grid cells that may not contain the user's farms; (b) the other candidate (Zenodo 17768444 oil palm maps) has 60-71 % overall accuracy;
(c) no post-2020 ground truth exists. Therefore the study is a **sensitivity analysis with one identifiable component**:
false flags caused by land that was already plantation in the reference snapshot. Missed flags are not identifiable
from this reference except as baseline omission. Claims are limited accordingly.

## Definitions
- Hansen label y_H in {0,1,2} as in docs/design.md. Binary task restricted to pixels with y_H in {1,2}: positive = loss (2).
- Reference class groups (from class_ID): FOREST = {9 Forest, 12 Mangrove}; PLANT = {1,2 oil palm, 4 cacao, 5 coconut, 7 rubber, 8 other agri, 6 rice};
  OTHER = everything else (shrub 10, grass 14, wetland 11, bare 15, built 3, water 13, unknown 0, other 16). Shrubland/grassland deliberately not FOREST.
- Hansen label for reference cells is re-derived with the identical recipe (treecover2000 >= 10, lossyear 21-24, GFC-2025-v1.13), at 30 m; reference polygons are rasterised to the same grid by majority class of polygon area within the 30 m cell, cells with < 50 % annotated coverage dropped.
- Pixel false-flag: y_H = 2 and reference in PLANT. Pixel baseline omission: y_H = 0 and reference in FOREST.
- Farm flag (compliance decision "non-compliant / needs review"): >= 0.5 ha of loss, i.e. >= 6 loss cells at 30 m (900 m2 each, 5,400 m2). Sensitivity thresholds: 1 cell, 3 cells, 12 cells (~1 ha).
- Farm false-flag rate = P(farm flagged | farm truly has < threshold area of loss on reference-FOREST land). Farm missed-flag rate (only as sensitivity) = P(not flagged | true loss >= threshold) under assumed Hansen cell recall r in {0.5,0.7,0.9}; never reported as an observed quantity.

## Transition matrices
1. Reference-based: 3x3 matrix P(y_H = j | ref = i) and the "precision" matrix P(ref = i | y_H = j) on the cells in reference grids; 95 % intervals by cluster bootstrap over grid cells (6x6 km), 2000 resamples. Grids split by provinces; report pooled and Riau/Sumatra separately. 
2. Anchor-point (Patrini et al. 2017) estimate of the binary T on the trained baseline model's softmax over user farms: anchor = 97th percentile of p(class | x) per class; labelled as secondary, with its assumption stated.
3. Transport rule to the user's farms (assumption, flagged as such): apply the Indonesia-reference precision of Hansen loss to ID oil palm farms; no transport to other countries/crops (they get no corrected rates).

## Models and training (CPU run + Kaggle)
- Input: per 30 m cell, SCL-masked block means of R,G,B,NIR,NDVI for 2020 and 2024 and their difference (15 channels) over a 3x3 neighbourhood. Cells with > 50 % cloud/shadow/no-data (SCL in {0,1,3,8,9,10}) in either year are ignored.
- Network: 3 conv layers (width 32), global pool, linear; ~20k parameters. Adam, lr 1e-3, 15 epochs, batch 256, 5 seeds, early stopping on a farm-held-out validation split by noisy loss only.
- Losses compared: CE (baseline); GCE q = 0.7 (primary robust loss: bounded, no T needed, simple; Zhang & Sabuncu 2018); forward-corrected CE using T from step 1 above (applied to Indonesia farms) and T from step 2 (all); symmetric CE (alpha 0.1, beta 1.0) as a secondary comparison. No other losses will be added post hoc.
- Splits: grouped by farm, 60/20/20 (train/val/test), seed 0; plus leave-one-group-out by country|crop for the transfer check. Test labels stay noisy Hansen labels, so test numbers are *agreement with Hansen*, not accuracy.
- Semi-synthetic validation (the only place the robust claim can be tested): inject class-conditional noise with known T* (loss->stable at rates 0.1/0.3, stable->loss at 0.05/0.15) into the training labels of ID oil palm farms and evaluate against the unperturbed Hansen labels on test farms. Primary metric: test F1 of loss vs injected-clean label; estimator check: max-abs error of T-hat vs T*.
- Success criteria set in advance: GCE or forward beats CE by >= 0.02 F1 with a paired farm-bootstrap 95 % interval excluding 0 at noise rate 0.3; otherwise reported as no improvement. T estimator counted as working if max-abs error < 0.05.

## Per-farm uncertainty
- Ensemble: 5 seeds; farm loss area = sum of cell probabilities; uncertainty = ensemble std of farm loss area and the fraction of members that flag the farm.
- Split conformal on the farm-level score |area_pred - area_Hansen| on a calibration split of farms, 90 % level; documented limitation: coverage holds for the noisy Hansen label, not for truth. Decisions with intervals straddling the threshold are labelled "uncertain" and reported as a third outcome, with their share.

## Metrics and intervals
- Primary: reference-based pixel false-flag share P(ref in PLANT | y_H = 2) and baseline omission P(y_H = 0 | ref = FOREST); farm-level false-flag rates under the flag thresholds above; cluster bootstrap (grid cells for reference, farms for model/user data), 2000 resamples, percentile 95 %.
- Secondary: model-vs-Hansen farm flag agreement (Cohen kappa, flag-rate difference), share of uncertain farms.
- Not reported: accuracy (class imbalance), any "true" missed-flag rate.

## Stopping/reporting rules
- Every number in the report must come from a run with a logged command and a saved results file in `results/`. Nothing from the GPU notebook is reported unless the user returns its output.
- If reference cells contain fewer than 5,000 FOREST+loss cells in total, or the reference download fails, the reference-based analysis is dropped and stated as unrun.

## Deviations
(none yet)
