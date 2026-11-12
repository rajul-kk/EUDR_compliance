# Results (all numbers from runs logged in results/; CPU only; protocol deviations in PROTOCOL.md)

## 1. Reference vs Hansen (72 Indonesian grids; docs/reference_labels.md)
Run: `analyse_reference.py` -> results/reference_metrics.json. Composition of Hansen 2021-24 loss cells by reference land use:
palm/crop 0.51 (95 % 0.42-0.60), forest/mangrove 0.15 (0.10-0.21), other 0.34. In the 7 grids with reference year <= 2021: palm/crop 0.61 (0.50-0.76).
Hansen "stable forest" cells on palm/crop: 0.48 (0.38-0.59). Reference-FOREST cells labelled class 0 by Hansen: 0.32 (0.25-0.41).
Caveat: reference years are mostly 2023-24, so palm under a loss cell can be replanting (false flag) or conversion (true violation); not separable.

## 2. Transported farm-level sensitivity (94 ID oil palm farms, 26 flagged at >= 6 loss cells in the usable footprint) -> results/sensitivity.json
Expected share of Hansen-flagged farms that are false flags if per-cell loss precision is p (cells independent, optimistic): p=0.9: 1.8 %, 0.7: 4.2 %, 0.5: 10 %, 0.3: 26 %, 0.2: 51 %.
Anchored at reference precision 1 - P(palm/crop | loss): all grids p=0.49 -> 10.5 % (farm-bootstrap 2.6-20 %; reference-interval edges 6.9-16 %); <=2021 grids p=0.39 -> 16.6 % (6.4-29 %; edges 10-39 %).
This is a sensitivity analysis, not a measured false-flag rate: no reference grid contains a user farm, and no post-2020 truth exists. Missed-flag rate is not estimated (not identifiable).

## 3. Noise-robust training (patch CNN, 5 seeds, 15 epochs; results/experiments.json)
- Real labels, 227 farms (136/45/46 split), agreement with Hansen on 46 test farms (1,339 cells): F1(loss) CE 0.888+-0.008, GCE 0.893+-0.005, SCE 0.902+-0.010. This is agreement with the noisy label, not accuracy; it cannot show robustness.
- Semi-synthetic noise on ID oil palm (21/7/7 farms, 213 test cells; clean = Hansen labels): F1 vs clean, mean of 5 seeds:
  mild noise (stable->loss 5 %, loss->stable 10 %): CE 0.824, GCE 0.847, SCE 0.851, forward(oracle T) 0.852, forward(anchor T-hat) 0.676. Paired farm-bootstrap diff vs CE: GCE +0.021 [0.002, 0.045], forward-oracle +0.025 [0.002, 0.054], anchor-forward -0.186 [-0.529, -0.024].
  severe noise (15 %, 30 %): CE 0.819, GCE 0.788, SCE 0.792, forward(oracle) 0.834, forward(anchor) 0.654. Diff vs CE: GCE -0.033 [-0.096, 0.005], forward-oracle +0.012 [0.001, 0.026], anchor-forward -0.207 [-0.559, -0.041].
- Pre-registered criteria: (a) GCE or forward beats CE by >= 0.02 F1 with interval excluding 0 at severe noise: **not met** (GCE worse; oracle forward +0.012 only). (b) anchor-point T estimator max-abs error < 0.05: **not met** (0.50 mild, 0.36 severe; the anchor assumption fails with so little data and a 75 % majority class). Anchor-based forward correction hurt.
- Sample is tiny (7 test farms); intervals resample farms; do not generalise.

## 4. Per-farm uncertainty
Experiment C ran but is degenerate: 7 evaluation farms, 3 calibration farms, so the split-conformal half-width is infinite (needs >= 9 calibration farms at 90 %) and every test farm is "uncertain"; ensemble std was 0.0 cells in the test farms (members agree). Implemented and unit-tested (coverage on synthetic data), **not** informative on real data here. Needs the full set of farms (labels exist for 996) with imagery, which this folder does not have.

## 5. Not run
GPU Siamese DeepLab with robust loss; Kaggle notebook (notebooks/kaggle_hansen_audit.ipynb) re-runs the CPU pipeline with 20 seeds and was not executed. Experiment D key in experiments.json is discarded (PROTOCOL deviation 1).

## 6. Land use at the 2020 cut-off under Hansen loss (Sentinel-2 + Sentinel-1; REPORT_S2.md)
Addresses the caveat in section 1. Third registered attempt (S2C) passed the quality gate: of Hansen 2021-24 loss cells in the 72 grids,
0.49 [0.30-0.65] were plantation in 2020 (not EUDR deforestation), 0.31 [0.15-0.50] forest, 0.21 other (confusion-corrected, grid bootstrap).
The <= 2021 grids give 0.61, matching the reference. First two attempts (optical only) failed the gate. Details and caveats in REPORT_S2.md.

## 7. Farm numbers inside the OpenStreetMap outlines (PROTOCOL_E5; results/e5/farms.json)
Run: `e5_outlines.py fetch` (Overpass mirror, 92 of 94 objects returned; 2 ways no longer exist) then `e5_outlines.py run`, same
cells and S2C predictions as E4, no refit. 83 farms included; 9 outlines are smaller than one 30 m cell (0.03-0.08 ha) and have no cell.
- The outlines are small: median 1.7 ha (quartiles 0.4-5.3, max 170), 29 under 0.5 ha, 57 under 4 ha, against about 120 ha per E4 box.
  Cell counts agree with outline areas (median ratio 1.01). No cell is inside two outlines.
- Flags: **4 of 83 farms flagged inside the outline, against 69 of the same 83 in the box.** 65 farms lose their flag; none gains one.
  All 4 are plantation-driven by the E4 rule (loss cells inside outlines by predicted 2020 class: FOREST 5, PLANT 46, OTHER 38 over all farms).
- So the E4 box numbers (79 of 94 flagged, 46 plantation-driven) describe the land around the farms more than the farms. The E4
  question "why is so little of the stable forest predicted PLANT" cannot be answered here: only 22 stable-forest cells fall inside
  outlines (0 predicted PLANT). By the pre-registered reading that is "under-calls palm", but 22 cells carry no weight.
- Limits: OSM outlines are volunteered; a 0.04 ha "oil palm farm" is more likely a mapping artefact than a plot. With 29 outlines under
  the 0.5 ha flag threshold, those farms cannot be flagged at all. Predictions remain transported (E3 would supply a reference).
