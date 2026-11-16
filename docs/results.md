# Results (all numbers from runs logged in results/; CPU only; protocol deviations in PROTOCOL.md)

## 1. Reference vs Hansen (72 Indonesian grids; docs/reference_labels.md)
Run: `analyse_reference.py` -> results/reference_metrics.json. Composition of Hansen 2021-24 loss cells by reference land use:
palm/crop 0.51 (95 % 0.42-0.60), forest/mangrove 0.15 (0.10-0.21), other 0.34. In the 7 grids with reference year <= 2021: palm/crop 0.61 (0.50-0.76).
Hansen "stable forest" cells on palm/crop: 0.48 (0.38-0.59). Reference-FOREST cells labelled class 0 by Hansen: 0.32 (0.25-0.41).
Caveat: reference years are mostly 2023-24, so palm under a loss cell can be replanting (false flag) or conversion (true violation); not separable.

## 2. Transported farm-level sensitivity (94 ID oil palm farms, 26 flagged at >= 6 loss cells in the usable footprint) -> results/sensitivity.json
*15 Nov 2026: the usable footprint is Hansen class 1 or 2, and class 1 was nearly empty (section 11). Rerun pending.*
Expected share of Hansen-flagged farms that are false flags if per-cell loss precision is p (cells independent, optimistic): p=0.9: 1.8 %, 0.7: 4.2 %, 0.5: 10 %, 0.3: 26 %, 0.2: 51 %.
Anchored at reference precision 1 - P(palm/crop | loss): all grids p=0.49 -> 10.5 % (farm-bootstrap 2.6-20 %; reference-interval edges 6.9-16 %); <=2021 grids p=0.39 -> 16.6 % (6.4-29 %; edges 10-39 %).
This is a sensitivity analysis, not a measured false-flag rate: no reference grid contains a user farm, and no post-2020 truth exists. Missed-flag rate is not estimated (not identifiable).

## 3. Noise-robust training (patch CNN, 5 seeds, 15 epochs; results/experiments.json)
*15 Nov 2026: void as a test of loss against stable forest. Class 1 held only 2025 loss (section 11), so the task here was "lost 2021-24" against "lost 2025". Rerun pending.*
- Real labels, 227 farms (136/45/46 split), agreement with Hansen on 46 test farms (1,339 cells): F1(loss) CE 0.888+-0.008, GCE 0.893+-0.005, SCE 0.902+-0.010. This is agreement with the noisy label, not accuracy; it cannot show robustness.
- Semi-synthetic noise on ID oil palm (21/7/7 farms, 213 test cells; clean = Hansen labels): F1 vs clean, mean of 5 seeds:
  mild noise (stable->loss 5 %, loss->stable 10 %): CE 0.824, GCE 0.847, SCE 0.851, forward(oracle T) 0.852, forward(anchor T-hat) 0.676. Paired farm-bootstrap diff vs CE: GCE +0.021 [0.002, 0.045], forward-oracle +0.025 [0.002, 0.054], anchor-forward -0.186 [-0.529, -0.024].
  severe noise (15 %, 30 %): CE 0.819, GCE 0.788, SCE 0.792, forward(oracle) 0.834, forward(anchor) 0.654. Diff vs CE: GCE -0.033 [-0.096, 0.005], forward-oracle +0.012 [0.001, 0.026], anchor-forward -0.207 [-0.559, -0.041].
- Pre-registered criteria: (a) GCE or forward beats CE by >= 0.02 F1 with interval excluding 0 at severe noise: **not met** (GCE worse; oracle forward +0.012 only). (b) anchor-point T estimator max-abs error < 0.05: **not met** (0.50 mild, 0.36 severe; the anchor assumption fails with so little data and a 75 % majority class). Anchor-based forward correction hurt.
- Sample is tiny (7 test farms); intervals resample farms; do not generalise.
- 20-seed rerun (14 Nov, the notebook's settings run locally; results/experiments_20seeds.json, reference 'none' so no experiment D).
  Same data and splits; the 5-seed run above stays the registered result. Severe noise, diff vs CE: GCE -0.023 [-0.071, 0.003],
  SCE -0.049 [-0.135, 0.004], forward(oracle T) +0.023 [0.004, 0.049], forward(anchor T-hat) -0.189 [-0.534, -0.031]. With 20 seeds the
  oracle-T forward correction clears criterion (a)'s numbers, but it is given the true noise matrix; the estimable version still hurts,
  and GCE is still no better than CE. The reading is unchanged; the intervals still resample only 7 test farms.

## 4. Per-farm uncertainty
*15 Nov 2026: experiment C used the same class 1/2 task; void until rerun (section 11).*
(Rerun on all farms in section 8: finite but wide intervals.) Experiment C ran but is degenerate: 7 evaluation farms, 3 calibration farms, so the split-conformal half-width is infinite (needs >= 9 calibration farms at 90 %) and every test farm is "uncertain"; ensemble std was 0.0 cells in the test farms (members agree). Implemented and unit-tested (coverage on synthetic data), **not** informative on real data here. Needs the full set of farms (labels exist for 996) with imagery, which this folder does not have.

## 5. Not run
GPU Siamese DeepLab with robust loss; Kaggle notebook (notebooks/kaggle_hansen_audit.ipynb) re-runs the CPU pipeline with 20 seeds and was not executed. Experiment D key in experiments.json is discarded (PROTOCOL deviation 1).

## 6. Land use at the 2020 cut-off under Hansen loss (Sentinel-2 + Sentinel-1; REPORT_S2.md)
Addresses the caveat in section 1. Third registered attempt (S2C) passed the quality gate: of Hansen 2021-24 loss cells in the 72 grids,
0.49 [0.30-0.65] were plantation in 2020 (not EUDR deforestation), 0.31 [0.15-0.50] forest, 0.21 other (confusion-corrected, grid bootstrap).
The <= 2021 grids give 0.61, matching the reference. First two attempts (optical only) failed the gate. Details and caveats in REPORT_S2.md.

## 7. Farm numbers inside the OpenStreetMap outlines (PROTOCOL_E5; results/e5/farms.json)
*15 Nov 2026: flag counts stand (class 2 only); the transport check below used class 1, which held 2025-loss cells (section 11). Rerun pending.*
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

## 8. The full set of farms (PROTOCOL_E6; results/e6/experiments.json)
*15 Nov 2026: void until rerun. The cells, the task and the usable-farm count all rest on class 1, which held only 2025 loss (section 11).*
The loader skipped 670 undated files per year (median composites of up to 3 scenes, `src/sentinel_client.py`), so every analysis
above used 415 farms. Read as well (opt-in, `build_features.py --composites`), all 996 labelled farms have both years; 586 have
usable class 1/2 cells (227 single-scene, 359 composite; 25,844 cells, loss 0.76). Same model, 15 epochs, 5 seeds; agreement
with Hansen, not accuracy. Run: `run_e6.py results/e6_features.npz results/e6/experiments.json` (461 s, CPU).
- Experiment C, registered code, 111 ID oil palm farms: 23 evaluation farms, 11 calibration, 12 test. The conformal half-width
  is now finite but wide: 33.2 cells (the flag threshold is 6). Model and Hansen flag the same 10 of 12 test farms (agreement
  1.0); 3 are flagged with the interval entirely above 6, 9 are "uncertain", none is cleared. Ensemble spread 4.0 cells. The
  experiment now runs as designed, and its answer is that the per-farm interval around a Hansen-trained count is too wide to
  clear or confirm most farms.
- Leave-one-group-out (11 groups with >= 20 farms), F1 of loss in-group vs held-out, gap with farm-bootstrap 95 % interval.
  Does not transfer (gap >= 0.05, interval above 0): VN Coffee 0.947 vs 0.404 (gap 0.54 [0.41, 0.67]; balanced accuracy 0.96 vs
  0.43), GH Cocoa 0.860 vs 0.691 (0.17 [0.07, 0.28]), BR Cattle 0.967 vs 0.816 (0.15 [0.07, 0.25]). Transfers: ID Oil Palm (gap
  0.017 [0.007, 0.028]), CM Cocoa, IN Rubber, ID Coffee, CI Cocoa, BR Coffee, TH Rubber, KE Coffee (all gaps under 0.05).
  Read "transfers" with the base rate beside it: loss is 0.72-0.89 of the cells in most of these groups, and balanced accuracy
  is 0.50-0.67 in five of them (CI Cocoa 0.50: the model calls every cell loss). A model that predicts loss everywhere
  "transfers" by this rule. The clear failure is VN Coffee, the one large group where loss is the minority (0.29).
- Image type (groups with >= 10 farms of each): ID Oil Palm scene 0.899 vs composite 0.875 (difference 0.025 [-0.036, 0.081]), CM
  Cocoa 0.821 vs 0.780 (0.041 [-0.035, 0.127]); no difference shown. BR Coffee 0.78 vs 0.95 (-0.17 [-0.24, -0.09]), but its loss
  share differs by type (0.62 vs 0.89), so that gap follows the base rate, not the imagery.
- Limits: everything is agreement with Hansen, whose loss label the audit finds is about half plantation turnover in ID oil
  palm; the composites' scene dates are not recorded per file; image type is tied to country in 8 of 11 groups. `git` in the
  results file is the commit before `run_e6.py` itself was committed (it is committed with these results).

## 9. E4 and E5 on the 106 oil palm farms added by E6 (PROTOCOL_E8; results/e8/)
*15 Nov 2026: flags and plantation-driven shares stand (class 2 only); the transport check used class 1 (section 11).*
The same code as E4 and E5 on the 106 ID oil palm farms that only have composites (all 19 windows fetched, 0.74 GB received by
the machine during the download against a 1 GB budget; outlines from the main Overpass server, deviation in the protocol).
Transported S2C output, no reference on these farms.
- Box (0.01 degree tile): 56 of 106 farms flagged; 9 of them plantation-driven, **0.16 [0.07, 0.25]**, against 0.58 [0.48, 0.70]
  on the E4 farms. On these farms most box loss is predicted FOREST in 2020 (3,254 of 5,152 loss cells on flagged farms).
- Outline (102 farms; 4 excluded with no cell of the outline inside the box): **7 flagged, all 7 plantation-driven**. 48 farms
  flagged in the box are not flagged in their outline, none the other way. No cell lies in two outlines.
- Transport check: 17 stable-forest cells inside the outlines, under the protocol's 100, so no reading (PLANT share 0.65).
- Pooled with E4/E5, descriptive: 11 of 185 farms flagged inside outlines, all plantation-driven; in the boxes 135 of 200, of
  which 55 plantation-driven (0.41 [0.33, 0.49]). No E5 outline overlaps an E8 outline.
- Reading: the E5 result repeats on a second, disjoint set of farms. Inside the plots, few farms carry 0.5 ha of Hansen loss,
  and those that do are predicted plantation or other land in 2020, not forest. The box numbers differ a lot between the two
  sets (0.58 against 0.16 plantation-driven), which says the land around the plots varies; the box is not a farm-level unit.

## 10. Hansen loss against the EU's 2020 forest map (PROTOCOL_E9; results/e9/gfc2020.json)
*15 Nov 2026: analyses 1 and 2 (loss on GFC forest, flags) and the loss-cell part of analysis 4 stand. The agreement and kappa, the class-1 share and the stable-cell comparison used class 1 and class 0 before the label fix (section 11); rerun pending.*
JRC GFC2020 version 3 (10 m, EUDR forest definition; "may be used by anyone, anywhere, anytime without permission", attribution:
Bourgoin et al., JRC), as the forest share of each 30 m label cell, computed on Earth Engine for all 996 farms' label boxes and
the 40 E4/E8 windows (2,072 grids, 0.10 GB received, none failed). A cell is GFC forest at >= 50 %. Two maps compared; neither is
truth.
- **Share of Hansen 2021-2024 loss on land GFC2020 calls forest in 2020: 0.35 [0.32, 0.38]** (661,505 loss cells). About two
  thirds of the loss Hansen reports on these farms is on land the EU's own map did not count as forest at the cut-off date.
- **Flags: 895 farms flagged by Hansen; 554 still flagged when only loss on GFC forest counts. 0.38 [0.35, 0.41] of the
  Hansen flags do not survive.** By group (farms, share of loss on GFC forest, flags not surviving): ID Oil Palm 200, 0.13,
  0.47 [0.41, 0.55]; IN Rubber 80, 0.00, all 77; VN Coffee 67, 0.00, all 67; CI Cocoa 35, 0.00, 33 of 35; while CM Cocoa (96,
  0.65), GH Cocoa (88, 0.80) and BR Cattle (69, 0.58) keep almost every flag. Groups below 20 farms are in the JSON.
- Agreement on forest in 2020 (Hansen class 1 or 2 against GFC forest), all cells of the label boxes: 0.79, but kappa **0.04**.
  The Hansen labels call 3 % of the box cells forest in 2020 (classes 1 and 2), GFC2020 20 %. Class 0 mixes "never forest" with
  "lost by 2020" (design.md), and GFC calls 0.20 of class-0 cells forest. Checked that this is not a grid error (shapes match,
  flipped grids correlate worse). Why the label rasters hold so little 2020 forest is open (see goals, E10).
- **S2C check (ID oil palm, E4 + E8 windows): S2C and GFC2020 disagree** (pre-registered). Of loss cells S2C calls PLANT, 0.98
  are GFC non-forest (consistent), but of those S2C calls FOREST only 0.07 are GFC forest (4,276 of 4,586 are not). Of Hansen
  stable-forest cells in the farm boxes, GFC calls 0.07 forest, where S2C called 0.55 FOREST (E4). So on the farms, GFC2020 sees
  very little forest, and S2C's FOREST class there is mostly land GFC does not call forest. This fits the E4 warning that S2C
  under-calls palm away from its training grids, and it means the E4/E8 "plantation-driven" shares, taken from S2C, are low.
- Sensitivity: the forest threshold barely matters (30 %: 0.36 of loss on forest, 0.38 of flags lost; 70 %: 0.34, 0.38).
  Version 4 puts less loss on forest (0.19 at 50 %) and loses a similar share of flags (0.40).
- Limits: GFC2020 is a model (V3 overall accuracy about 0.91 in the JRC assessment; no assessment for V4 yet); it is a map of
  2020 only, so it cannot date a conversion. Gating Hansen with GFC2020 is a candidate rule for the product, not adopted.

## 11. The farm labels had no stable-forest class (PROTOCOL_E10; results/e10/)
`generate_labels.py` tested Hansen `lossyear == 0` for "no loss", but on Earth Engine `lossyear` is masked where there was no
loss and never holds 0. So on every farm, forest with no loss was written as class 0, and class 1 held only forest lost in 2025
(`results/e10/diagnosis.md`). Fixed (`lossyear.unmask(0)`), and all 996 label rasters rebuilt on their own grids from Hansen
v1.13 with the tested `reference.hansen_label` (`hansen_labels_v2`, outside git; 87 MB of data).
- Class 2 (loss 2021-2024) is identical on all 996 farms; old class 1 is inside new class 1.
- Shares of all label cells, old -> new: class 0 0.970 -> 0.455, **class 1 0.007 -> 0.522**, class 2 0.023 -> 0.023.
- The 72 reference grids are not affected (`reference.py` reads Hansen tiles, where no loss is 0).

What this does to earlier results (nothing rerun yet; each rerun is a dated deviation of its own protocol):

| Result | Uses | Status |
|---|---|---|
| Sections 1, 6 (reference grids, S2/S2B/S2C) | Hansen tiles | not affected |
| Farm flags, loss counts (E2 rule, E4/E5/E8 flags and plantation-driven shares, E9 analyses 1, 2) | class 2 only | not affected |
| Product change model (`hansen_change_mask`) | class 2 against the rest | not affected |
| E3 box-loss sample (300) and outline census | class 2 | not affected |
| **E3 box-stable sample (120 cells)** | class 1 strata | **drawn from 2025-loss cells, not stable forest; redrawn 15 Nov from `hansen_labels_v2` (PROTOCOL_E3 deviation; `e3_draw/box_stable_v2`, old sheets void). Box-loss sheets and census redrawn on v2 came out byte-identical, so they stand** |
| **Sections 3, 4, 8 (noise experiments A/B, experiment C, E6 transfer)** | task = class 2 against class 1 | **the task was "lost 2021-24" against "lost 2025", not loss against stable forest; rerun** |
| Section 2 (farm sensitivity, usable footprint) | footprint = class 1 or 2 cells | footprint far too small; rerun |
| E4/E5/E8 transport checks (PLANT share of stable cells) | class 1 | measured on 2025-loss cells; rerun |
| E9 analysis 3 (agreement, kappa), class-1 share, stable share in analysis 4 | classes 0 and 1 | rerun |

## 12. Every farm inside its OpenStreetMap outline (PROTOCOL_E12; results/e12/outlines.json)
One Overpass request for all 996 farm ids (3.8 MB, 994 returned; overpass-api.de gave 502, so the E5 mirror was used, a
recorded deviation). 969 farms have at least one label cell inside their outline (25 outlines hold none, 2 not returned).
Outlines are small: median 4.9 ha (quartiles 1.0, 28.5), against about 2,500 ha per label box; 136 outlines have fewer than 6
cells and cannot be flagged at all. Labels `hansen_labels_v2` (loss only); GFC2020 V3 at 50 % from E9.
- **Flags: 873 of the 969 farms are flagged in their box; 138 inside their outline. 0.84 [0.82, 0.87] of box flags are not
  flagged inside the outline**, and no farm is flagged inside its outline but not in its box.
- E9 gate inside outlines: 0.24 [0.18, 0.29] of the 10,645 loss cells inside outlines are GFC2020 forest; of the 138 farms
  flagged inside outlines, 55 still are on GFC-forest loss alone (0.60 [0.52, 0.68] do not survive).
- By group (box flagged -> outline flagged -> outline and GFC-gated): ID Oil Palm 185 -> 10 -> 0; VN Coffee 67 -> 3 -> 0;
  IN Rubber 77 -> 1 -> 0; CI Cocoa 34 -> 3 -> 0; BR Cattle 69 -> 5 -> 2; GH Cocoa 88 -> 20 -> 12; CM Cocoa 96 -> 71 -> 36.
  Cameroon's outlines are larger (median 45 ha against 1.1 ha for ID Oil Palm), so more of them can reach 6 cells.
- Check against E5/E8 (analysis 5): of 177 Indonesian oil palm farms in both, 176 agree on the outline flag; the other has 7
  loss cells on the E4 window grid and 4 on its label grid (the label rasters are on a 0.00027 degree grid, the windows on
  Hansen's 0.00025). Secondary (coverage >= 0.5): identical, every outline is at least 0.55 inside its label raster.
- Reading: on these farms the box, not the plot, carries almost all Hansen flags. What remains inside plots is concentrated in
  Cameroon and Ghana cocoa, where about half also survives the GFC2020 gate. Both maps are models; E3 labels remain the
  reference, and nothing here changes a product outcome.

## 13. Hansen loss against oil palm planting year (PROTOCOL_E13; results/e13/descals.json)
Descals et al. (2024) planting-year layer (Zenodo 13379129 v1.2, CC BY 4.0; 147 MB, MD5s match). The tiles hold two values
the record and paper do not define, 1989 (5.6 % of palm pixels) and 2022 (0.06 %); the run first stopped there, and the owner
then chose the reading: 1989 = planted 1989 or earlier (palm by 2020), 2022 its own category. The tiles share the label
rasters' cell size and line up with them (offset 0.02 cell), so each label cell is one Descals pixel and the 30/50/70 % share
thresholds give identical results.
- **Only 0.011 [0.007, 0.015] of Hansen's 661,505 loss cells on the farms are oil palm planted by 2020** (ID Oil Palm 0.011
  [0.007, 0.017]; highest TH Rubber 0.065, ID Coffee 0.041, CI Cocoa 0.025). Palm planted in 2021: 0.001.
- No Hansen flag is removed by the Descals gate (895 of 895 survive; ID Oil Palm 200 of 200). Inside the E12 outlines:
  0.002 of 10,645 loss cells, 138 of 138 flags survive. Counting 1989 as not palm changes nothing (0.010).
- Analysis 3: every one of the 7,259 palm-by-2020 loss cells is GFC2020 non-forest ("the two maps agree" on those cells), but
  they are few.
- Analysis 5 (E4/E8 windows): of the loss cells S2C calls PLANT, Descals calls 0.19 palm by 2020; of those S2C calls FOREST,
  0.997 not palm. Pre-registered reading: "S2C and Descals disagree".
- Analysis 6: the palm-by-2020 loss cells on ID Oil Palm farms were planted across 1989-2020, most in 2005-2011.
- Diagnostic, not registered (`results/e13/diagnostic_outline_palm.log`): Descals maps 0.061 of the cells inside the ID Oil
  Palm OSM outlines as palm (any year to 2021), and the raw tile around the first farm checked is 0.03 palm against 0.27 for
  its 100 km tile, so this is the data, not the resampling.
- Reading: Descals does not support "the loss is oil palm replanting": by its map these farms are mostly not oil palm, and
  nearly all Hansen loss on them is outside mapped palm. GFC2020 (E9) says most of the same loss is not 2020 forest. So the two
  maps together place most of it on land that is neither 2020 forest nor oil palm (other tree crops, plantation forestry,
  shrub or regrowth are candidates; none is measured here). Either the OSM crop tags, Descals, or both are wrong on these
  farms; the E3 labels (FOREST / PLANT / OTHER) are what can settle it. The E4/E8 "plantation-driven" shares rest on S2C,
  which disagrees with both maps.
