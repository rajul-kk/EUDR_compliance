# PROTOCOL_E3: a reference sample on the farms

Status: frozen

Drafted 13 Nov 2026; frozen 14 Nov 2026 with the owner's decisions on the items marked **(owner)**, recorded under
"Owner decisions" below. `scripts/hansen_audit/e3_sample.py draw` and `census` refuse to run unless the line above reads
`Status: frozen`. Changes from here are appended as dated deviations, as for the other protocols.

## Owner decisions (14 Nov 2026)

- Population: **both**. A census of every Hansen loss cell inside the farms' OSM outlines (89 cells: 5 predicted
  FOREST, 46 PLANT, 38 OTHER; `e3_sample.py census`), plus the stratified box sample below for the E4 comparison.
  A cell can fall in both; it is labelled on each sheet it appears on, and each estimate uses its own sheet.
- Sample: as proposed, 300 loss cells (floor 60 per stratum) and 120 stable cells (40 per stratum).
- Labellers: the owner labels every sheet A (labeller A). A second person, blind to the first labels and to the
  strata, labels sheet B, a seeded random quarter of each sheet's cells (`--second-frac 0.25`).
- Imagery: Esri World Imagery Wayback, the release whose imagery date is nearest to 31 Dec 2020 at each cell.
- Seeds: box loss draw 20261114, box stable draw 20261115, outline census 20261116.

Imagery terms as read on 14 Nov 2026 from the World Imagery item's metadata in ArcGIS Online (item
10df2279f9684e4a9f6a7f08febac2a9): "This work is licensed under the Esri Master License Agreement", and "This layer may
be used in various ArcGIS apps to support data collection and editing, with the results used internally or shared with
others". Wayback releases are archived versions of that layer. The Master License Agreement itself was not read here;
the owner confirms the terms cover these labels before they are published.

Imagery date rule: each label records the date of the imagery used (shown in the Wayback release metadata for that
location). A cell whose nearest available imagery is more than 12 months from 31 Dec 2020 is labelled UNSURE.

Commands (sheets and keys go to `eudr-hansen-audit-refdata/e3_draw/`, outside git; only `design.json` is copied to
`results/e3/`; labellers open the sheets only, never the keys):

    python -I scripts/hansen_audit/e3_sample.py draw <data_root> <refdata>/e4 results/e4/farm_preds.npz PROTOCOL_E3.md <refdata>/e3_draw/box_loss --n-total 300 --n-min 60 --seed 20261114 --domain 6 7 8 --second-frac 0.25
    python -I scripts/hansen_audit/e3_sample.py draw <data_root> <refdata>/e4 results/e4/farm_preds.npz PROTOCOL_E3.md <refdata>/e3_draw/box_stable --n-total 120 --n-min 40 --seed 20261115 --domain 3 4 5 --second-frac 0.25
    python -I scripts/hansen_audit/e3_sample.py census <data_root> <refdata>/e4 results/e4/farm_preds.npz <refdata>/e5_outlines PROTOCOL_E3.md <refdata>/e3_draw/outline_census --seed 20261116 --second-frac 0.25

## Question

`REPORT_S2.md` E4 reports that 46 of 79 flagged farms (0.58 [0.48-0.70]) look plantation-driven, from transported S2C
predictions with no reference on the farms. It warns that the classifier may under-call mature palm there (0.13 of
Hansen stable-forest cells on the farms predicted PLANT, against 0.53 in the reference grids). E3 replaces or confirms
that number with labels on the farms.

## Population and strata

- Cells: every 0.00025 degree Hansen cell whose centre is inside one of the 94 Indonesian oil palm farm boxes used in
  E4 (`results/e4/` and the windows in `eudr-hansen-audit-refdata/e4/windows.json`). A cell inside several boxes is one
  population unit (merged on the global Hansen grid).
- Strata: Hansen class (2 = loss 2021-24, 1 = stable tree cover) crossed with the S2C 2020 prediction (FOREST, PLANT,
  OTHER). Stratum id = 3 x Hansen class + prediction.
- Population counts (`results/e3/population.json`, 13 Nov): loss strata 6 FOREST 1,182, 7 PLANT 465, 8 OTHER 637
  (2,284 cells); stable strata 3 FOREST 408, 4 PLANT 104, 5 OTHER 263 (775 cells). Hansen non-forest cannot be
  stratified: the label files on disk declare 0 as no-data, so non-forest reads as no-data (the E7 encoding problem;
  labels generated after 12 Nov use 255). Non-forest cells are outside the population.

## Population choice after E5 (owner: both, see above)

E5 (13 Nov, `docs/results.md` section 7) found the OpenStreetMap outlines are small (median 1.7 ha against about 120 ha per box):
inside the outlines there are only 89 valid loss cells over 83 farms, and only 4 farms are flagged. Two options:
- Boxes (as above): measures the land around the farms; keeps the comparison with E4.
- Outlines: measures the farms themselves; 89 loss cells are few enough to label all of them (a census, no sampling error
  for that domain), plus a sample of the boxes for the E4 comparison.

## Sample (owner: as proposed)


- Loss strata: 300 cells, at least 60 per stratum, the rest proportional to stratum size (`farm_sample.allocate`). For a
  share near 0.5 in the loss domain, 300 cells give a 95 % half-width of about 0.06 under proportional allocation
  (`n_for_halfwidth(0.5, 0.06) = 267`); the floors widen it a little.
- Stable strata: 120 cells, 40 per stratum, to measure the S2C accuracy on the farms (the transport warning above).
- Seed: fixed above, before the draw.
- Second pass: a second labeller, blind to the first and to the strata, labels a seeded random quarter of each sheet's
  cells. The draw writes `sheet_A.csv` over all cells and `sheet_B.csv` over the quarter, in different orders with
  different ids; keys are kept apart from the sheets.

## Labels

- Who labels: the owner (sheet A) and a second person (sheet B), as above. An agent does not produce labels.
- Imagery: Esri World Imagery Wayback, nearest release to 31 Dec 2020, with the imagery date recorded for every label
  and the 12-month rule above.
- Classes, at 31 Dec 2020, for the land the cell centre falls on:
  - FOREST: tree cover that is not a plantation of a tree crop (natural or semi-natural forest, including degraded
    forest), the regulation's sense of forest (`docs/regulation_check.md`).
  - PLANT: an oil palm or other tree-crop plantation, including blocks cleared or replanted inside a plantation.
  - OTHER: any other land: non-tree crops, shrub, bare ground, water, roads and buildings.
  - UNSURE: imagery missing, clouded or too far from the date. UNSURE cells are dropped from the estimates and counted.
- Fields: label, confidence (high, medium, low), imagery date, note.

## Estimates (`hansen_audit/farm_sample.py`)

1. Share of FOREST, PLANT and OTHER among the farms' loss cells: stratified estimator, standard error
   `sum_h W_h^2 p_hk (1 - p_hk) / (n_h - 1)`, normal 95 % interval (Olofsson et al. 2014). No finite-population correction
   (conservative; the loss strata are sampled at about 13 %).
2. S2C accuracy on the farms: estimated area-proportion confusion matrix with user's and producer's accuracy, on the
   sampled strata.
3. The E4 number with measured rates: P(FOREST | predicted class) on the loss cells replaces the raw prediction, every
   farm's FOREST loss count is drawn as a sum of binomials, and the share of flagged farms with fewer than 6 FOREST loss
   cells is reported with a bootstrap interval (resampling labelled cells within each predicted class).
4. Agreement between passes, on the quarter labelled twice: percent agreement and Cohen's kappa. If kappa is below
   0.6, the disagreements are adjudicated and both the adjudicated and the first-pass estimates are reported.
5. Inside the outlines (census): the share of FOREST, PLANT and OTHER among the 89 loss cells, overall and by predicted
   class, as plain proportions with no sampling error (UNSURE cells counted and left out), and the number of farms with
   6 or more FOREST loss cells inside their outline, against the 4 flagged by E5.

## Decision

- The E4 sentence in `REPORT_S2.md` is replaced by estimate 3 with its interval, and estimate 1 is reported beside the
  grid result (PLANT 0.49 [0.30-0.65]).
- "Confirmed" if the interval of estimate 3 contains 0.58, "replaced" otherwise. Either way the measured number is the one
  reported.

## Limits

- The farm boxes are 0.01 degree tiles around OpenStreetMap points, not plot outlines (E5); a box can include land
  outside the plantation, and estimates are for the boxes.
- One reading of the imagery per cell (two where the second pass covers it); the reference is a labeller, not truth.
