# PROTOCOL_E3: a reference sample on the farms

Status: draft

Drafted 13 Nov 2026. Not frozen: the owner decides the items marked **(owner)** and then changes the status line to
`Status: frozen`. `scripts/hansen_audit/e3_sample.py draw` refuses to draw a sample until that line is present. After
freezing, changes are appended as dated deviations, as for the other protocols.

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

## Population choice after E5 (owner)

E5 (13 Nov, `docs/results.md` section 7) found the OpenStreetMap outlines are small (median 1.7 ha against about 120 ha per box):
inside the outlines there are only 89 valid loss cells over 83 farms, and only 4 farms are flagged. Two options:
- Boxes (as above): measures the land around the farms; keeps the comparison with E4.
- Outlines: measures the farms themselves; 89 loss cells are few enough to label all of them (a census, no sampling error
  for that domain), plus a sample of the boxes for the E4 comparison.

## Sample (owner)

Proposal:
- Loss strata: 300 cells, at least 60 per stratum, the rest proportional to stratum size (`farm_sample.allocate`). For a
  share near 0.5 in the loss domain, 300 cells give a 95 % half-width of about 0.06 under proportional allocation
  (`n_for_halfwidth(0.5, 0.06) = 267`); the floors widen it a little.
- Stable strata: 120 cells, 40 per stratum, to measure the S2C accuracy on the farms (the transport warning above).
- Seed: fixed in this file before the draw.
- Second pass: a second labeller, blind to the first and to the strata, labels **(owner)** all cells, or a random
  quarter of them. The draw writes both sheets (`sheet_A.csv`, `sheet_B.csv`) over the same cells in different orders
  with different ids; keys are kept apart from the sheets.

## Labels

- Who labels **(owner)**. An agent does not produce labels.
- Imagery **(owner)**: very-high-resolution imagery as close to 31 Dec 2020 as possible, with the date recorded for
  every label. Candidates to check for licence and coverage before use: Google Earth historical imagery, Esri World
  Imagery Wayback, Planet NICFI tropical basemaps (monthly mosaics). Licence terms are not asserted here.
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
4. Agreement between passes: percent agreement and Cohen's kappa. If kappa is below 0.6, the disagreements are
   adjudicated and both the adjudicated and the first-pass estimates are reported.

## Decision

- The E4 sentence in `REPORT_S2.md` is replaced by estimate 3 with its interval, and estimate 1 is reported beside the
  grid result (PLANT 0.49 [0.30-0.65]).
- "Confirmed" if the interval of estimate 3 contains 0.58, "replaced" otherwise. Either way the measured number is the one
  reported.

## Limits

- The farm boxes are 0.01 degree tiles around OpenStreetMap points, not plot outlines (E5); a box can include land
  outside the plantation, and estimates are for the boxes.
- One reading of the imagery per cell (two where the second pass covers it); the reference is a labeller, not truth.
