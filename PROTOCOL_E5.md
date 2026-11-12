# PROTOCOL_E5: farm-level numbers inside the plot outlines

Status: frozen

Frozen 13 Nov 2026, before any outline was fetched or any count inside an outline was made. Changes after this are appended
below as dated deviations.

## Question

E4 (PROTOCOL_S2 deviation 5, `results/e4/`) counted Hansen loss and the S2C 2020 prediction in the full 0.01 degree box
around each of the 94 Indonesian oil palm farms. Boxes overlap, and only 0.13 of the farms' Hansen stable-forest cells
were predicted PLANT (0.53 in the reference grids). E5 asks: inside the farms' own OpenStreetMap outlines, how do the
farm-level numbers change, and does the PLANT share of stable cells rise (the box included forest outside the
plantation) or stay low (the classifier under-calls palm off the grids)?

## Data

- Outlines: the OpenStreetMap way or relation named by each farm id (`way_<id>`, `relation_<id>`), fetched once from the
  Overpass API with full geometry; the fetch time and each object's version and timestamp are recorded. Data
  (c) OpenStreetMap contributors, ODbL. Outlines are stored with the reference data outside git; the repo keeps the
  per-farm counts and a manifest.
- Cells, Hansen labels and S2C predictions: exactly those of E4 (`results/e4/farm_preds.npz`, the windows in
  `eudr-hansen-audit-refdata/e4/windows.json`, the labels in `hansen_labels/`). No refit, no new imagery.

## Rules

- A cell belongs to a farm if its centre lies inside the outline (holes of multipolygons excluded) and inside the farm's
  E4 box. Only box cells have predictions, so a large outline is clipped to its box.
- Coverage = outline area inside the box / outline area (equal-area projection, EPSG:6933). A farm with coverage below 0.5
  is "partial".
- Excluded and counted: ids whose object no longer exists, is not a closed area, or has no cell inside the box.
- Flag and plantation-driven rules as in E4: flagged at 6 or more loss cells; plantation-driven if fewer than 6 of them
  are predicted FOREST in 2020 (`hansen_audit.farms_e4.farm_summary`).

## Outputs (`results/e5/`)

1. Primary: `farm_summary` over all included farms, counted inside outlines; beside it the E4 box numbers for the same
   farms.
2. Flag changes: farms flagged in the box and not in the outline, and the reverse; plantation-driven changes likewise.
3. Overlap: cells inside more than one outline.
4. Transport check: share of stable-forest cells predicted PLANT inside outlines, against 0.13 in the boxes.
   Pre-registered reading: 0.40 or more supports "the boxes included forest outside the plantation"; 0.20 or less
   supports "the classifier under-calls palm on the farms"; between is inconclusive.
5. Secondary: outputs 1 and 4 restricted to farms that are not partial.

## Limits

- OSM outlines are volunteered and may be outdated or drawn loosely; they are the best open outline available here, not
  a cadastral plot.
- Predictions are still transported S2C output with no reference on the farms (E3 would supply one).
