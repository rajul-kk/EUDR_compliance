# PROTOCOL_E8: E4 and E5 on the 106 Indonesian oil palm farms added by E6

Status: frozen

Frozen 15 Nov 2026, before any Sentinel-2 or Sentinel-1 data or OpenStreetMap outline was fetched for these farms and before
any count on them was made. What was seen before freezing: the farm count (200 ID Oil Palm farms with both years, 94 in E4,
106 new, all 106 composite farms) and the window grouping (19 windows, 160,439 cells; an estimate of about 0.8 GB from the
cell ratio to E4). No label share or prediction on the new farms was looked at. Changes are appended as dated deviations.

## Why

E4 (PROTOCOL_S2 deviation 5) and E5 (PROTOCOL_E5) used the 94 ID Oil Palm farms that had dated June scenes. E6 found the
loader skipped undated composites; with them 200 ID Oil Palm farms have both years. E5's result on the 94 rests on few cells
(4 of 83 farms flagged inside outlines; 22 stable-forest cells inside outlines for the transport check). E8 repeats E4 and
E5 unchanged on the 106 new farms, to see whether those numbers hold on a second, disjoint set of farms.

## Data

- Farms: the ID Oil Palm farms in `results/e6_features.npz` not listed in `eudr-hansen-audit-refdata/e4/windows.json`. Farm
  boxes are the bounds of each farm's 2020 image (composites included). The farm images are used for nothing else.
- 2020 Sentinel-2 and Sentinel-1 composites per window: `e4_farms.py download ... --composites --exclude <E4 windows.json>`,
  the same code, windows (at most 0.05 degree) and compositing rules as E4, into `eudr-hansen-audit-refdata/e8/`.
  Download budget approved by the owner on 15 Nov: 1 GB, enforced by the script's byte guard (`--budget-gb 1`). If the budget
  stops the download, the windows not fetched are reported as missing and nothing is fetched beyond it.
- Classifier: S2C fit once on all 72 reference grids, exactly as E4 (`e4_farms.py run`). No refit on farm data.
- Outlines: one Overpass request for the new farms' ways and relations, as E5 (`e5_outlines.py fetch`), into
  `eudr-hansen-audit-refdata/e8_outlines/`. (c) OpenStreetMap contributors, ODbL; stored outside git.

## Analyses

All rules (flag at 6 or more loss cells, plantation-driven if fewer than 6 of them predicted FOREST in 2020, outline cell
membership, coverage and "partial" at 0.5, exclusions) are those of E4 and E5, with their code unchanged.

1. Primary, new farms only: E4 box numbers (`results/e8/box/`) and E5 outline numbers (`results/e8/outline/`): farms flagged,
   plantation-driven share with its farm-bootstrap interval, flag changes box to outline, overlap.
2. Transport check on the new farms inside outlines: share of stable-forest cells predicted PLANT. Same pre-registered
   reading as E5: 0.40 or more supports "the boxes included forest outside the plantation", 0.20 or less "the classifier
   under-calls palm on the farms", between inconclusive. If fewer than 100 stable-forest cells fall inside the outlines, it
   is reported as "too few cells" with no reading.
3. Pooled with E4/E5 (all 200 farms), descriptive: the same summaries. The two runs use separate windows, so a cell can sit in
   an E4 window and an E8 window; overlap between the two sets of outlines is counted and reported.

## Not done here

- E3's sample stays as drawn on the 94 E4 farms; nothing here changes its population, sheets or estimator. The new farms have
  no reference labels, and every number here is transported model output, as in E4.
- No other country or crop (the S2C classifier was trained on Indonesian grids only).

## Outputs

`results/e8/box/` (farms.json, farm_preds.npz, run.log), `results/e8/outline/` (farms.json, run.log),
`results/e8/download_manifest.json` (copied from the refdata folder, with bytes received), `results/e8/pooled.json`, and a
section in `docs/results.md`.
