# PROTOCOL_E9: Hansen loss against the EU's 2020 forest map (JRC GFC2020)

Status: frozen

Frozen 15 Nov 2026, before any GFC2020 pixel was fetched. What was seen before freezing: the Earth Engine catalogue pages for
`JRC/GFC2020/V3` and `JRC/GFC2020/V4` (terms, band, grid) and the projection of V3 (EPSG:4326, 1/12000 degree, so not aligned
with the label grid). No GFC value on any farm was looked at. Changes are appended as dated deviations.

## Why

The audit's main finding is that Hansen counts plantation replanting as forest loss. Under Regulation (EU) 2023/1115,
deforestation is conversion of forest that existed at 31 December 2020; land that was not forest then cannot be deforested.
The JRC's Global Forest Cover 2020 map (GFC2020) is the EU's own reference map for that date, built on the regulation's
forest definition (it is labelled non-mandatory and not legally binding). E9 asks: of the loss Hansen reports on the farms
after 2020, how much falls on land GFC2020 calls forest in 2020, and how many Hansen flags survive if loss is counted only
there? On the Indonesian oil palm farms it also checks the transported S2C classifier (E4/E8) against GFC2020.

GFC2020 is a model too (version 3: about 91 % overall accuracy in the JRC assessment). Nothing here measures truth; it
measures agreement between two maps. E3's human labels remain the only reference.

## Data

- GFC2020 version 3 (`JRC/GFC2020/V3`, band `Map`, 1 = forest), primary, because its accuracy is documented. Version 4
  (`JRC/GFC2020/V4`), sensitivity only; its technical report is not yet out. Terms (catalogue): "The data may be used by
  anyone, anywhere, anytime without permission, license or royalty payment"; attribution requested (Bourgoin et al., JRC,
  PIDs http://data.europa.eu/89h/8c561543-31df-4e1b-9994-e529afecaf54 for V3 and
  https://data.europa.eu/89h/cf7c49ea-8f00-4c5a-bd40-6f25dc341580 for V4).
- Fetched from Earth Engine with the owner's project (credentials from the existing `.env`, never printed). For each grid
  the server computes the area-weighted forest fraction of GFC pixels in each cell (`Map == 1`, unmasked to 0,
  `reduceResolution(mean)`), returned as an integer percentage. Grids: every farm's Hansen label raster (996 farms), and the
  E4 and E8 window grids (Hansen 0.00025 degree). Stored in `eudr-hansen-audit-refdata/e9_gfc2020/`, outside git.
- Download budget approved by the owner on 15 Nov: 1 GB (`hansen_audit.budget.DownloadBudget`, which counts all bytes the
  machine receives, so it is conservative). Expected about 60 MB.
- Hansen labels as in the audit: 0 = not forest in 2000 or lost by 2020, 1 = forest through 2024, 2 = lost 2021 to 2024.
  Label nodata 255 (E7) is excluded.

## Rules

- A cell is GFC forest if its forest fraction is at least 50 %. Sensitivity: 30 % and 70 %.
- Farm flag as in the product and audit: 6 or more loss cells (`decisions.DEFAULT_MIN_CELLS`). The GFC-gated count is the
  number of Hansen loss cells that are GFC forest; a farm is gated-flagged if that count is 6 or more.
- Population: all cells of each farm's label raster (the box the audit uses). Groups: country and crop from
  `results/e6_features.npz`. Intervals: 95 %, 2,000 bootstrap resamples of farms.

## Analyses

1. Primary: share of Hansen loss cells that are GFC forest, pooled over all farms, with a farm-bootstrap interval, and by
   group. Beside it: share of Hansen class-1 cells and of class-0 cells that are GFC forest.
2. Primary: farms flagged by Hansen, farms still flagged after gating, and the share of Hansen flags that do not survive
   (farm-bootstrap interval), overall and by group.
3. Agreement on forest in 2020 (Hansen 1 or 2 against GFC forest): confusion counts, overall agreement and Cohen's kappa.
4. ID oil palm, S2C check: on Hansen loss cells inside the E4 and E8 farm boxes with a valid S2C prediction, the cross-table of
   the S2C 2020 class (FOREST, PLANT, OTHER) against GFC forest. Pre-registered reading: if GFC calls at least 0.70 of the
   S2C-PLANT loss cells non-forest and at least 0.70 of the S2C-FOREST loss cells forest, "S2C and GFC2020 agree"; if either
   share is 0.50 or less, "they disagree"; otherwise "mixed". Also the share of stable-forest cells (Hansen 1) that GFC calls
   forest, beside S2C's 0.55 FOREST (E4).
5. Sensitivity: analyses 1 and 2 with version 4 and with thresholds 30 % and 70 %.

## Not done here

- No farm outcome changes in the product. Gating Hansen with GFC2020 is a candidate rule; adopting it is the owner's call.
- No legal statement beyond the regulation's cut-off date, which is already in the README.
- Hansen loss before 2021 is in class 0 and is not looked at.

## Outputs

`scripts/hansen_audit/gfc2020.py`, `hansen_audit/gfc.py` with tests; `results/e9/gfc2020.json`, `results/e9/download_manifest.json`,
`results/e9/run.log`; a section in `docs/results.md`.
