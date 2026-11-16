# PROTOCOL_E13: Hansen loss against oil palm planting year (Descals et al. 2024)

Status: frozen

Frozen 15 Nov 2026, before any Descals file was downloaded. Owner's go-ahead: 15 Nov ("do E3 and new data", on the list
that named the Descals planting-year maps, CC BY 4.0, 304 MB). What was seen before freezing: the Zenodo record 13379129
(v1.2) metadata and description, nothing else. Changes are appended as dated deviations.

## Why

The audit's claim is that Hansen counts the clearing of existing plantations as forest loss. GFC2020 (E9) says where the
EU's 2020 map saw forest; it does not say what the non-forest land was. Descals et al. map oil palm and its planting year
from the Landsat series, so they can say directly whether a Hansen loss cell was already oil palm before the cut-off. This is
a second independent map, not truth; E3's labels remain the reference.

## Data

- Zenodo record 13379129, v1.2, "Global oil palm extent and planting year from 1990 to 2021" (Descals, Gaveau, Wich, Szantoi,
  Meijaard; ESSD 16, 5111-5129, 2024), licence CC BY 4.0. Per the record: `GlobalOilPalm_OP-YoP.zip`, 609 GeoTIFF tiles of
  100 x 100 km, 30 m, the year of oil palm planting (1990-2021); `Grid_OilPalm2016-2021.zip`, the tile grid;
  `Validation_points_GlobalOP2016-2021.zip`, 18,812 visually labelled points. The 10 m extent layer
  (`GlobalOilPalm_OP-extent.zip`, 157 MB) is not downloaded: the planting-year layer marks oil palm cells too.
- Download: the three files above, 148 MB in all, checked against Zenodo's MD5s. Budget: the owner's 304 MB, counted as the
  job's own streamed bytes (the download refuses to start if the declared sizes pass it, and stops past it), under the
  machine-wide download lock. Stored in `eudr-hansen-audit-refdata/e13_descals/` (outside git); only tiles that intersect a
  farm's label raster are unpacked.
- Before analysis, the value encoding (no-data and non-palm values, data type, grid) is read from the tiles and recorded in
  `results/e13/encoding.json`. If it does not match "planting year 1990-2021 on palm, a single no-data or zero value
  elsewhere", nothing is run and the owner is told.
- Labels: `hansen_labels_v2`. Primary analyses use class 2 only (loss 2021-24, unchanged by E10).

## Rules

- On each farm's label grid, the share of each label cell covered by Descals palm planted in or before 2020 is computed by
  area-weighted resampling of the indicator (`rasterio` average resampling from the tile grid). A cell is "palm by 2020" if
  that share is at least 50 % (sensitivity 30 % and 70 %). Same for "palm planted 2021", reported separately: it cannot tell
  a 2021 replanting from a 2021 conversion.
- Flag as in the product (`decisions.farm_flag`). The Descals-gated count of a farm is its Hansen loss cells that are not
  palm by 2020; gated-flagged if 6 or more.
- Groups: country and crop from `results/e6_features.npz`; the primary group is ID Oil Palm (200 farms). Intervals: 95 %,
  2,000 farm-bootstrap resamples.

## Analyses

1. Primary: the share of Hansen loss cells that are palm by 2020, pooled and by group, with intervals; beside it the share
   that is palm planted 2021.
2. Primary: Hansen flags, Descals-gated flags, and the share of Hansen flags that do not survive, overall and by group.
3. Agreement with E9 on loss cells: cross-table of palm by 2020 against GFC2020 V3 forest (50 %). Pre-registered reading on
   ID Oil Palm: if GFC calls at least 0.70 of the palm-by-2020 loss cells non-forest, "the two maps agree that this loss was
   not on 2020 forest"; if 0.50 or less, "they disagree"; otherwise "mixed".
4. Inside outlines (E12), when E12 has run: analyses 1 and 2 on outline cells.
5. ID oil palm S2C check, E4 and E8 windows: of the loss cells S2C predicts PLANT, the share that is palm by 2020; of those it
   predicts FOREST, the share that is not. Same reading thresholds as analysis 3 (agree at both >= 0.70, disagree if either
   <= 0.50).
6. Descriptive: planting-year histogram of palm-by-2020 loss cells (ID Oil Palm).
7. Sensitivity: analyses 1 and 2 at 30 % and 70 %.

## Limits

- The planting-year layer is from 2021 imagery; palm cleared before 2021 and not replanted is not palm here, so the share
  palm by 2020 is a lower bound on loss inside existing plantations.
- Descals' own accuracy (in the paper) is not re-measured here; the validation points are downloaded but not used in these
  analyses.

## Outputs

`hansen_audit/descals.py`, `scripts/hansen_audit/descals.py` with tests; `results/e13/` (encoding.json, descals.json,
download manifest, run.log); a section in `docs/results.md`. Attribution: Descals et al. 2024, CC BY 4.0.

## Deviations

- 15 Nov 2026, stopped at the encoding check. The three files came in at 147 MB (MD5s match; `results/e13/download_manifest.json`).
  The 609 tiles (uint16, EPSG:4326, 0.000269495 degree, no declared no-data) hold 0, every year 1990-2021, and two values the
  record and the paper (ESSD 16, 5111-5129) do not define: 1989 (5.6 % of palm pixels) and 2022 (0.06 %)
  (`results/e13/encoding.json`, `encoding_counts.log`). As registered, nothing was run; the owner decides how to read 1989
  and 2022. The paper defines the planting year as the year the current (2021) plantation was established, new or
  replanted, and says it may be later than the first establishment.
- 15 Nov 2026, owner's decision ("run it", on the recommended reading): 1989 is read as "planted in 1989 or earlier" and
  counts as palm by 2020; 2022 is reported as its own category and never counts as palm by 2020. Sensitivity: 1989 counted
  as not palm. The encoding check accepts exactly these two extra values; any other value still stops the run. To allow
  this sensitivity without recomputing, `grids` stores the cell shares of 1989, 1990-2020, 2021 and 2022 separately, plus
  the nearest-neighbour planting year at each cell centre (for the histogram, analysis 6).
- 15 Nov 2026, run: the tiles share the label rasters' cell size and align with them, so each label cell is one Descals
  pixel and the 30/50/70 % thresholds coincide (reported, not a change of rule). After the run, a diagnostic outside the
  registered analyses was added because the result looked implausible: the Descals palm share of the ID Oil Palm outline
  cells, and the raw tile read at one farm (`results/e13/diagnostic_outline_palm.log`); it changes no registered number.
