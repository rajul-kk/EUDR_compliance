# PROTOCOL_E12: plot outlines for all 996 farms (goal U6)

Status: frozen

Frozen 15 Nov 2026, before any outline beyond the E5 and E8 sets was fetched. Owner's go-ahead: 15 Nov ("do E3 and new
data", on the list that named outlines for all 996 farms). What was seen before freezing: the farm ids in the label index
(783 ways, 213 relations), the label grids (EPSG:4326, 0.000269495 degree cells, one raster per farm), the E5 and E8 results.
No outline outside E5/E8 was looked at. Changes are appended as dated deviations.

## Why

Every per-farm number outside E5 and E8 is counted over the farm's label raster, a box around an OpenStreetMap point that
can include land outside the plot. E5 and E8 showed, on the Indonesian oil palm farms, that most box flags disappear inside
the outline (4 of 83 and 7 of 102 flagged inside outlines). E12 asks the same on every group, and repeats the E9 GFC2020
gate inside the outlines.

## Data

- One Overpass request (`out geom meta`) for every farm id in `hansen_labels_v2` (the ids are OSM way and relation ids).
  Endpoint `https://overpass-api.de/api/interpreter` (the E8 deviation: the kumi mirror timed out). One request; if it
  fails or times out, that is recorded and the owner is told, with no retry loop. Stored in
  `eudr-hansen-audit-refdata/e12_outlines/` (outside git), with a manifest (time, endpoint, bytes, versions, timestamps).
  Data (c) OpenStreetMap contributors, ODbL 1.0.
- Budget: 0.1 GB of the job's own bytes (the response is counted as it streams; the fetch stops past the cap), under the
  machine-wide download lock. Expected a few MB.
- Labels: `eudr-satellite-upload/hansen_labels_v2` (PROTOCOL_E10). Only class 2 (loss 2021-24) and class 1 enter the
  analyses; class 2 is identical to the old files.
- GFC2020 V3 forest fractions on the label grids, already on disk from E9 (`e9_gfc2020/`), threshold 50 %.

## Rules

- A farm's outline cells: label cells whose centres fall inside the outline (holes excluded), from the label raster's own
  transform. Cells outside the label raster are not counted; `coverage` is the share of the outline's area inside the raster
  bounds (equal-area), as E5.
- Included farms: returned by Overpass, a closed area, and at least one outline cell. Every exclusion is listed with its
  reason.
- Flag as in the product (`decisions.farm_flag`, 6 or more loss cells, about 0.5 ha). Outlines smaller than 6 cells can
  never be flagged; their number is reported.
- Groups: country and crop from `results/e6_features.npz`. Intervals: 95 %, 2,000 farm-bootstrap resamples
  (`decisions.rate_ci`).

## Analyses

1. Primary: farms flagged in the box and inside the outline, on the same included farms, overall and by group; the share of
   box flags that are not flagged inside the outline, with an interval.
2. Primary: the E9 gate inside the outlines: Hansen loss cells inside outlines that are GFC2020 forest, as a share of loss
   cells inside outlines (with an interval), and farms still flagged on GFC-forest loss inside the outline.
3. Descriptive: outline area (median, quartiles), coverage, outlines under 6 cells, outline cells shared by two or more
   farms.
4. Secondary: analyses 1 and 2 on farms with coverage at least 0.5 only.
5. Check against E5/E8: on the Indonesian oil palm farms in both sets, the outline flag must agree with E5/E8 wherever the
   outline is unchanged (same OSM version) and lies inside both the label raster and the E4/E8 box. Disagreements are
   listed and explained, not dropped.

## Not done here

- No farm outcome changes in the product; using outlines is the owner's call (the regulation's geolocation rules for plots
  are not interpreted here).
- No S2C classifier reading (the 2020 classifier exists only for the E4/E8 windows).

## Outputs

`scripts/hansen_audit/e12_outlines.py` with tests, `results/e12/` (outlines.json, run.log, fetch manifest without
geometries), a section in `docs/results.md`.

## Deviations

- 15 Nov 2026, fetch: the one request to overpass-api.de returned 502 Bad Gateway (`results/e12/fetch.log`). One further
  request, no more, was sent to the mirror E5 used (`https://overpass.kumi.systems/api/interpreter`); if that also failed,
  the fetch was to stop and the owner be told.
- 15 Nov 2026, fetch outcome: the mirror returned 3.8 MB, 994 of 996 farms. Reading it back failed on Windows' default
  cp1252 decoding (the response is UTF-8), after the file had been saved; the code now reads it as UTF-8 and the manifest
  was written from the saved file (its time is the file's time). No further request was sent.
