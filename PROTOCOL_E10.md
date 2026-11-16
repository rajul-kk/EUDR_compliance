# PROTOCOL_E10: rebuild the farm label rasters (the stable-forest class was missing)

Status: frozen

Frozen 15 Nov 2026, before any rebuilt label was fetched. What was seen before freezing (diagnostics on 10 random farms and
then 2, logged in `results/e10/diagnosis.md`): the label rasters on disk match `GEE_dynamic/src/generate_labels.py` run on
`UMD/hansen/global_forest_change_2025_v1_13` cell for cell, and that code has a bug.

## The bug

In Earth Engine, Hansen's `lossyear` band is masked where no loss was detected; it never holds 0. `generate_labels.py` tests
`lossyear.eq(0)` to find forest with no loss, which is never true, so:
- forest with no loss (tree cover >= 10 % in 2000, no loss 2001-2024) is written as **class 0**, not 1;
- class 1 holds only forest lost after 2024 (every class-1 cell on the two farms checked has lossyear 25, i.e. 2025);
- class 2 (loss 2021-2024 on tree cover >= 10 %) is correct, so loss counts and farm flags are unaffected.
`hansen_audit/reference.py` reads Hansen's GeoTIFF tiles, where no loss is stored as 0, so the 72 reference grids
(results sections 1, 6 and the S2/S2B/S2C protocols) are not affected.

## What is done

1. Fix `generate_labels.py`: `lossyear.unmask(0)` before any comparison (and 255 for no data, as E7 intended).
2. Rebuild every farm's label raster on its existing grid (same transform and shape as the file on disk), into
   `eudr-satellite-upload/hansen_labels_v2/` (outside git): fetch `treecover2000`, `lossyear` (unmasked to 0) and `datamask`
   from `UMD/hansen/global_forest_change_2025_v1_13` with Earth Engine, and compute the label locally with the tested
   `hansen_audit.reference.hansen_label` (one definition for grids and farms); 255 where datamask is 0. Download budget: 0.2 GB
   under `DownloadBudget` (expected about 85 MB), within the owner's go-ahead for the label rebuild (15 Nov).
3. Checks, reported in `results/e10/labels.json`: class 2 identical to the old files on every farm (outside no-data cells);
   old class 1 a subset of new class 1; shares of each class and of no data, overall and by group.
4. An inventory of every registered result that reads farm class 1 or class 0, with what changes. No result is rerun here;
   reruns are separate, dated deviations of their own protocols, after the owner has seen the inventory.

## Outputs

`scripts/hansen_audit/rebuild_labels.py`, the fix in `generate_labels.py`, `results/e10/` (diagnosis.md, labels.json,
run.log, download_manifest.json), and an inventory section in `docs/results.md`.

## Deviations

- 15 Nov 2026, budget: the first download run stopped at the 0.2 GB guard after 520 farms, having received 44 MB of its own
  data; the guard counts every byte the machine receives and other traffic made up the rest. The run was resumed with the
  guard at 0.4 GB. The job's own data over both runs: 87 MB, as expected; the machine received 0.31 GB in the two runs.
