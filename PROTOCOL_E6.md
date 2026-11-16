# PROTOCOL_E6: the full set of farms (experiment C and the transfer check)

Status: frozen

Frozen 15 Nov 2026, before any model was trained on the new farms. What was seen before freezing: file counts, raster
format checks, the per-group count of farms with usable cells (below), and the check that the old features are reproduced.
No label share, score or prediction on the new farms was looked at. Changes after this are appended as dated deviations.

## Why

`docs/results.md` section 4: experiment C (per-farm uncertainty, PROTOCOL.md) ran on 7 evaluation farms with 3 for
calibration, so the conformal half-width was infinite and it "needs the full set of farms". The goal (E6) assumed the
missing imagery had to be downloaded. On 14 Nov it turned out to be on disk: `hansen_audit/io.py` only read file names with
a scene date, and 670 undated files per year were never read. With them all 996 labelled farms have both years. No
download is needed.

## Data

- Two image types, never mixed within a farm:
  - scene (415 farms): one Sentinel-2 L2A scene per year, June 2020 and June 2024, as used so far.
  - composite (581 farms): `src/sentinel_client.py` median composites of up to 3 scenes, May to September or the full calendar
    year (tropical profile), file names without a date. Cloudy pixels (SCL 0, 1, 3, 8, 9, 10) are removed from each scene
    before the median, and the SCL band is rewritten so that a pixel counts as cloudy only when no scene saw it clear. So the
    feature mask means "no clear observation" for both types, but the spectral values are a median over a wider window.
- Features: `build_features.py <data_root> results/e6_features.npz --composites` (log in `results/e6/build_features.log`), the
  same features as PROTOCOL.md. On the 415 scene farms they are identical to `results/features.npz` (checked: X, Y, V equal).
- Farms with usable cells (Hansen class 1 or 2, valid, full 3x3 window): 586 of 996 (227 scene, 359 composite), 25,844 cells.
  By group (scene / composite): ID Oil Palm 111 (35 / 76), CM Cocoa 74 (51 / 23), GH Cocoa 71 (0 / 71), IN Rubber 64 (0 / 64),
  VN Coffee 54 (54 / 0), BR Cattle 46 (46 / 0), ID Coffee 30 (0 / 30), CI Cocoa 26 (0 / 26), BR Coffee 24 (10 / 14),
  TH Rubber 21 (21 / 0), KE Coffee 20 (0 / 20); 8 smaller groups with 1 to 14 farms.

## Analyses

All models are the PROTOCOL.md patch CNN with cross-entropy, 15 epochs, 5 seeds, standardised on the training farms. All
scores are agreement with Hansen labels, not accuracy (the labels are what the audit questions).

1. Experiment C, as registered. The PROTOCOL.md code unchanged (`run_experiments.py`, block "Exp C"): the 111 ID Oil Palm farms
   with usable cells, split 60/20/20 with seed 1, the first half of the test farms as calibration, split conformal at 90 %,
   flag at `DEFAULT_MIN_CELLS` (6) cells. Expected sizes: 23 evaluation farms, 11 calibration, 12 test, enough for a finite
   half-width (9 needed). Reported: half-width, Hansen and model flag rates, agreement, uncertain share, ensemble spread.
   Added: the image type of each test farm and the flag agreement by type, as a description only.
2. Leave-one-group-out transfer (registered in PROTOCOL.md, never run). Groups = country|crop with at least 20 farms with usable
   cells (the 11 above). Two predictions for every cell of those groups:
   - in-group: 5-fold cross-validation over all 586 farms, folds by farm (seed 0), inside each fold 20 % of the training
     farms for early stopping; each cell predicted by the fold that held its farm out, so its group's other farms were seen.
   - held-out group: train on every farm not in the group (20 % of them for early stopping, seed 0), predict the group.
   Per group: F1 of loss and balanced accuracy for both, and the gap (in-group minus held-out) in F1 with a 95 % interval
   from 2,000 bootstrap resamples of the group's farms (pooled counts per resample). Predictions use the mean probability of
   the 5 seeds, threshold 0.5. Reading: a group "does not transfer" if the F1 gap is at least 0.05 and its interval excludes 0;
   otherwise "transfers". No overall test.
3. Image type. Within the groups that hold at least 10 farms of each type (ID Oil Palm, CM Cocoa, BR Coffee), the in-group F1
   on scene farms against composite farms, with a farm-bootstrap interval of the difference, and the Hansen loss share of
   the usable cells by type. Descriptive: image type is tied to country for most groups, so this is the only place it can be
   looked at, and it still mixes farms that differ in other ways.

## Not done here

- The Hansen-agreement experiments A and B are not rerun on the new farms; their registered result stands.
- The S2C classifier and E4/E5 used the 94 dated ID Oil Palm farms; extending them to the composite farms is not part of E6.
- Nothing here measures truth. E3 (labels pending) is the only reference on the farms.

## Outputs

`results/e6/experiments.json` (with the command and git commit), `results/e6/run.log`; a section in `docs/results.md`.

## Deviations

- 15 Nov 2026, rerun on the rebuilt labels (owner's go-ahead, 15 Nov: "run it", after the PROTOCOL_E10 inventory). The label rasters used before had no stable-forest class (class 1 held only 2025 loss). Experiment C on all farms and the leave-one-group-out transfer check (run_e6.py) is rerun unchanged except that the labels are `hansen_labels_v2` (`--labels-dir`); the earlier outputs are kept and the reruns are written beside them with a `_v2` suffix. Two reads were made safe for labels that use 255 for Hansen no-data, with no change on the old files (checked: features of three farms reproduce exactly): `model.make_patches` keeps classes 1 and 2 only, and `align.label_on_image_grid` measures coverage from the raster extent.
