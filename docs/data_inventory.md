# Data inventory (computed by `hansen_audit/inventory.py`; raw output `results/inventory.json`)

Folder `D:/Work/eudr-satellite-upload` (read-only). All numbers below were computed from the files on 2026-11-05.

## Counts
- `farms_osm.csv`: 1,020 rows, 996 unique farm keys (24 duplicate rows). Columns: farm_id, lat, lon, crop_type, country_iso2.
  Not a ~1,000 *complete* set: see triplet count.
- Files: baseline 627 TIFFs (623 farms; 4 farms have 2 dates), current 623 TIFFs, labels 996.
- 208 imagery farms are **not** in the CSV and have **no label** (unusable). 
- **Complete triplets (baseline + current + label + CSV row): 415 farms.** This is the usable analysis set.
- Countries (CSV, all rows): ID 231, BR 196, CM 100, GH 100, IN 81, VN 67, US 46, TH 40, CI 35, AR 33, KE 31, PA 15, CR 15, IT 12, ET 9, AU 9.
  **Malaysia (MY) does not appear at all.** Crops: Cocoa 250, Oil Palm 200, Coffee 200, Cattle 150, Rubber 120, Soy 100.
- Complete triplets by country|crop: ID|Oil Palm 94, CM|Cocoa 72, BR|Cattle 69, VN|Coffee 67, BR|Soy 41, TH|Rubber 40,
  US|Cattle 18, BR|Coffee 14. (Indonesian coffee, Ghana, Cote d'Ivoire, India etc. have labels only, no imagery.)

## Imagery
- Dates: baseline all in 2020-06 (range 2020-06-01..30), current all 2024-06 (2024-06-01..30). One composite scene date per farm-year (4 farms two baseline dates).
- 5 bands float32. Band order inferred from the repo code (`change_dataset.py`): R, G, B, NIR, SCL. Verified by values: bands 1-4 in 0.0-1.56 (reflectance), band 5 integers 0-10 (Sentinel-2 SCL classes; 0 = no data). Sampled 40 farms x 2 years: no NaN; SCL classes present: 4 (vegetation) 10.4 M px, 5 3.8 M, 8/9/10 (cloud medium/high/cirrus) 4.9 M px (~23 % of sampled pixels; class 3 cloud shadow adds 1.3 M), 0 262k.
- Size 512 x 512, EPSG:4326, pixel 1.953e-5 deg (~2.17 m at the equator, i.e. an upsampled/resampled grid, not native 10 m), tile ~0.01 deg (~1.1 km). Single raster signature across all 415 complete farms. nodata tag: None.
- Tile is centred on the CSV farm point (median offset 0.0 m, measured on all 415).

## Labels
- uint8, 1 band, EPSG:4326, pixel 2.6949e-4 deg (~30 m), nodata tag **0** (same value as class 0, see design.md). 16 distinct raster shapes (166-204 x 167-168), ~5 km box. Label extent covers the image footprint 100 % for all 415 farms.
- Values observed: {0, 1, 2} only. Semantics (from `generate_labels.py`): 0 = non-forest or lost <= 2020; 1 = tree cover (>=10 % in 2000) with no loss through 2024; 2 = loss 2021-24.
- Class balance over the 415 complete farms (11.88 M px): 0: 96.7 %, 1: 0.71 %, 2: 2.60 %.
  Among Hansen "forest/loss" pixels (classes 1+2) **loss is 79 %**; by group: ID Oil Palm 75.4 % (71,019 loss vs 23,199 stable), CM Cocoa 75.9 %, TH Rubber 80.1 %, BR Cattle 96.3 %, VN Coffee 90.2 %.
  393 of 415 farms contain at least one class-1/2 pixel; 22 contain none; 393 contain at least one loss pixel; 86 of 94 Indonesian oil palm farms contain loss pixels.
  Such a high loss share inside OSM farm polygons of established crops is itself a warning sign for plantation harvest/replanting being counted as loss (hypothesis, tested in the reference step).
- The class balance is extremely skewed toward class 0, and within forest pixels toward loss, so accuracy is not a usable metric.
