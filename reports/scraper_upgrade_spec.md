# Farm Image Scraper Upgrade Spec

Date: 2026-03-27
Target file: src/sentinel_client.py

## 1) Why the current loop is slow/risky

Current behavior:
- Per farm: two STAC searches (2020 and 2024) in find_cleanest_date.
- Per farm: two image process requests in download_farm_image.
- Date windows are hardcoded to June only.
- Bounding box is fixed via delta around a point.

Primary issues:
- O(farms x years) catalog requests with repeated nearby queries.
- Fixed month window can miss phenology in non-EU climates.
- Point-based box can clip parcel geometry and increase label noise.
- Limited retry/backoff and no persistent job state.

## 2) Upgrade goals

- Reduce API calls per farm.
- Improve image quality selection beyond cloud cover metadata.
- Add region-aware date windows (EU + optional other countries).
- Keep deterministic outputs and resume capability.

## 3) Architecture changes

### A. Country-aware acquisition profile

Add per-country profile settings:
- preferred months per year (or seasonal windows)
- max cloud threshold
- minimum solar elevation
- buffer/scale defaults

Example profile keys:
- EU default: June to August
- Tropics default: dry season months by hemisphere

### B. Replace point buffer with parcel geometry where available

If farm polygon exists:
- use polygon bounds directly for STAC search and process request region.
Else:
- fallback to point buffer logic.

### C. Candidate ranking instead of first cleanest date

For each year, get top N candidates and rank using score:
score = w1 * cloud_cover + w2 * view_angle + w3 * temporal_distance + w4 * no_data_ratio

Pick best valid scene after quick quality checks.

### D. Batch metadata fetch + cached selection

- Query STAC by region chunks and time range once.
- Cache scene candidates per farm-year in cache/scene_candidates.json.
- Reuse cache to avoid repeated catalog calls during reruns.

### E. Resume-safe job queue

Track states in a persistent manifest:
- pending
- selected_scene
- downloaded
- failed

Store in reports/download_manifest.csv or data/manifest/*.json.

### F. Robust retry strategy

- Exponential backoff with jitter for catalog and process endpoints.
- Retry 429, 500, 502, 503, 504.
- Token refresh on 401 and retry once.

## 4) Optional multi-country support

Input CSV additions:
- country_iso2 (recommended)
- hemisphere or climate_zone (optional)
- geometry_wkt (optional but preferred)

Behavior:
- Apply country-specific profile automatically.
- If country missing, infer from coordinates using a country boundary lookup.

## 5) Concrete code changes in src/sentinel_client.py

Add functions:
- load_acquisition_profiles(path)
- infer_country(lat, lon)
- get_time_windows(year, country_code)
- search_candidate_scenes(aoi, time_window)
- rank_scenes(candidates, aoi)
- download_scene(scene_id, aoi, output_path)
- update_manifest(record)

Modify:
- find_cleanest_date -> replace with select_best_scene_for_year
- process_single_farm -> no hardcoded June dates
- download_all_farms -> supports profile_path and target_countries filters

CLI args to add:
- --countries "DE,FR,ES" (optional)
- --year-pairs "2020:2024"
- --profile "inputs/acquisition_profiles.yaml"
- --max-workers 4
- --resume

## 6) Performance impact (expected)

- 30% to 60% fewer catalog calls with cached/batched candidate retrieval.
- Higher success rate for non-EU or mixed-climate farms.
- Better reproducibility and easier failure recovery.

## 7) Minimum viable implementation order

1. Add manifest + retry/backoff + resume.
2. Parameterize date windows and country profiles.
3. Add scene candidate cache and ranking.
4. Add polygon AOI support.
5. Add country inference for missing metadata.

## 8) Validation checks

- Download success rate per country and year.
- Mean cloud cover and no-data ratio of selected scenes.
- Runtime per 100 farms before vs after.
- Downstream segmentation quality delta on same farm subset.
