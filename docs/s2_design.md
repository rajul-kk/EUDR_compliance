# Design: Sentinel-2 over the reference grids (extension S2 of the Hansen audit)

Written before any Sentinel-2 data was downloaded or any result computed. Pre-registered rules are in `PROTOCOL_S2.md`.

## Why
The first audit hit three walls (docs/results.md):
1. Imagery existed only for the user's farms (7.6k usable 30 m cells, 94 ID oil palm farms), and none of those farms lies in a reference grid.
2. The human reference (Zenodo 23874371, 72 grids of 6 x 6 km in Indonesia) has no imagery attached, so it could only be compared with Hansen, not used to train or check a model.
3. Most reference polygons are dated 2023-24. A Hansen 2021-24 loss cell that is palm in 2023-24 could be replanting (a false flag) or forest cleared and planted (a real violation). That split was **not identified** (PROTOCOL deviation 1).

Sentinel-2 L2A is free and covers every grid in every year, so wall 3 can be attacked directly. **Train a land-use classifier (FOREST / PLANT / OTHER) on Sentinel-2 of each grid's reference year, check it on held-out grids, and apply it to the 2020 imagery of the same cells.** That gives each Hansen loss cell a predicted land use at the EUDR cut-off date (31 Dec 2020), with a measured error rate. Plantation in 2020 followed by loss is replanting or harvest, not deforestation.

This turns about 3.4M reference cells into training and test data. The earlier setup had 7.6k cells.

## Data
| Item | Source | Notes |
|---|---|---|
| Sentinel-2 L2A, bands B02 B03 B04 B08 (10 m), B11 B12 SCL (20 m) | Microsoft Planetary Computer STAC `sentinel-2-l2a`, anonymous SAS token | No account or key needed (checked on 2026-11-05). Windowed COG reads, so only the grid window is fetched. |
| Hansen labels, reference classes | existing `D:/Work/eudr-hansen-audit-refdata/reference_cells.npz` | The same 0.00025 deg grid per reference grid (`origin_XX`) |
| Grid metadata | `zenodo_23874371/grid_index.csv` | label_year, island, province |

Storage: `D:/Work/eudr-hansen-audit-refdata/s2/` (outside the read-only data folder), one `grid_XX.npz` per grid and period, already resampled to the Hansen grid. Raw scenes are not kept.

Download cap: **5 GB in total** (the cap carried over from the original instructions). This is enforced by a guard that counts **system-wide** received bytes through psutil, an upper bound on our traffic, and stops the downloader at 4.5 GB.

## Pipeline
```
grid_index + reference_cells.npz (origin, shape per grid)
  s2.py     search(bbox, period) -> items ; harmonise(DN, baseline) ; scene ranking by in-grid clear fraction (SCL read first, 20 m, cheap)
            read bands for the best K=6 scenes per period -> mask SCL {0,1,3,8,9,10} -> per-pixel median (10 m, UTM)
            reproject to the Hansen grid (average) + per-cell NDVI std from 10 m pixels (texture)
  -> s2/grid_XX_{2020,ref}.npz : (C, H, W) float32 features + valid-obs count
  s2features.py  per cell: 6 bands, NDVI, NDMI, NDVI-std, 3x3 neighbourhood means of NDVI/NDMI/B08 -> tabular
  landuse.py     LightGBM 3-class, grouped 6-fold CV by grid (stratified by island) -> out-of-fold P(class) at ref year
                 confusion matrix C[true, pred] on tree-covered cells (y_H in {1,2}), the population we apply it to
                 temporal check on the 7 grids with label_year <= 2021
                 apply each fold's model to its held-out grids' 2020 composite -> P_2020(class) per cell
  decompose.py   Hansen loss cells x predicted 2020 class -> raw shares ; corrected shares (confusion-matrix inversion) ;
                 cluster bootstrap over grids (re-estimating C inside each resample)
scripts/hansen_audit/s2_download.py, s2_build_features.py, s2_run_landuse.py, s2_run_decompose.py
```

## Decisions (and what was rejected)
| Decision | Chosen | Rejected and why |
|---|---|---|
| Data source | Planetary Computer STAC (anonymous) | GEE: needs the user's account and auth in this session; Copernicus Data Space: needs an account; AWS `sentinel-cogs`: also anonymous, but its L2A predates the harmonised reprocessing and the metadata on PC is easier to query |
| Where to fetch | the 72 reference grids | Malaysian tiles: there is no human reference in Malaysia, so any number there is a transported model output. It is kept as an optional descriptive step, run only if budget remains (PROTOCOL_S2 E4) |
| Resolution | read 10 m natively, average onto the Hansen 0.00025 deg grid | read 20 m overviews: 4x cheaper but loses the crown and row texture that separates palm from forest; work at 10 m: labels are 30 m, and 10 m would fake sample size |
| Compositing | per-pixel median of SCL-clear pixels, best 6 scenes by in-grid clear fraction, whole calendar year | single best scene: clouds in the tropics leave holes; seasonal composites: too few clear scenes in a year |
| Processing baseline | subtract 1000 DN when `s2:processing_baseline` >= 04.00 | ignore it: 2022+ scenes carry a +1000 offset and 2020 scenes don't, which would shift every reflectance between training and application years |
| Classifier | LightGBM on per-cell plus 3x3 neighbourhood features | CNN: CPU-only laptop, and tabular boosting is strong at this scale; per-pixel RF: slower, no better |
| Classes | FOREST / PLANT / OTHER (protocol groups) | full 16 classes: rare classes are noisy and not needed for the EUDR question |
| Validation | grouped 6-fold by grid, out-of-fold for every grid | random cell split: neighbouring cells leak, so accuracy would look far too high |
| Temporal transfer | trained at ref year, applied to 2020; checked on grids whose reference year is <= 2021 | assume it: phenology, sensor state and the processing baseline all change |
| Error correction | invert the confusion matrix (P(pred \| true)) on y_H in {1,2} cells, and bootstrap both steps | report raw predicted shares only: classifier error biases shares toward the majority class |
| Intervals | cluster bootstrap over grids, 2000 resamples, percentile | cell bootstrap: ignores spatial correlation |

## Risks
- **Temporal transfer**: the classifier never sees a label in 2020 (except 1 grid). The <=2021 grid check has only 7 grids, so its interval will be wide.
- **Young plantations**: newly planted palm (< 3 years) looks like bare or shrub. Forest cleared in 2019-20 and planted shows up as OTHER in 2020, which is fine for the EUDR question (it was not forest at the cut-off) but should be named.
- **Reference error**: 0.83 OA per its authors; the classifier learns that error too. Confusion against the reference is not confusion against truth.
- **Clouds**: Sumatra and Kalimantan are very cloudy, so some cells will lack clear observations in 2020; they are dropped and their share reported.
- **Licence**: the reference is CC BY-NC-SA 4.0. The trained classifier and maps are derived works: non-commercial, share-alike.
- **Laptop memory**: the edalint P9 runs are sharing the machine. Downloading is light, but training waits until those runs finish.
