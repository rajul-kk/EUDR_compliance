# Design: Hansen label-noise audit (written before results)

Location: `hansen_audit/` package inside the existing EUDR-compliance repo (branch `feat/hansen-noise-audit`,
git worktree `D:/Work/Segmentation-logistics/EUDR-compliance-hansen-audit`; the main checkout has untracked files only).
The model code (Siamese DeepLab, `src/change_siamese_*.py`, label recipe `GEE_dynamic/src/generate_labels.py`) is
reused conceptually, not imported: this package has no GEE/torch-vision dependency and must run on a CPU laptop.
Data: `D:/Work/eudr-satellite-upload`, **read-only**. Reference downloads: `D:/Work/eudr-hansen-audit-refdata` (outside
data folder, sha256 recorded in `docs/reference_labels.md`).

## Label semantics (assumptions, verified where noted)
Source: `generate_labels.py`. Value 0 = not (treecover2000 >= 10 %), or forest lost before 2021; 1 = forest in 2020
(treecover2000 >= 10 %, no loss <= 2020, and no loss 2021-24); 2 = loss in 2021-2024 (lossyear 21-24 on treecover >= 10 %).
Threats to meaning (all measured/handled in the audit, not assumed away):
1. "Forest" is canopy cover >= 10 %, i.e. tree cover, so mature oil palm, rubber, cacao shade, agroforestry count as forest.
2. Loss = any stand-replacing disturbance incl. plantation harvest/replanting; EUDR deforestation needs forest -> agriculture conversion.
3. GeoTIFF nodata is 0, which collides with class 0. We treat 0 as a real class (the downloader clips a box fully covered by data); verified by
   the label extent covering the image footprint 100 %.
4. Hansen is 30 m, imagery is ~10 m resampled to 2.17 m grid; label is a 5 km box, image a 1.1 km box centred on the farm point (offset 0 m, measured).
5. Inventory shows data are global, 6 crops, **no Malaysia**; the Indonesian oil palm subset is the on-topic slice.

## Module layout / data flow
```
farms_osm.csv + 2020_baseline + 2024_current + hansen_labels  (read-only)
   io.py        file-name parsing, farm key join
   inventory.py compute counts/CRS/bands/class balance        -> results/inventory.json -> docs/data_inventory.md
   align.py     window Hansen label onto image grid
   features.py  image pairs -> 30 m cell features (block means, SCL-masked)
Reference: Zenodo 23874371 polygons + Hansen tiles (public, GFC-2025-v1.13) re-derived with the same recipe
   reference.py rasterise polygons to the Hansen 30 m grid; reference-vs-Hansen confusion
   noise.py     transition matrices: reference-based (+Wilson/bootstrap), anchor-point (Patrini 2017) from model outputs
   losses.py    CE, GCE, symmetric CE, forward-corrected CE
   model.py     small CNN on 3x3-cell neighbourhoods (CPU) ; train.py ensembles with farm-level splits
   uncertainty.py  ensemble spread + split conformal at farm level
   decisions.py farm flag rule, false-/missed-flag rates, cluster (farm-level) bootstrap
scripts/hansen_audit/*.py  runnable entry points ;  notebooks/kaggle_hansen_audit.ipynb  (GPU/longer runs)
```

## Decisions (alternatives considered)
| Decision | Chosen | Rejected and why |
|---|---|---|
| Home | package in existing repo, own worktree | new repo: model code was found, so reuse the repo; editing in main checkout: it is dirty |
| Task framing | binary loss-vs-no-loss on pixels that Hansen labels forest/loss (classes 1,2); class 0 ignored | 3-class: class 0 mixes "never forest" with "pre-2021 loss", not label noise of interest |
| Unit of supervision | 30 m cell (the label's native grid) | 2 m pixels: replicates one label value 14x14, fake sample size and fake precision |
| Model | small CNN on 3x3 neighbourhood of 30 m cell features (t1, t2, diff, NDVI) | DeepLab/Siamese: needs GPU (packaged for Kaggle only); pixel GBM: no spatial context, no loss choice |
| Robust loss | primary GCE (q=0.7); also forward-corrected CE with estimated T; SCE as comparison | MAE: slow to train; co-teaching: needs two nets, harder to attribute effects; Forward needs T so it also tests the estimator |
| T estimation | (a) reference-based confusion where polygons exist; (b) anchor-point estimator on ensemble outputs | (b) only: anchor assumption (some clean examples) is doubtful for plantation noise, so it is reported as secondary |
| Uncertainty | 5-member ensemble (CPU) + split conformal on farm score | MC-dropout: weaker; conformal only: no epistemic signal. Conformal guarantees are w.r.t. the *noisy* label and say so |
| Evaluation split | by farm (grouped), and by country/crop for transfer | pixel split: spatial leakage inside one 5 km box |
| CI | cluster bootstrap over farms, 2000 resamples, percentile | pixel bootstrap: ignores within-farm correlation, intervals too tight |
| Missed flags | identifiable only as baseline omission (ref forest, Hansen 0) and via sensitivity over assumed Hansen recall | claiming a missed-flag rate: the reference is a 2019/20 snapshot, post-2020 conversion is invisible to it |

## Risks
- Reference cells (72 x 36 km2) likely do not overlap the user's farms; then reference-derived T is *transported*, an assumption, and farm-level rates are sensitivity analyses.
- Reference licence CC BY-NC-SA 4.0 (research only); reference OA 0.83 (published), annotators' error is not removed.
- Only 94 Indonesian oil palm farms have all of baseline, current and label; small n, wide intervals.
- Model quality vs. truth cannot be measured on farms (no truth). Real-data model results measure agreement with Hansen only; noise-robustness claims rest on semi-synthetic injected noise with known T.
- Images are June composites; cloud/SCL masking and phenology can look like change.
