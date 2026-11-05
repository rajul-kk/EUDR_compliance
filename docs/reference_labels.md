# Reference labels: what can serve, and its own error

## Decision
| Candidate | Use | Reason |
|---|---|---|
| Zenodo 10.5281/zenodo.23874371 (Wafiq et al., latest version of the 2509.08303 dataset): human-digitised land-use polygons, 72 grid cells of 6 x 6 km, 10 Indonesian provinces, 47,646 polygons | **Used** as reference for Hansen-vs-land-use composition | Human annotation, published OA 0.83 / kappa 0.76 (on the 6-class grouped validation of the arXiv paper's version; oil palm PA 0.96, UA 0.88), wall-to-wall. Limits: **no overlap with any of the user's farms** (0 of 231 Indonesian farms within 5 km of any grid); reference year mostly 2023-24 (2019: 2 grids, 2020: 1, 2021: 4, 2022: 3, 2023: 44, 2024: 15, 2025: 3); licence **CC BY-NC-SA 4.0** (non-commercial; the arXiv abstract says CC BY 4.0, the record says otherwise, the stricter one is followed); the superseded v0.1 record 15618532 has no files. |
| Zenodo 10.5281/zenodo.17768444 (Kuapanich et al.) oil palm maps 2020/2022/2024, 3 x ~700 MB | **Not downloaded** | Model outputs with 60-71 % overall accuracy: using them as truth would swap one noisy label for another. Possible later use only as a plantation mask, with that error carried. |
| Manual visual checks of the user's farms | **Not done** | Needs high-resolution imagery review by a person; a protocol-compatible next step (stratified sample of 100 flagged farms). |

Downloads: `D:/Work/eudr-hansen-audit-refdata/` (outside the data folder, 1.2 GB incl. extracted copy). sha256 (see also `SHA256SUMS.txt` there; md5 of labels.zip matches Zenodo's c92bebc164703264d2909b000ea9d5a1):
- labels.zip `4e7e3837e5e89dbe8e43cb7d854ff3f257a045f75806c138db919fc2e9980c76` (265,035,152 bytes)
- grid_index.csv `71d820b2fe2768c6fcdc50f7cb22200f055119e5954036dcbd2a22574724a8ab`; class_schema.csv `de8288acffe563a1a15254d7201635be8439073828ca05a30771e3fb8fc5bad7`; README.md `c48ce169519599744e2550ba123612b2bb26dbf19f0594ed030e02063431aa16`
- derived `reference_cells.npz` `fc771c8b42faa4ef86c24d44fb721b18bc745f8f8703145f739288b3da4c0c68`
- Hansen tiles: not downloaded; windows read over HTTP from the public GFC-2025-v1.13 bucket (`earthenginepartners-hansen`), no key. Layers treecover2000 and lossyear.

## Method (hansen_audit/reference.py, scripts/hansen_audit/build_reference_cells.py)
Hansen label re-derived with the identical recipe as `generate_labels.py` on Hansen's native 0.00025 deg grid (the production labels were resampled by GEE to a 0.000269 deg grid; this resampling difference is not quantified). Polygons rasterised at 5x supersampling, majority class per cell; cells with < 50 % annotated area dropped (nearly all cells are >= 98 % annotated).
Groups: FOREST = Forest, Mangrove; PLANT = oil palm (both stages), cacao, coconut, rubber, rice, other agricultural; OTHER = rest (shrubland, grassland, bare, built, water, wetland, unknown).

## Results (real, computed; cluster bootstrap over grids, 2000 resamples, 95 % interval)
| subset | P(PLANT|loss) | P(FOREST|loss) | P(OTHER|loss) | P(PLANT|stable_forest) | P(hansen>=1|PLANT) | P(hansen=0|FOREST) |
|---|---|---|---|---|---|---|
| all 72 grids | 0.509 [0.417, 0.602] | 0.150 [0.098, 0.209] | 0.341 [0.259, 0.428] | 0.481 [0.378, 0.586] | 0.357 [0.293, 0.424] | 0.320 [0.246, 0.406] |
| ref year <= 2021 (7 grids) | 0.609 [0.497, 0.763] | 0.268 [0.049, 0.410] | 0.122 [0.054, 0.249] | 0.540 [0.242, 0.817] | 0.347 [0.224, 0.483] | 0.169 [0.088, 0.307] |
| ref year >= 2022 (65 grids) | 0.492 [0.384, 0.599] | 0.129 [0.083, 0.183] | 0.379 [0.290, 0.469] | 0.475 [0.362, 0.577] | 0.358 [0.284, 0.435] | 0.336 [0.253, 0.424] |

By island (point estimates):
| island | loss cells | P(PLANT|loss) | P(PLANT|stable) | P(hansen=0|FOREST) |
|---|---|---|---|---|
| Kalimantan | 43910 | 0.390 | 0.209 | 0.279 |
| Sulawesi | 42090 | 0.507 | 0.480 | 0.154 |
| Sumatra | 91104 | 0.568 | 0.636 | 0.474 |

Reading, with the limits above:
- **Tree-cover baseline contains plantation.** About half of Hansen "stable forest" cells (no loss 2001-2024, treecover2000 >= 10 %) lie on mapped palm/crops in 2023-24; this is robust to the reference-year issue because the pixel never lost tree cover. Only ~28 % of them are FOREST.
- **Hansen baseline also misses forest:** 0.32 of reference-FOREST cells are class 0 (95 % 0.25-0.41). Class 0 includes forest lost before 2021 and cover below 10 %, so part of this is legitimate, not error (not separated here).
- **Loss composition is ambiguous**: 0.51 of 2021-24 loss cells are palm/crop in the reference, only 0.15 still forest. In the 7 grids referenced at <= 2021, 0.61 are already palm/crop, which is the nearest evidence for replanting/harvest being counted as loss; 7 grids is a small sample and the interval is wide.
- Not identified: true post-2020 conversion and Hansen recall for it (no post-2020 ground truth).
- Own error of the reference: annotator disagreement was measured by the authors (25 % double-annotated) but is not propagated here; polygon edge / registration error vs a 28 m grid is not modelled; the 83 % OA means some PLANT/FOREST mislabels. Treat all fractions as having an extra uncertainty beyond the bootstrap interval.
