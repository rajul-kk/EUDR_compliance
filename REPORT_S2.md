# What was the land in 2020 under Hansen's 2021-24 loss? (Sentinel-2 / Sentinel-1 extension, registered report)

Protocols: `PROTOCOL_S2.md` (first attempt), `PROTOCOL_S2B.md` (second), `PROTOCOL_S2C.md` (third), each frozen before its run, deviations
appended. Results: `results/s2/`, `results/s2b/`, `results/s2c/` (`landuse.json`, `decompose.json`, `pred_2020.npz`, run logs). Every number
below comes from those files. Design: `docs/s2_design.md`.

## Question
The first audit (docs/results.md section 1) found that 0.51 of Hansen 2021-24 loss cells in the 72 reference grids are palm or crop in the
reference (mostly dated 2023-24), but could not say whether that palm was already there at the EUDR cut-off (replanting, a false flag) or
planted on cleared forest (a real violation). This extension trains a land-use classifier (FOREST / PLANT / OTHER) on each grid's
reference year, checks it on held-out grids, and applies it to 2020 imagery of the same cells.

## Three registered attempts

| | S2 | S2B | S2C |
|---|---|---|---|
| features | annual S2 composite, per cell + 3x3 mean (12) | + NBR, NBR2, block-scale window mean/std (41) | + Sentinel-1 VV/VH annual statistics and context (54) |
| new download | 7.63 GB counted (system-wide counter) | none | 3.65 GB counted |
| FOREST recall, tree-covered cells (gate >= 0.70) | 0.62 [0.51-0.72] | 0.62 [0.50-0.74] | **0.77 [0.67-0.84]** |
| PLANT recall, tree-covered cells (gate >= 0.70) | 0.63 [0.55-0.71] | 0.70 [0.63-0.77] | **0.79 [0.72-0.84]** |
| OTHER recall, tree-covered cells | 0.40 | 0.44 | 0.44 |
| balanced accuracy, all annotated cells | 0.59 | 0.62 | 0.70 |
| transfer to 2020 (7 grids, label_year <= 2021) | 0.61 (passes) | 0.66 (passes) | 0.75 (passes) |
| quality gate | fail | fail | **pass** |
| headline allowed | no | no | **yes** |

Grouped 6-fold CV by grid (folds balanced by island, seed 0); intervals are 95 % grid-cluster bootstrap (2000 resamples). The decision
rules were copied unchanged from S2 into S2B and S2C, so only the features changed. S2B and S2C were written after the earlier results were
seen in aggregate; that is disclosed in each protocol. This is the third look; no multiplicity correction is applied.

Reading: optical annual composites recognise plantation once block-scale context is added (PLANT recall 0.63 -> 0.70) but confuse
natural forest with mature plantation and other land; C-band radar (canopy structure, cloud-independent) is what fixed forest recall.

## Headline (S2C): Hansen 2021-24 loss cells by predicted land use on 31 Dec 2020

177,080 loss cells in 72 grids (0.99 of all loss cells; the rest lacked a valid 2020 optical composite). Confusion-corrected shares
(`pi = solve(C^T, q)`, C re-estimated in every bootstrap resample; 4 % of resamples needed clipping):

| 2020 land use | corrected share [95 %] | raw share | EUDR reading (fixed in advance) |
|---|---|---|---|
| PLANT | **0.49 [0.31-0.65]** | 0.45 | replanting or harvest on land that was not forest at the cut-off: upper part of the false-flag share |
| FOREST | **0.31 [0.15-0.50]** | 0.32 | candidate deforestation |
| OTHER | 0.21 [0.00-0.44] | 0.23 | cleared before the cut-off or not forest: also not EUDR deforestation |

The PLANT interval excludes 0.20, the pre-registered headline condition. **About half of Hansen's post-2020 loss in these grids is on land
the classifier sees as plantation in 2020; roughly a third is on land it sees as forest.**

Checks:
- 7 grids with reference year <= 2021, where the reference itself is close to the cut-off: corrected PLANT 0.61 [0.46-0.80]; the human
  reference gives 0.61 (docs/results.md section 1). Agreement check only, as registered.
- Hansen "stable forest" (y_H = 1) cells predicted PLANT in 2020 (E3): corrected 0.53 [0.41-0.64], raw 0.48. This dates the earlier 0.48
  (2023-24 reference) to the cut-off year: Hansen's 2020 forest mask includes much plantation.
- By island (corrected PLANT): Sulawesi 0.61 [0.12-0.88], Kalimantan 0.39 [0.00-0.61], Sumatra 0.37 [0.00-0.69]. Intervals are wide
  and overlap, and island subsets need heavy clipping (36-57 % of resamples on Sulawesi and Kalimantan); no island difference is claimed.

## What this does and does not show
- It is accuracy **against the reference**, whose authors report 0.83 overall accuracy; the classifier learns its errors too.
- PLANT-in-2020 is an upper part of the false-flag share, not all of it is a false flag in law: the classifier cannot tell harvest of a
  planted forest from palm replanting, and both are outside the EUDR deforestation definition only if the land was not forest in 2020.
- The 72 grids are Indonesian; nothing here is measured for Malaysia or for the user's farms (none lies in a reference grid). A farm-level
  application (PROTOCOL_S2 E4) is transported model output and is reported separately.
- Three attempts were made; the gate and headline rule never changed, but the feature set was chosen after seeing failures.

## Run record
- S2 download: stopped twice at its guard; raised to 6 GB then 9 GB with the owner's approval (PROTOCOL_S2 deviations 3-4); one truncated
  file from a killed run was re-fetched and the downloader made atomic (`00e9fa3`). 143 grid-periods (72 grids, one shares 2020 and ref).
- S1 download: 143 grid-periods, 6 scenes each except one grid-period (47 ref) with fewer; 3.65 GB counted against its 4.5 GB guard.
- Run times on the laptop CPU: S2 18 min, S2B 28 min, S2C 4.2 h (mostly bootstrap). Each registered run executed once.
- Licence: the reference (Zenodo 23874371) is CC BY-NC-SA 4.0; the classifier outputs in `results/s2*/pred_2020.npz` are derived works.
