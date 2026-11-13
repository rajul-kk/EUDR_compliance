# EUDR Compliance AI

Satellite-based forest-loss detection for EUDR supply-chain screening. Pairs 2020 and 2024 Sentinel-2 composites across
~1,000 farms and trains change models on Hansen Global Forest Change labels.

> **Read this first: the labels are not ground truth for EUDR deforestation.** Hansen loss is any stand-replacing loss
> of tree cover (canopy of 10 % or more), which includes harvest and replanting of plantations. In 72 Indonesian
> reference grids, the label audit in this repo finds that about half of Hansen's 2021-24 loss (0.49,
> confusion-corrected) is on land that was already plantation on 31 Dec 2020, which `REPORT_S2.md` reads, as fixed
> before the run, as replanting or harvest rather than deforestation; roughly a third is on land that was forest. That
> reading has not yet been checked against the text of Regulation (EU) 2023/1115. The models below learn Hansen, so
> they inherit this: treat a flag as a lead for review, not a finding. Accuracy figures are against Hansen or against
> the reference maps, not against truth. Details: [Hansen label audit](#hansen-label-audit), `REPORT_S2.md`,
> `docs/results.md`.

[![CI](https://github.com/rajul-kk/EUDR-compliance/actions/workflows/ci.yml/badge.svg)](https://github.com/rajul-kk/EUDR-compliance/actions/workflows/ci.yml)

---

## Architecture

```
OpenStreetMap  ──► find_farms.py           (farm geometries → inputs/farms_osm.csv)
Copernicus     ──► sentinel_client.py      (Sentinel-2 L2A downloads → data/raw_satellite/)
Google EE      ──► GEE_dynamic/generate_labels.py  (Hansen GFC labels → data/hansen_labels/)
               ──► change_siamese_train.py (M3 Siamese-DeepLabV3)  ─┐
               ──► ML_farm_net.py          (M1a DeepLabV3)          ─┤ → models/*.pth
               ──► tessera_train.py        (M1b TESSERA head)       ─┘
               ──► inference.py            (predictions → data/predictions_2024/)
               ──► detect_deforestation.py (change detection → reports/)
```

### Models

| ID | Backend | Description |
|---|---|---|
| M3 | `siamese` | Siamese DeepLabV3-ResNet50 — shared encoder processes t1 & t2, detects change from feature difference |
| M1a | `deeplab` | DeepLabV3-ResNet50 single-image segmentation on 2020 baseline |
| M1b | `tessera` | Frozen ResNet50 encoder + ASPP head (lightweight, faster to train) |
| M1c | `tessera-embed` | Head trained on precomputed GeoTESSERA embeddings (CPU-only) |

**Labels:** Hansen Global Forest Change 2025 v1.13 (`UMD/hansen/global_forest_change_2025_v1_13`). Forest baseline = 31 Dec 2020 (EUDR cutoff). Pixel values: `0` = non-forest, `1` = tree cover in 2020, `2` = tree-cover loss 2021–2024 (not necessarily deforestation; see the note at the top), `255` = no Hansen data (labels generated from 12 Nov 2026 on).

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/rajul-kk/EUDR-compliance.git
cd EUDR-compliance
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env: set CDSE_EMAIL, CDSE_PASSWORD, GEE_PROJECT_ID
```

### 3. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

---

## Running the pipeline

```bash
# Full pipeline (train M1a, infer, detect deforestation)
python main_audit.py

# Include farm discovery and image download
python main_audit.py --run-farm-discovery --run-image-download

# Train M3 Siamese change-detection model
python src/change_siamese_train.py \
  --t1-dir data/raw_satellite/2020_baseline \
  --t2-dir data/raw_satellite/2024_current \
  --mask-dir data/hansen_labels \
  --output-model-path models/farm_siamese.pth \
  --epochs 20 --batch-size 8

# Generate Hansen GFC labels for all farms
python GEE_dynamic/src/generate_labels.py
```

### Kaggle training (T4 ×2 GPU)

Attach the `rajulkabir/eudr-satellite` dataset and run one of:

| Notebook | Model | Runtime |
|---|---|---|
| `notebooks/kaggle_siamese_training.ipynb` | M3 Siamese | ~2 h |
| `notebooks/kaggle_deeplab_training.ipynb` | M1a DeepLabV3 | ~1 h |
| `notebooks/kaggle_tessera_training.ipynb` | M1b TESSERA | ~45 min |

---

## Project structure

```
EUDR-compliance/
├── main_audit.py                     # Main entry point
├── requirements.txt
│
├── src/                              # Core pipeline modules
│   ├── pipeline_runner.py
│   ├── find_farms.py                 # OSM farm discovery
│   ├── sentinel_client.py            # Copernicus image downloader
│   ├── change_siamese_model.py       # M3 Siamese-DeepLabV3
│   ├── change_siamese_train.py       # M3 training script
│   ├── ML_farm_net.py                # M1a DeepLabV3 training
│   ├── tessera_backbone.py           # M1b model
│   ├── tessera_train.py              # M1b training
│   ├── tessera_embed_train.py        # M1c embedding head
│   ├── inference.py                  # Model inference
│   ├── detect_deforestation.py       # Change detection + risk scoring
│   ├── benchmark.py                  # Evaluation metrics
│   ├── audit_trail.py                # Append-only SHA-256 audit log
│   ├── dds_exporter.py               # DDS export: JSON / XML / PDF
│   └── preprocessing/
│       ├── change_dataset.py         # Paired (t1, t2, label) dataset
│       └── histogram_match.py        # Temporal normalisation
│
├── GEE_dynamic/                      # Google Earth Engine integration
│   ├── config.py                     # Asset IDs
│   └── src/
│       └── generate_labels.py        # Hansen GFC label download
│
├── notebooks/                        # Kaggle training notebooks
│   ├── kaggle_siamese_training.ipynb
│   ├── kaggle_deeplab_training.ipynb
│   └── kaggle_tessera_training.ipynb
│
├── scripts/
│   └── validate_notebooks.py         # CI notebook static validator
│
├── tests/                            # pytest suite
│   ├── test_metrics.py
│   ├── test_change_dataset.py
│   ├── test_histogram_match.py
│   └── test_model_forward.py
│
├── data/                             # Runtime data (git-ignored)
│   ├── raw_satellite/
│   │   ├── 2020_baseline/            # Sentinel-2 composites (t1)
│   │   └── 2024_current/             # Sentinel-2 composites (t2)
│   └── hansen_labels/                # Per-farm Hansen GFC labels
│
├── inputs/
│   └── farms_osm.csv                 # ~1,000 farms, 7 EUDR crops
│
└── reports/                          # Pipeline outputs
    ├── deforestation_report.csv
    ├── audit_log.jsonl               # Tamper-evident chain log
    └── dds/                          # Due Diligence Statement exports
        ├── dds.json
        ├── dds.xml
        └── dds.pdf
```

---

## Audit trail & Due Diligence Statements

Every pipeline run appends a tamper-evident entry to `reports/audit_log.jsonl` containing SHA-256 hashes of all inputs, outputs, and a chain hash linking to the previous entry.

```python
from src.audit_trail import AuditLog
log = AuditLog("reports/audit_log.jsonl")
violations = log.verify()
print("Log intact" if not violations else violations)
```

Export Due Diligence Statement drafts, modelled on the DDS content. They are not in the EU Information System format and
cannot be submitted as they are (see `docs/dds_schema_check.md`):

```bash
python main_audit.py \
  --export-dds \
  --operator-name "Acme Trading BV" \
  --operator-country NL \
  --commodity-hs-code 1801 \
  --commodity-description "Cocoa beans, whole or broken, raw or roasted" \
  --commodity-quantity 50000 --commodity-unit kg
```

| File | Format |
|---|---|
| `dds.json` | JSON modelled on the DDS content |
| `dds.xml` | XML modelled on the DDS content (own schema, not TRACES NT) |
| `dds.pdf` | Human-readable summary |

---

## Decision rule

One rule for every model path and for the audit: `hansen_audit/decisions.py`, `plot_decision`. The regulation sets no
percentage tolerance per plot, and its "forest" is land of more than 0.5 ha (Art. 2(4); `docs/regulation_check.md`,
read from a compiled copy and pending a check against EUR-Lex), so a plot is flagged on the area of forest it lost.

| Level | Condition |
|---|---|
| `FLAGGED` | More than 0.5 ha of the plot's 2020 forest lost (6 or more 30 m cells) |
| `CLEAR` | 0.5 ha or less lost |
| `UNCERTAIN` | The loss estimate's interval spans 0.5 ha (when an interval is supplied) |
| `NO_BASELINE_FOREST` | No forest detected in the 2020 baseline |
| `UNASSESSED` | No baseline, no prediction, failed analysis, or unknown pixel size; the reason is reported |

Area comes from each mask's pixel size (its transform and CRS). The DDS export maps `FLAGGED` and `UNCERTAIN` to
manual review, and `NO_BASELINE_FOREST` and `UNASSESSED` to `INCONCLUSIVE`. A flag is a detection to review, not a
legal finding.

The earlier percentage levels (`COMPLIANT` under 5 % of the farm's 2020 forest lost, `WARNING` 5–10 %, `VIOLATION`
over 10 %) remain available with `EUDR_DECISION_RULE=legacy_percent`, for comparison with earlier reports only.

---

## Hansen label audit
How noisy are Hansen loss labels as an EUDR deforestation signal? `PROTOCOL.md` and `docs/results.md` (reference comparison, farm-level
sensitivity, noise-robust training); `REPORT_S2.md` (Sentinel-2/-1 land use at the 2020 cut-off under Hansen 2021-24 loss: about half is
plantation in 2020, a third forest).

## Requirements

- Python 3.10+
- CUDA GPU recommended for training (Kaggle T4 ×2 notebooks provided)
- [Copernicus Data Space](https://dataspace.copernicus.eu/) account (free)
- [Google Earth Engine](https://earthengine.google.com/) project (free for research)
