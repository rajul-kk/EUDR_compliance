# EUDR Compliance AI

Satellite-based pipeline for verifying compliance with the
[EU Deforestation Regulation (EUDR)](https://environment.ec.europa.eu/topics/forests/deforestation/regulation-deforestation-free-products_en).

The system acquires Sentinel-2 imagery, trains a segmentation model to map forest/cropland,
compares 2020 baseline masks against 2024 predictions, and flags farms with significant
forest loss as **VIOLATION**, **WARNING**, or **COMPLIANT**.

---

## Architecture

```
OpenStreetMap  ──► find_farms.py          (farm geometries → inputs/farms_osm.csv)
Copernicus     ──► sentinel_client.py     (Sentinel-2 L2A downloads → data/raw_satellite/)
Google EE      ──► GEE_dynamic/           (DynamicWorld + Canopy Height masks → data/hybrid_masks/)
               ──► ML_farm_net.py         (DeepLabV3 training)   ─┐
               ──► tessera_train.py       (TESSERA head training) ─┤ → models/*.pth
               ──► tessera_embed_train.py (embedding head)        ─┘
               ──► inference.py           (2024 predictions → data/predictions_2024/)
               ──► detect_deforestation.py (change detection → reports/)
```

Model backends (selectable via `--model-type`):

| Backend | Description |
|---|---|
| `deeplab` | DeepLabV3-ResNet50, 6-channel input (RGB + NIR + SCL + NDVI) |
| `tessera` | Lightweight TESSERA wrapper with frozen ResNet50 encoder |
| `tessera-embed` | Head trained on precomputed GeoTESSERA embeddings |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/EUDR-compliance.git
cd EUDR-compliance
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in CDSE_EMAIL, CDSE_PASSWORD, GEE_PROJECT_ID
```

### 3. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

---

## Running the pipeline

```bash
# Full pipeline: train DeepLab, infer, detect deforestation
python main_audit.py

# Include farm discovery and image download steps
python main_audit.py --run-farm-discovery --run-image-download

# Use TESSERA backbone
python main_audit.py --model-type tessera

# Use precomputed GeoTESSERA embeddings
python main_audit.py --model-type tessera-embed

# All options
python main_audit.py --help
```

---

## Project structure

```
EUDR-compliance/
├── main_audit.py                  # Main entry point
├── requirements.txt
├── pyproject.toml
├── .env.example                   # Credential template (copy → .env)
│
├── src/                           # Core pipeline modules
│   ├── pipeline_runner.py         # Orchestrates all steps
│   ├── find_farms.py              # OSM farm discovery
│   ├── sentinel_client.py         # Copernicus image downloader
│   ├── ML_farm_net.py             # DeepLabV3 training
│   ├── tessera_backbone.py        # TESSERA model wrapper
│   ├── tessera_train.py           # TESSERA head training
│   ├── tessera_embed_train.py     # Embedding head training
│   ├── tessera_embed_infer.py     # Embedding inference
│   ├── inference.py               # Model inference
│   ├── detect_deforestation.py    # Change detection
│   ├── benchmark.py               # Evaluation metrics
│   ├── cloud_filter.py            # s2cloudless cloud masking
│   ├── vegetation_index.py        # NDVI / EVI calculation
│   ├── change_detector.py         # ChangeDetector / RiskLevel classes
│   ├── audit_trail.py             # Append-only SHA-256 audit log
│   ├── dds_exporter.py            # DDS export: JSON / XML / PDF
│   └── postprocessing/
│       └── refiner.py             # Morphological mask refinement
│
├── GEE_dynamic/                   # Google Earth Engine integration
│   ├── auth.py                    # GEE authentication (git-ignored)
│   ├── config.py                  # DynamicWorld / Canopy Height asset IDs
│   ├── preprocessing/
│   │   ├── aligner.py             # Rasterio-based mask alignment
│   │   └── dataset_loader.py      # PyTorch Dataset
│   └── src/
│       ├── generate_hybrid.py     # Hybrid mask download
│       └── fusion_engine.py       # DW + Canopy Height fusion
│
├── NDVI_pred/                     # Standalone NDVI utilities
│   └── vegetation_index.py
│
├── data/                          # Runtime data (git-ignored)
│   ├── raw_satellite/             # Downloaded Sentinel-2 GeoTIFFs
│   ├── hybrid_masks/              # GEE-generated masks
│   ├── geotessera_tile_masks/     # TESSERA-aligned masks
│   └── embeddings/                # Precomputed GeoTESSERA embeddings
│
├── inputs/
│   └── farms_osm.csv              # Farm locations and crop types
│
└── reports/                       # Pipeline outputs
    ├── deforestation_report.csv
    ├── summary_stats.json
    ├── audit_log.jsonl            # Append-only tamper-evident audit log
    ├── vectors/                   # GeoJSON deforestation polygons
    └── dds/                       # Due Diligence Statement exports
        ├── dds.json               # EU IS API format
        ├── dds.xml                # TRACES NT XML format
        └── dds.pdf                # Human-readable summary
```

---

## Audit trail & Due Diligence Statements

Every pipeline run that performs inference automatically appends a tamper-evident
entry to `reports/audit_log.jsonl`.  Each entry contains:

- SHA-256 hashes of every input satellite image and output prediction mask
- SHA-256 hash of the deforestation report CSV
- A **chain hash** linking each entry to the previous one (any modification to a
  past entry breaks all subsequent chain hashes)
- Model type, git SHA, run UUID, and UTC timestamp

Verify the integrity of the full log at any time:

```python
from src.audit_trail import AuditLog
log = AuditLog("reports/audit_log.jsonl")
violations = log.verify()
print("Log intact" if not violations else violations)
```

### Exporting Due Diligence Statements (DDS)

Pass `--export-dds` with operator and commodity metadata to generate
EU IS-compatible reports after detection:

```bash
python main_audit.py \
  --model-type deeplab \
  --export-dds \
  --operator-name "Acme Trading BV" \
  --operator-address "Herengracht 1, Amsterdam, NL" \
  --operator-country NL \
  --operator-eori NL123456789 \
  --commodity-hs-code 1801 \
  --commodity-description "Cocoa beans, whole or broken, raw or roasted" \
  --commodity-quantity 50000 \
  --commodity-unit kg
```

Three files are written to `reports/dds/`:

| File | Format | Purpose |
|---|---|---|
| `dds.json` | EU IS API JSON | Machine submission to the EU Information System |
| `dds.xml` | TRACES NT XML | Submission via TRACES NT |
| `dds.pdf` | PDF | Human-readable summary for auditors and operators |

---

## Output risk levels

| Level | Condition |
|---|---|
| `COMPLIANT` | < 5 % forest loss since 2020 |
| `WARNING` | 5–10 % forest loss |
| `VIOLATION` | > 10 % forest loss |
| `NO_BASELINE_FOREST` | No forest detected in 2020 baseline |

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for DeepLabV3 / TESSERA training
- [Copernicus Data Space](https://dataspace.copernicus.eu/) account (free)
- [Google Earth Engine](https://earthengine.google.com/) project (free for research)
