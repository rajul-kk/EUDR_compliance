# EUDR Pipeline — Command Reference

## Prerequisites

```bash
pip install -r requirements.txt
pip install geotessera        # for TESSERA-embed path
pip install reportlab         # optional — PDF DDS export
```

Set credentials in `.env` (copy from `.env.example`):
```
CDSE_USER=your_copernicus_username
CDSE_PASSWORD=your_copernicus_password
GEE_PROJECT_ID=your_gee_project_id
```

---

## Verification

```bash
# PyTorch GPU check
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s)')"

# AMP check
python -c "import torch; s = torch.cuda.amp.GradScaler(); print('AMP enabled:', s._enabled)"

# Import sanity check
python -m compileall src/
```

---

## Step-by-step pipeline

### 1 — Farm discovery (OpenStreetMap)
```bash
python src/find_farms.py
# Output: inputs/farms_osm.csv
```

### 2 — Enrich farms with country ISO2
```bash
python src/enrich_farm_countries.py
# Output: inputs/farms_osm.csv (in-place update)
```

### 3 — Download Sentinel-2 cloud-free composites
```bash
# 2020 baseline
python src/sentinel_client.py \
  --csv-path inputs/farms_osm.csv \
  --year 2020 \
  --output-dir data/raw_satellite/2020_baseline

# 2024 current
python src/sentinel_client.py \
  --csv-path inputs/farms_osm.csv \
  --year 2024 \
  --output-dir data/raw_satellite/2024_current
```

### 4 — Generate GEE hybrid masks (DynamicWorld + Canopy Height)
```bash
python GEE_dynamic/src/generate_hybrid.py
# Output: data/hybrid_masks/
```

---

## Model training

### DeepLabV3 (GPU — P100 preferred, AMP + DataParallel auto-enabled)
```bash
python src/ML_farm_net.py \
  --raw-dir data/raw_satellite/2020_baseline \
  --mask-dir data/hybrid_masks \
  --output-model-path models/farm_deeplab.pth \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 42
```

### TESSERA head (GPU — frozen encoder, AMP + DataParallel auto-enabled)
```bash
python src/tessera_train.py \
  --raw-dir data/raw_satellite/2020_baseline \
  --mask-dir data/hybrid_masks \
  --output-model-path models/farm_tessera.pth \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --val-ratio 0.15 \
  --patience 3 \
  --num-workers 4 \
  --seed 42
```

### TESSERA-embed (CPU — embed head is 100K params, GPU overhead > benefit)

**3a. Download precomputed GeoTESSERA embeddings (auto bbox from farms CSV)**
```bash
python src/tessera_embedding_generation.py \
  --farms-csv inputs/farms_osm.csv \
  --year 2024 \
  --output-dir data/embeddings \
  --bbox-padding 0.5
```

**3b. Build tile-aligned masks**
```bash
python src/geotessera_mask_tiler.py \
  --tile-tiff-dir data/embeddings/global_0.1_degree_tiff_all \
  --source-mask-dir data/hybrid_masks \
  --out-dir data/geotessera_tile_masks \
  --year 2024 \
  --summary-json reports/geotessera_mask_tiling_summary.json
```

**3c. Train embedding head**
```bash
python src/tessera_embed_train.py \
  --embeddings-dir data/embeddings/global_0.1_degree_representation \
  --mask-dir data/geotessera_tile_masks \
  --output-model-path models/farm_tessera_embed_head.pth \
  --dataset-mode geotessera \
  --year 2024 \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --patience 3 \
  --device cpu \
  --split-manifest-path reports/tessera_embed_split.json
```

---

## Inference

### DeepLabV3 or TESSERA
```bash
python src/inference.py \
  --model-path models/farm_deeplab.pth \
  --input-dir data/raw_satellite/2024_current \
  --output-dir data/predictions_2024_deeplab \
  --model-type deeplab         # or: tessera
```

### TESSERA-embed
```bash
python src/tessera_embed_infer.py \
  --model-path models/farm_tessera_embed_head.pth \
  --embeddings-dir data/embeddings/global_0.1_degree_representation \
  --output-dir data/predictions_2024_tessera-embed \
  --year 2024 \
  --reference-image-dir data/embeddings/global_0.1_degree_tiff_all
```

---

## Deforestation detection + reporting

```bash
# Detect deforestation (set EUDR_PRED_DIR to whichever predictions to use)
EUDR_PRED_DIR=data/predictions_2024_deeplab python src/detect_deforestation.py
```

---

## Audit trail + DDS export

```bash
# Full pipeline with audit log (picks model type via --model-type)
python main_audit.py --model-type deeplab

# With Due Diligence Statement export
python main_audit.py \
  --model-type deeplab \
  --export-dds \
  --operator-name "Acme Trading GmbH" \
  --operator-address "Hauptstrasse 1, 10115 Berlin" \
  --operator-country DE \
  --operator-eori DE123456789 \
  --commodity-hs-code 1801 \
  --commodity-description "Cocoa beans" \
  --commodity-quantity 5000.0 \
  --commodity-unit kg
# Output: reports/dds/dds.json, dds.xml, dds.pdf
```

---

## Full pipeline (one command)

```bash
# DeepLabV3 (skip farm discovery and image download if data already present)
python main_audit.py \
  --model-type deeplab \
  --run-farm-discovery \
  --run-image-download

# TESSERA head
python main_audit.py --model-type tessera

# TESSERA-embed (auto-downloads embeddings if missing)
python main_audit.py --model-type tessera-embed
```

---

## Parallel execution (3 terminals)

| Terminal | Command | Hardware |
|----------|---------|----------|
| A | `python main_audit.py --model-type deeplab --skip-inference` | P100 (training) |
| B | `python src/sentinel_client.py ...` or tessera-embed training | i7 CPU |
| C | `python src/inference.py --model-path models/farm_deeplab.pth ...` | T4 x2 (inference) |

**Rules:**
- DeepLabV3 training + Sentinel download → safe in parallel
- DeepLabV3 training + tessera-embed training → safe (different hardware)
- DeepLabV3 training + tessera-embed inference on T4s → safe
- DeepLabV3 training + TESSERA head training → **do not run together** (same GPU)

---

## Smoke tests

```bash
# 1 epoch DeepLabV3 to confirm AMP works
python src/ML_farm_net.py --epochs 1 --batch-size 8 \
  --raw-dir data/raw_satellite/2020_baseline \
  --mask-dir data/hybrid_masks \
  --output-model-path models/test_deeplab.pth

# Confirm tessera-embed runs on CPU
python src/tessera_embed_train.py --device cpu --epochs 1 \
  --embeddings-dir data/embeddings/global_0.1_degree_representation \
  --mask-dir data/geotessera_tile_masks \
  --output-model-path models/test_embed.pth \
  --dataset-mode geotessera --year 2024
```

---

## Outputs

| Path | Description |
|------|-------------|
| `models/farm_deeplab.pth` | Trained DeepLabV3 checkpoint |
| `models/farm_tessera.pth` | Trained TESSERA head checkpoint |
| `models/farm_tessera_embed_head.pth` | Trained embedding head checkpoint |
| `data/predictions_2024_*/` | Per-farm predicted segmentation masks |
| `reports/deforestation_report.csv` | Farm-level alert levels and metrics |
| `reports/summary_stats.json` | Aggregate statistics |
| `reports/audit_log.jsonl` | Tamper-evident SHA-256 chained audit log |
| `reports/dds/dds.json` | Due Diligence Statement (EU IS API format) |
| `reports/dds/dds.xml` | Due Diligence Statement (TRACES NT format) |
| `reports/dds/dds.pdf` | Due Diligence Statement (human-readable) |
