# TESSERA CPU Mitigation Spec (i7-1355U)

Date: 2026-03-27
Project: EUDR compliance segmentation and deforestation detection
Target machine: Intel i7-1355U (10 cores / 12 threads), Intel Iris Xe, no discrete GPU

## 1) Objective

Deploy a TESSERA-based encoder with a fine-tuned segmentation/classification head while keeping runtime and quality risk under control on CPU-only hardware.

Success means:
- Better or equal deforestation detection quality vs current DeepLab baseline.
- Wall-clock training feasible on an i7 laptop.
- Inference throughput acceptable for batch operation.
- Reproducible outputs and fallback path preserved.

## 2) Current Baseline (from this repo)

- Current training model: DeepLabV3 in src/ML_farm_net.py.
- Current inference model: src/inference.py.
- Change detection and reporting: src/detect_deforestation.py.
- Available local data volume (observed):
  - 2020 TIFFs: ~627
  - 2024 TIFFs: ~623
  - Hybrid masks: ~1248

## 3) Cost Envelope on i7-1355U

These are practical planning ranges for 512x512 tiles and ~1.2k samples.
Real time will vary with model size, dataloader behavior, and thermal throttling.

### 3.1 Training time (CPU only)

- Head-only tuning (encoder frozen):
  - ~35 to 90 minutes per epoch
  - 8 to 15 epochs: ~5 to 22.5 hours
- Partial unfreeze (last encoder blocks + head):
  - ~1.5 to 3.5 hours per epoch
  - 6 to 12 epochs: ~9 to 42 hours
- Full-model fine-tune:
  - ~3 to 6 hours per epoch
  - 5 to 10 epochs: ~15 to 60 hours

### 3.2 Inference time (CPU only)

- 1 tile (512x512): ~0.8 to 3.0 seconds (batch=1)
- 623 current images: ~8 to 31 minutes for pure forward passes
- End-to-end with IO/postprocessing overhead: ~20 to 75 minutes

### 3.3 Electricity cost (local)

Assume average system draw 35W to 65W under sustained training load.

Energy (kWh) = Power (kW) * Time (h)

Examples:
- 10h run at 50W average -> 0.5 kWh
- 24h run at 50W average -> 1.2 kWh

At $0.12 to $0.35 per kWh:
- 10h run: ~$0.06 to $0.18
- 24h run: ~$0.14 to $0.42

Main cost driver is engineer wait-time and iteration speed, not electricity.

## 4) Risk Register and Mitigations

### Risk A: CPU training too slow for iteration

Mitigations:
- Stage 1: freeze encoder, train head only.
- Use cached embeddings for head-only runs.
- Use smaller crops first (384 or 256) then short 512 finetune.
- Trigger early stop after 3 non-improving validations.

Gate:
- If epoch time > 90 minutes in Stage 1, reduce crop size and/or increase sample stride.

### Risk B: Label noise from hybrid masks limits gains

Mitigations:
- Build a curated validation split (300 to 500 tiles) manually audited.
- Confidence filtering for pseudo-labels (drop uncertain regions).
- Reweight classes and use boundary-aware loss in head training.

Gate:
- If validation mIoU gain < 1.5 points after Stage 1, audit labels before unfreezing encoder.

### Risk C: Domain mismatch (TESSERA pretraining vs local data)

Mitigations:
- Match TESSERA expected channel normalization and ordering exactly.
- Add spectral indices as auxiliary channels only if adapter layer supports them.
- Perform per-region evaluation, not only aggregate metrics.

Gate:
- If one region regresses > 3 points F1, block promotion to production.

### Risk D: Production regression in compliance alerts

Mitigations:
- Keep current DeepLab path as fallback branch.
- Calibrate alert thresholds from validation set.
- Run side-by-side reports for at least one full batch before cutover.

Gate:
- If violation false-positive rate increases by > 20% relative, keep DeepLab as primary.

## 5) Implementation Plan (Phased)

### Phase 0: Instrumentation (1 day)

Deliverables:
- Unified training config (model, channels, crop size, loss, lr, seed).
- Benchmark script producing CSV for throughput and metrics.

### Phase 1: Head-only TESSERA (2 to 4 days)

Design:
- Freeze encoder parameters.
- Train segmentation/classification head on current dataset.
- Validate every epoch on curated split.

Expected compute:
- 5 to 20 total training hours on i7.

Promote if:
- Forest F1 and deforestation-change F1 both improve or remain neutral.

### Phase 2: Partial unfreeze (optional, 2 to 5 days)

Design:
- Unfreeze top encoder blocks only.
- Lower LR for encoder, higher LR for head.

Expected compute:
- 10 to 40 additional hours on i7.

Promote if:
- At least +2 points in deforestation-change F1 or strong calibration improvement.

### Phase 3: Report calibration and rollout (1 to 2 days)

Design:
- Refit warning/violation thresholds from validation curves.
- Side-by-side report generation with both models.
- Keep rollback switch.

## 6) Data Requirements for i7-feasible Finetuning

Minimum viable (recommended to start now):
- Existing ~1248 paired tiles + curated 300 to 500 validation tiles.

Better target:
- 2000 to 5000 quality-labeled tiles before partial unfreeze.

Do not block Stage 1 on collecting all new data. Start with current set plus curation.

## 7) Runtime Mitigation Tactics (Concrete)

- Use gradient accumulation to emulate larger batches at batch=1 or 2.
- Enable mixed precision only if numerically stable on target backend.
- Keep num_workers conservative on Windows (0 to 2); benchmark, do not assume.
- Pin deterministic seeds and cache aligned masks to reduce rerun variance.
- Save best-by-metric checkpoints, not only final epoch.

## 8) Acceptance Criteria (Go/No-Go)

Go only if all are true:
- No regression in compliance alert precision on validation set.
- Deforestation-change F1 >= baseline + 1 point (or same F1 with lower false positives).
- End-to-end 2024 batch inference + report generation under 90 minutes on i7.
- Reproducible rerun variance within +/-0.5 metric points.

No-Go triggers:
- Runtime exceeds envelope without measurable quality gains.
- Label noise dominates errors after two mitigation passes.

## 9) Fallback Strategy

- Keep DeepLab training and inference scripts operational in parallel.
- Add model selector flag (deeplab | tessera).
- Produce both reports during trial period.
- If TESSERA fails gates, keep DeepLab as production default and iterate offline.

## 10) Immediate Next Tasks

1. Add a TESSERA model wrapper and head module.
2. Add benchmark script for CPU throughput and validation metrics.
3. Build curated validation subset list (300 to 500 tiles).
4. Run Phase 1 head-only training with strict early stopping.
5. Compare side-by-side deforestation reports before switching default model.
