
import logging
import subprocess
import sys
import os
import argparse
import json
import time

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path for src.* imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_early = os.path.dirname(_src_dir)
if _project_root_early not in sys.path:
    sys.path.insert(0, _project_root_early)

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
find_farms_script = os.path.join(current_dir, 'find_farms.py')
sentinel_script = os.path.join(current_dir, 'sentinel_client.py')
generate_script = os.path.join(project_root, 'GEE_dynamic', 'src', 'generate_hybrid.py')
train_script = os.path.join(current_dir, 'ML_farm_net.py')
tessera_train_script = os.path.join(current_dir, 'tessera_train.py')
inference_script = os.path.join(current_dir, 'inference.py')
tessera_embed_train_script = os.path.join(current_dir, 'tessera_embed_train.py')
tessera_embed_infer_script = os.path.join(current_dir, 'tessera_embed_infer.py')
detection_script = os.path.join(current_dir, 'detect_deforestation.py')
benchmark_script = os.path.join(current_dir, 'benchmark.py')


def run_pipeline(
    skip_farm_discovery=True,
    skip_image_download=True,
    skip_inference=True,
    skip_baseline_metrics=False,
    model_type='deeplab',
    model_path=None,
    export_dds=False,
    operator_name="",
    operator_address="",
    operator_country="",
    operator_eori="",
    commodity_hs_code="",
    commodity_description="",
    commodity_quantity=0.0,
    commodity_unit="kg",
):
    logger.info("Starting EUDR compliance pipeline | root=%s model=%s", project_root, model_type)
    python_exe = sys.executable

    # --- Step 1: Find Farms from OSM ---
    if not skip_farm_discovery:
        logger.info("[1/8] Discovering farm locations from OpenStreetMap")
        try:
            if not os.path.exists(find_farms_script):
                raise FileNotFoundError(f"Script not found: {find_farms_script}")
            subprocess.check_call([python_exe, find_farms_script], cwd=project_root)
            logger.info("Farm discovery complete")
        except subprocess.CalledProcessError as e:
            logger.warning("Farm discovery failed (exit %d) — continuing with existing CSV", e.returncode)
        except Exception as e:
            logger.warning("Farm discovery error: %s — continuing with existing CSV", e)
    else:
        logger.info("[1/8] Skipping farm discovery")

    # --- Step 2: Download Satellite Images ---
    if not skip_image_download:
        logger.info("[2/8] Downloading Sentinel-2 imagery")
        try:
            if not os.path.exists(sentinel_script):
                raise FileNotFoundError(f"Script not found: {sentinel_script}")
            subprocess.check_call([python_exe, sentinel_script], cwd=project_root)
            logger.info("Image download complete")
        except subprocess.CalledProcessError as e:
            logger.error("Image download failed (exit %d)", e.returncode)
            return
        except Exception as e:
            logger.error("Image download error: %s", e)
            return
    else:
        logger.info("[2/8] Skipping image download")

    # --- Step 3: Generate/Download Masks ---
    logger.info("[3/8] Generating hybrid masks from GEE")
    try:
        if not os.path.exists(generate_script):
            raise FileNotFoundError(f"Script not found: {generate_script}")
        subprocess.check_call([python_exe, generate_script], cwd=project_root)
        logger.info("Mask generation complete")
    except subprocess.CalledProcessError as e:
        logger.error("Mask generation failed (exit %d)", e.returncode)
        return
    except Exception as e:
        logger.error("Mask generation error: %s", e)
        return

    # --- Step 3b: Download embeddings (tessera-embed only) ---
    tessera_embed_gen_script = os.path.join(current_dir, 'tessera_embedding_generation.py')
    farms_csv = os.path.join(project_root, 'inputs', 'farms_osm.csv')
    embeddings_dir = os.path.join(project_root, 'data', 'embeddings', 'global_0.1_degree_representation')

    if model_type == 'tessera-embed':
        raw_emb_dir = os.path.join(project_root, 'data', 'embeddings')
        if os.path.isdir(embeddings_dir) and any(True for _ in os.scandir(embeddings_dir)):
            logger.info("[3b/8] Embeddings already present in %s — skipping download", embeddings_dir)
        else:
            logger.info("[3b/8] Downloading precomputed TESSERA embeddings (bbox from farms CSV)")
            try:
                if not os.path.exists(tessera_embed_gen_script):
                    raise FileNotFoundError(f"Script not found: {tessera_embed_gen_script}")
                subprocess.check_call(
                    [python_exe, tessera_embed_gen_script,
                     '--farms-csv', farms_csv,
                     '--year', '2024',
                     '--output-dir', raw_emb_dir],
                    cwd=project_root,
                )
                logger.info("Embedding download complete")
            except subprocess.CalledProcessError as e:
                logger.error("Embedding download failed (exit %d)", e.returncode)
                return
            except Exception as e:
                logger.error("Embedding download error: %s", e)
                return

    # --- Step 4: Train Model ---
    logger.info("[4/8] Training %s model", model_type.upper())
    try:
        if model_type == 'deeplab':
            training_script = train_script
        elif model_type == 'tessera':
            training_script = tessera_train_script
        elif model_type == 'tessera-embed':
            training_script = tessera_embed_train_script
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if not os.path.exists(training_script):
            raise FileNotFoundError(f"Script not found: {training_script}")

        if model_type == 'deeplab':
            subprocess.check_call([python_exe, training_script], cwd=project_root)
            if model_path is None:
                model_path = os.path.join(project_root, 'models', 'farm_deeplab.pth')
        elif model_type == 'tessera':
            if model_path is None:
                model_path = os.path.join(project_root, 'models', 'farm_tessera.pth')
            raw_dir = os.path.join(project_root, 'data', 'raw_satellite', '2020_baseline')
            mask_dir = os.path.join(project_root, 'data', 'hybrid_masks')
            subprocess.check_call(
                [python_exe, training_script,
                 '--raw-dir', raw_dir, '--mask-dir', mask_dir,
                 '--output-model-path', model_path],
                cwd=project_root,
            )
        else:
            if model_path is None:
                model_path = os.path.join(project_root, 'models', 'farm_tessera_embed_head.pth')
            mask_dir = os.path.join(project_root, 'data', 'geotessera_tile_masks')
            subprocess.check_call(
                [python_exe, training_script,
                 '--embeddings-dir', embeddings_dir, '--mask-dir', mask_dir,
                 '--output-model-path', model_path,
                 '--dataset-mode', 'geotessera', '--year', '2024'],
                cwd=project_root,
            )
        logger.info("Training complete — model: %s", model_path)
    except subprocess.CalledProcessError as e:
        logger.error("Training failed (exit %d)", e.returncode)
        return
    except Exception as e:
        logger.error("Training error: %s", e)
        return

    inference_seconds = None

    # --- Step 5: Run Inference ---
    if not skip_inference:
        logger.info("[5/8] Running inference on 2024 imagery")
        try:
            if model_path is None:
                model_path = os.path.join(project_root, 'models', 'farm_deeplab.pth')
            output_dir = os.path.join(project_root, 'data', f'predictions_2024_{model_type}')
            inference_start = time.time()

            if model_type == 'tessera-embed':
                if not os.path.exists(tessera_embed_infer_script):
                    raise FileNotFoundError(f"Script not found: {tessera_embed_infer_script}")
                emb_dir = os.path.join(project_root, 'data', 'embeddings', 'global_0.1_degree_representation')
                ref_dir = os.path.join(project_root, 'data', 'embeddings', 'global_0.1_degree_tiff_all')
                subprocess.check_call(
                    [python_exe, tessera_embed_infer_script,
                     '--model-path', model_path, '--embeddings-dir', emb_dir,
                     '--output-dir', output_dir, '--year', '2024',
                     '--reference-image-dir', ref_dir],
                    cwd=project_root,
                )
            else:
                if not os.path.exists(inference_script):
                    raise FileNotFoundError(f"Script not found: {inference_script}")
                input_dir = os.path.join(project_root, 'data', 'raw_satellite', '2024_current')
                subprocess.check_call(
                    [python_exe, inference_script,
                     '--model-path', model_path, '--input-dir', input_dir,
                     '--output-dir', output_dir, '--model-type', model_type],
                    cwd=project_root,
                )

            inference_seconds = time.time() - inference_start
            logger.info("Inference complete (%.1fs)", inference_seconds)
        except subprocess.CalledProcessError as e:
            logger.error("Inference failed (exit %d)", e.returncode)
            return
        except Exception as e:
            logger.error("Inference error: %s", e)
            return
    else:
        logger.info("[5/8] Skipping inference")

    # --- Step 6: Export Baseline Metrics ---
    if not skip_inference and not skip_baseline_metrics and model_type == 'deeplab':
        logger.info("[6/8] Exporting DeepLab baseline metrics")
        try:
            if not os.path.exists(benchmark_script):
                raise FileNotFoundError(f"Script not found: {benchmark_script}")
            prediction_dir = os.path.join(project_root, 'data', f'predictions_2024_{model_type}')
            mask_dir = os.path.join(project_root, 'data', 'hybrid_masks')
            farms_csv = os.path.join(project_root, 'inputs', 'farms_osm.csv')
            metrics_csv = os.path.join(project_root, 'reports', 'deeplab_baseline_metrics.csv')
            cmd = [python_exe, benchmark_script,
                   '--mode', 'baseline',
                   '--prediction-dir', prediction_dir,
                   '--mask-dir', mask_dir,
                   '--farms-csv', farms_csv,
                   '--output-csv', metrics_csv,
                   '--model-name', 'deeplab']
            if inference_seconds is not None:
                cmd.extend(['--inference-seconds', str(inference_seconds)])
            subprocess.check_call(cmd, cwd=project_root)
            logger.info("Baseline metrics written to %s", metrics_csv)
        except subprocess.CalledProcessError as e:
            logger.error("Baseline metrics failed (exit %d)", e.returncode)
            return
        except Exception as e:
            logger.error("Baseline metrics error: %s", e)
            return
    else:
        logger.info("[6/8] Skipping baseline metrics")

    # --- Step 7: Detect Deforestation ---
    if not skip_inference:
        logger.info("[7/8] Detecting deforestation (2020 vs 2024)")
        try:
            if not os.path.exists(detection_script):
                raise FileNotFoundError(f"Script not found: {detection_script}")
            env = os.environ.copy()
            env['EUDR_PRED_DIR'] = os.path.join(project_root, 'data', f'predictions_2024_{model_type}')
            subprocess.check_call([python_exe, detection_script], cwd=project_root, env=env)
            logger.info("Deforestation detection complete")
        except subprocess.CalledProcessError as e:
            logger.error("Detection failed (exit %d)", e.returncode)
            return
        except Exception as e:
            logger.error("Detection error: %s", e)
            return
    else:
        logger.info("[7/8] Skipping deforestation detection")

    # --- Step 8: Audit trail + optional DDS export ---
    if not skip_inference:
        logger.info("[8/8] Writing audit trail")
        try:
            from src.audit_trail import AuditLog, build_audit_entry

            report_csv = os.path.join(project_root, 'reports', 'deforestation_report.csv')
            summary_json = os.path.join(project_root, 'reports', 'summary_stats.json')
            summary = {}
            if os.path.exists(summary_json):
                with open(summary_json, 'r') as f:
                    summary = json.load(f)

            try:
                git_sha = subprocess.check_output(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    cwd=project_root, stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                git_sha = "unknown"

            entry = build_audit_entry(
                model_type=model_type,
                model_version=git_sha,
                input_image_dir=os.path.join(project_root, 'data', 'raw_satellite', '2024_current'),
                prediction_dir=os.path.join(project_root, 'data', f'predictions_2024_{model_type}'),
                report_csv_path=report_csv,
                summary=summary,
            )
            log = AuditLog(log_path=os.path.join(project_root, 'reports', 'audit_log.jsonl'))
            entry = log.append(entry)
            logger.info("Audit entry written (run_id=%s)", entry.run_id)

            if export_dds and operator_name:
                logger.info("Exporting Due Diligence Statements")
                from src.dds_exporter import CommodityInfo, DDSExporter, OperatorInfo
                import pandas as pd

                report_df = pd.read_csv(report_csv)
                op = OperatorInfo(name=operator_name, address=operator_address,
                                  country_iso2=operator_country, eori=operator_eori)
                com = CommodityInfo(
                    hs_code=commodity_hs_code, description=commodity_description,
                    quantity=commodity_quantity, unit=commodity_unit,
                    production_start=f"{entry.assessment_year}-01-01",
                    production_end=f"{entry.assessment_year}-12-31",
                )
                exporter = DDSExporter(op, com, model_version=git_sha)
                records = exporter.from_report(
                    report_df,
                    farms_csv=os.path.join(project_root, 'inputs', 'farms_osm.csv'),
                    evidence_hash=entry.report_hash,
                )
                dds_dir = os.path.join(project_root, 'reports', 'dds')
                exporter.to_json(records, os.path.join(dds_dir, 'dds.json'))
                exporter.to_xml(records, os.path.join(dds_dir, 'dds.xml'))
                try:
                    exporter.to_pdf(records, os.path.join(dds_dir, 'dds.pdf'))
                except ImportError:
                    logger.warning("PDF export skipped — reportlab not installed")

        except Exception as e:
            logger.warning("Audit/DDS step failed (non-fatal): %s", e)

    logger.info("Pipeline finished | model=%s", model_path)
    if not skip_inference:
        logger.info("Outputs: predictions=data/predictions_2024_%s  report=reports/deforestation_report.csv  audit=reports/audit_log.jsonl", model_type)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the EUDR pipeline with selectable model backend.')
    parser.add_argument('--run-farm-discovery', action='store_true')
    parser.add_argument('--run-image-download', action='store_true')
    parser.add_argument('--skip-inference', action='store_true')
    parser.add_argument('--skip-baseline-metrics', action='store_true')
    parser.add_argument('--model-type', choices=['deeplab', 'tessera', 'tessera-embed'], default='deeplab')
    parser.add_argument('--model-path', default=None)

    dds = parser.add_argument_group('DDS export (requires --export-dds)')
    dds.add_argument('--export-dds', action='store_true')
    dds.add_argument('--operator-name', default='')
    dds.add_argument('--operator-address', default='')
    dds.add_argument('--operator-country', default='')
    dds.add_argument('--operator-eori', default='')
    dds.add_argument('--commodity-hs-code', default='')
    dds.add_argument('--commodity-description', default='')
    dds.add_argument('--commodity-quantity', type=float, default=0.0)
    dds.add_argument('--commodity-unit', default='kg')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        skip_farm_discovery=not args.run_farm_discovery,
        skip_image_download=not args.run_image_download,
        skip_inference=args.skip_inference,
        skip_baseline_metrics=args.skip_baseline_metrics,
        model_type=args.model_type,
        model_path=args.model_path,
        export_dds=args.export_dds,
        operator_name=args.operator_name,
        operator_address=args.operator_address,
        operator_country=args.operator_country,
        operator_eori=args.operator_eori,
        commodity_hs_code=args.commodity_hs_code,
        commodity_description=args.commodity_description,
        commodity_quantity=args.commodity_quantity,
        commodity_unit=args.commodity_unit,
    )


if __name__ == "__main__":
    main()
