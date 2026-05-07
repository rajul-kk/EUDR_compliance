
import subprocess
import sys
import os
import argparse
import json
import time

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
    """
    Runs the end-to-end pipeline:
    1. Discover farm locations from OSM (find_farms.py) - Optional
    2. Download Sentinel-2 satellite imagery (sentinel_client.py) - Optional
    3. Generate Hybrid Masks using GEE (generate_hybrid.py)
    4. Train DeepLabV3 Model (ML_farm_net.py)
    5. Run Inference on 2024 images (inference.py) - Optional
    6. Export baseline metrics (benchmark.py baseline mode) - Optional
    7. Detect Deforestation (detect_deforestation.py) - Optional
    
    Args:
        skip_farm_discovery: Skip Step 1 if farms CSV already exists
        skip_image_download: Skip Step 2 if satellite images already downloaded
        skip_inference: Skip Steps 5-6 if only training is needed
    """
    print("🚀 Starting End-to-End EUDR Compliance Pipeline")
    print(f"Project Root: {project_root}")
    print(f"Model Type: {model_type}")
    
    # Use the current python executable
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")

    # --- Step 1: Find Farms from OSM ---
    if not skip_farm_discovery:
        print("\n" + "="*60)
        print("[Step 1/6] Discovering Farm Locations from OpenStreetMap...")
        print("="*60)
        try:
            if not os.path.exists(find_farms_script):
                raise FileNotFoundError(f"Script not found: {find_farms_script}")
                
            subprocess.check_call([python_exe, find_farms_script], cwd=project_root)
            print("✅ Farm Discovery Complete (inputs/farms_osm.csv updated).")
        except subprocess.CalledProcessError as e:
            print(f"❌ Farm Discovery Failed with exit code {e.returncode}.")
            print("   Continuing with existing farms_osm.csv if available...")
        except Exception as e:
            print(f"❌ An error occurred during farm discovery: {e}")
            print("   Continuing with existing farms_osm.csv if available...")
    else:
        print("\n[Step 1/6] ⏭️  Skipping Farm Discovery (using existing inputs/farms_osm.csv)")

    # --- Step 2: Download Satellite Images ---
    if not skip_image_download:
        print("\n" + "="*60)
        print("[Step 2/6] Downloading Sentinel-2 Satellite Imagery...")
        print("="*60)
        try:
            if not os.path.exists(sentinel_script):
                raise FileNotFoundError(f"Script not found: {sentinel_script}")
                
            subprocess.check_call([python_exe, sentinel_script], cwd=project_root)
            print("✅ Satellite Image Download Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Image Download Failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during image download: {e}")
            return
    else:
        print("\n[Step 2/6] ⏭️  Skipping Image Download (using existing data/raw_satellite/)")

    # --- Step 3: Generate/Download Masks ---
    print("\n" + "="*60)
    print("[Step 3/6] Generating Hybrid Masks from GEE...")
    print("="*60)
    try:
        # Check if script exists
        if not os.path.exists(generate_script):
            raise FileNotFoundError(f"Script not found: {generate_script}")
            
        subprocess.check_call([python_exe, generate_script], cwd=project_root)
        print("✅ Mask Generation Complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Mask Generation Failed with exit code {e.returncode}.")
        return
    except Exception as e:
        print(f"❌ An error occurred during mask generation: {e}")
        return

    # --- Step 4: Train Model ---
    print("\n" + "="*60)
    print(f"[Step 4/6] Training {model_type.upper()} Model...")
    print("="*60)
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
                [
                    python_exe,
                    training_script,
                    '--raw-dir', raw_dir,
                    '--mask-dir', mask_dir,
                    '--output-model-path', model_path,
                ],
                cwd=project_root,
            )
        else:
            if model_path is None:
                model_path = os.path.join(project_root, 'models', 'farm_tessera_embed_head.pth')

            embeddings_dir = os.path.join(project_root, 'data', 'embeddings', 'global_0.1_degree_representation')
            mask_dir = os.path.join(project_root, 'data', 'geotessera_tile_masks')
            subprocess.check_call(
                [
                    python_exe,
                    training_script,
                    '--embeddings-dir', embeddings_dir,
                    '--mask-dir', mask_dir,
                    '--output-model-path', model_path,
                    '--dataset-mode', 'geotessera',
                    '--year', '2024',
                ],
                cwd=project_root,
            )

            print("✅ Training Complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training Failed with exit code {e.returncode}.")
        return
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        return

    inference_seconds = None

    # --- Step 5: Run Inference on 2024 Images ---
    if not skip_inference:
        print("\n" + "="*60)
        print("[Step 5/6] Running Inference on 2024 Satellite Images...")
        print("="*60)
        try:
            if model_path is None:
                model_path = os.path.join(project_root, 'models', 'farm_deeplab.pth')
            output_dir = os.path.join(project_root, 'data', f'predictions_2024_{model_type}')

            inference_start = time.time()
            if model_type == 'tessera-embed':
                if not os.path.exists(tessera_embed_infer_script):
                    raise FileNotFoundError(f"Script not found: {tessera_embed_infer_script}")

                embeddings_dir_2024 = os.path.join(project_root, 'data', 'embeddings', 'global_0.1_degree_representation')
                reference_image_dir = os.path.join(project_root, 'data', 'embeddings', 'global_0.1_degree_tiff_all')
                subprocess.check_call(
                    [
                        python_exe,
                        tessera_embed_infer_script,
                        '--model-path', model_path,
                        '--embeddings-dir', embeddings_dir_2024,
                        '--output-dir', output_dir,
                        '--year', '2024',
                        '--reference-image-dir', reference_image_dir,
                    ],
                    cwd=project_root,
                )
            else:
                if not os.path.exists(inference_script):
                    raise FileNotFoundError(f"Script not found: {inference_script}")

                input_dir = os.path.join(project_root, 'data', 'raw_satellite', '2024_current')
                subprocess.check_call(
                    [
                        python_exe,
                        inference_script,
                        '--model-path', model_path,
                        '--input-dir', input_dir,
                        '--output-dir', output_dir,
                        '--model-type', model_type,
                    ],
                    cwd=project_root,
                )
            inference_seconds = time.time() - inference_start
            print("✅ Inference Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference Failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during inference: {e}")
            return
    else:
        print("\n[Step 5/6] ⏭️  Skipping Inference")

    # --- Step 6: Export Baseline Metrics ---
    if not skip_inference and not skip_baseline_metrics and model_type == 'deeplab':
        print("\n" + "="*60)
        print("[Step 6/7] Exporting DeepLab Baseline Metrics...")
        print("="*60)
        try:
            if not os.path.exists(benchmark_script):
                raise FileNotFoundError(f"Script not found: {benchmark_script}")

            prediction_dir = os.path.join(project_root, 'data', f'predictions_2024_{model_type}')
            mask_dir = os.path.join(project_root, 'data', 'hybrid_masks')
            farms_csv = os.path.join(project_root, 'inputs', 'farms_osm.csv')
            metrics_csv = os.path.join(project_root, 'reports', 'deeplab_baseline_metrics.csv')

            cmd = [
                python_exe,
                benchmark_script,
                '--mode', 'baseline',
                '--prediction-dir', prediction_dir,
                '--mask-dir', mask_dir,
                '--farms-csv', farms_csv,
                '--output-csv', metrics_csv,
                '--model-name', 'deeplab',
            ]
            if inference_seconds is not None:
                cmd.extend(['--inference-seconds', str(inference_seconds)])

            subprocess.check_call(cmd, cwd=project_root)
            print(f"✅ Baseline metrics exported to {metrics_csv}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Baseline metric export failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during baseline metric export: {e}")
            return
    elif model_type != 'deeplab':
        print("\n[Step 6/7] ⏭️  Skipping baseline metrics (DeepLab-only step)")
    elif skip_baseline_metrics:
        print("\n[Step 6/7] ⏭️  Skipping baseline metrics")
    else:
        print("\n[Step 6/7] ⏭️  Skipping baseline metrics")

    # --- Step 7: Detect Deforestation ---
    if not skip_inference:
        print("\n" + "="*60)
        print("[Step 7/7] Detecting Deforestation (2020 vs 2024)...")
        print("="*60)
        try:
            if not os.path.exists(detection_script):
                raise FileNotFoundError(f"Script not found: {detection_script}")
                
            env = os.environ.copy()
            env['EUDR_PRED_DIR'] = os.path.join(project_root, 'data', f'predictions_2024_{model_type}')
            subprocess.check_call([python_exe, detection_script], cwd=project_root, env=env)
            print("✅ Deforestation Detection Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Detection Failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during detection: {e}")
            return
    else:
        print("\n[Step 7/7] ⏭️  Skipping Deforestation Detection")

    # --- Step 8: Audit trail + optional DDS export ---
    if not skip_inference:
        print("\n" + "="*60)
        print("[Step 8/8] Writing audit trail...")
        print("="*60)
        try:
            from src.audit_trail import AuditLog, build_audit_entry

            report_csv = os.path.join(project_root, 'reports', 'deforestation_report.csv')
            summary_json = os.path.join(project_root, 'reports', 'summary_stats.json')
            summary = {}
            if os.path.exists(summary_json):
                with open(summary_json, 'r') as f:
                    summary = json.load(f)

            try:
                import subprocess as _sp
                git_sha = _sp.check_output(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    cwd=project_root, stderr=_sp.DEVNULL
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
            print(f"Audit entry written (run_id={entry.run_id})")

            if export_dds and operator_name:
                print("Exporting Due Diligence Statements...")
                from src.dds_exporter import CommodityInfo, DDSExporter, OperatorInfo
                import pandas as pd

                report_df = pd.read_csv(report_csv)
                op = OperatorInfo(
                    name=operator_name,
                    address=operator_address,
                    country_iso2=operator_country,
                    eori=operator_eori,
                )
                com = CommodityInfo(
                    hs_code=commodity_hs_code,
                    description=commodity_description,
                    quantity=commodity_quantity,
                    unit=commodity_unit,
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
                    print("PDF export skipped (reportlab not installed).")

        except Exception as e:
            print(f"Audit/DDS step failed (non-fatal): {e}")

    print("\n🎉 Pipeline Finished Successfully!")
    print("📊 Outputs:")
    print(f"   - Trained model: {model_path}")
    if not skip_inference:
        print(f"   - 2024 predictions: data/predictions_2024_{model_type}/")
        if model_type == 'deeplab' and not skip_baseline_metrics:
            print("   - Baseline metrics: reports/deeplab_baseline_metrics.csv")
        print("   - Compliance report: reports/deforestation_report.csv")
        print("   - Summary stats:     reports/summary_stats.json")
        print("   - Audit log:         reports/audit_log.jsonl")
        if export_dds:
            print("   - DDS export:        reports/dds/ (JSON, XML, PDF)")


def parse_args():
    parser = argparse.ArgumentParser(description='Run the EUDR pipeline with selectable model backend.')
    parser.add_argument('--run-farm-discovery', action='store_true', help='Run farm discovery step.')
    parser.add_argument('--run-image-download', action='store_true', help='Run Sentinel image download step.')
    parser.add_argument('--skip-inference', action='store_true', help='Skip inference and deforestation detection steps.')
    parser.add_argument('--skip-baseline-metrics', action='store_true', help='Skip baseline metric export step (DeepLab only).')
    parser.add_argument('--model-type', choices=['deeplab', 'tessera', 'tessera-embed'], default='deeplab')
    parser.add_argument('--model-path', default=None, help='Optional explicit model checkpoint path.')

    dds = parser.add_argument_group('DDS export (requires --export-dds)')
    dds.add_argument('--export-dds', action='store_true',
                     help='Export Due Diligence Statements (JSON, XML, PDF) after detection.')
    dds.add_argument('--operator-name', default='', help='Legal name of the operator submitting the DDS.')
    dds.add_argument('--operator-address', default='', help='Operator registered address.')
    dds.add_argument('--operator-country', default='', help='Operator country (ISO-3166-1 alpha-2).')
    dds.add_argument('--operator-eori', default='', help='EORI number (optional).')
    dds.add_argument('--commodity-hs-code', default='', help='HS code of the commodity (e.g. 1801 for cocoa).')
    dds.add_argument('--commodity-description', default='', help='Commodity description.')
    dds.add_argument('--commodity-quantity', type=float, default=0.0, help='Quantity of commodity.')
    dds.add_argument('--commodity-unit', default='kg', help='Unit of measure (kg, t, m3 …).')
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
