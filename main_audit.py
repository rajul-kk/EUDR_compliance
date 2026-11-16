"""
main_audit.py — project-root entry point for the EUDR compliance pipeline.

Usage examples:
  # Full run with default DeepLab model
  python main_audit.py

  # Train + infer with TESSERA backbone
  python main_audit.py --model-type tessera --run-image-download

  # Train only, no inference
  python main_audit.py --model-type tessera-embed --skip-inference

Run `python main_audit.py --help` for all options.
"""

import sys

from src.logging_config import setup_pipeline_logging
from src.pipeline_runner import main

if __name__ == "__main__":
    setup_pipeline_logging()
    sys.exit(main())
