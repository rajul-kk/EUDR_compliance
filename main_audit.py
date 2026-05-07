"""
main_audit.py — project-root entry point for the EUDR compliance pipeline.

Usage examples:
  # Full run with default DeepLab model
  python main_audit.py

  # Train + infer with TESSERA backbone
  python main_audit.py --model-type tessera --run-image-download

  # Skip training, re-run inference only
  python main_audit.py --model-type tessera-embed --skip-training

Run `python main_audit.py --help` for all options.
"""

import sys
import os

# Ensure project root is first on sys.path so all src.* imports resolve.
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.pipeline_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
