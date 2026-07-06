"""Backwards-compatibility shim — real implementation lives in src/preprocessing/aligner.py."""
import os, sys as _sys
_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _src not in _sys.path:
    _sys.path.insert(0, _src)
from preprocessing.aligner import align_mask_to_image, batch_align_masks, check_alignment  # noqa: F401
