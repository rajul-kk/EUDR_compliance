"""Static validation for Kaggle training notebooks.

Checks:
1. Valid JSON / parseable notebook format
2. No stale path references to replaced pipeline components
3. Required dataset variables are defined
"""
import sys
import re
from pathlib import Path

import nbformat

STALE_PATTERNS = [
    (r"hybrid_masks", "references removed hybrid_masks directory"),
    (r"generate_hybrid", "calls deprecated generate_hybrid.py"),
    (r"raw_satellite", "uses raw_satellite path (should be 2020_baseline / 2024_current)"),
    (r"2020_hybrid|2024_hybrid", "references legacy per-year hybrid mask naming"),
    (r"CDSE|cdse", "references CDSE credentials (not needed for training notebooks)"),
]

REQUIRED_VARS = ["T1_DIR", "LABEL_DIR"]

SIAMESE_REQUIRED_VARS = ["T1_DIR", "T2_DIR", "LABEL_DIR"]


def validate_notebook(path: Path) -> list[str]:
    errors = []

    try:
        nb = nbformat.read(str(path), as_version=4)
    except Exception as e:
        return [f"JSON parse error: {e}"]

    all_source = "\n".join(
        cell["source"] for cell in nb.cells if cell["cell_type"] in ("code", "markdown")
    )

    for pattern, message in STALE_PATTERNS:
        if re.search(pattern, all_source):
            errors.append(f"Stale reference — {message}")

    required = SIAMESE_REQUIRED_VARS if "siamese" in path.name else REQUIRED_VARS
    for var in required:
        if var not in all_source:
            errors.append(f"Missing required variable: {var}")

    return errors


def main(notebook_dir: str) -> int:
    nb_dir = Path(notebook_dir)
    notebooks = sorted(nb_dir.glob("*.ipynb"))

    if not notebooks:
        print(f"No notebooks found in {nb_dir}")
        return 1

    failed = 0
    for nb_path in notebooks:
        errs = validate_notebook(nb_path)
        if errs:
            print(f"FAIL {nb_path.name}")
            for e in errs:
                print(f"  - {e}")
            failed += 1
        else:
            print(f"OK   {nb_path.name}")

    if failed:
        print(f"\n{failed}/{len(notebooks)} notebook(s) failed validation.")
        return 1

    print(f"\nAll {len(notebooks)} notebooks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "notebooks/"))
