import importlib.util
import pathlib

import numpy as np

spec = importlib.util.spec_from_file_location(
    "rebuild_labels", pathlib.Path(__file__).parents[1] / "scripts" / "hansen_audit" / "rebuild_labels.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def test_no_loss_forest_is_class_1_once_lossyear_is_unmasked_to_0():
    tc = np.array([50, 50, 50, 50, 5, 50])
    ly = np.array([0, 10, 22, 25, 0, 0])   # no loss, 2010, 2022, 2025, non-forest, no data
    dm = np.array([1, 1, 1, 1, 1, 0])
    assert rl.label_from_bands(tc, ly, dm).tolist() == [1, 0, 2, 1, 0, 255]
