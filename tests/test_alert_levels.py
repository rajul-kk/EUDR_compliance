"""Every alert level is counted in the summary and maps to a defined DDS conclusion."""
import json

import numpy as np
import pandas as pd

from dds_exporter import CommodityInfo, DDSExporter, OperatorInfo
from detect_deforestation import compute_deforestation_metrics, generate_summary_stats


def _report():
    return pd.DataFrame({
        "farm_id": ["a", "b", "c", "d"],
        "alert_level": ["VIOLATION", "COMPLIANT", "NO_BASELINE_FOREST", "WARNING"],
        "deforestation_percent": [20.0, 0.0, 0.0, 7.0],
    })


def test_an_empty_baseline_is_its_own_level():
    m = compute_deforestation_metrics(np.zeros((4, 4)), np.zeros((4, 4)))
    assert m["alert_level"] == "NO_BASELINE_FOREST"


def test_summary_counts_add_up_to_the_farms(tmp_path):
    s = generate_summary_stats(_report(), str(tmp_path / "s.json"))
    assert s["no_baseline_forest"] == 1
    assert s["violations"] + s["warnings"] + s["compliant"] + s["no_baseline_forest"] + s["other"] == s["total_farms"]
    assert json.loads((tmp_path / "s.json").read_text()) == s


def test_dds_never_carries_an_undefined_conclusion(tmp_path):
    exporter = DDSExporter(
        OperatorInfo(name="Op", address="x", country_iso2="GH"),
        CommodityInfo(hs_code="1801", description="cocoa", quantity=1.0, unit="kg",
                      production_start="2024-01-01", production_end="2024-12-31"),
    )
    report = pd.concat([_report(), pd.DataFrame({"farm_id": ["e"], "alert_level": ["garbled"],
                                                 "deforestation_percent": [0.0]})])
    records = exporter.from_report(report, farms_csv=str(tmp_path / "missing.csv"))
    by_farm = {r.farm_id: r for r in records}
    assert {r.risk_level for r in records} <= {"COMPLIANT", "WARNING", "VIOLATION", "INCONCLUSIVE"}
    assert by_farm["c"].risk_level == "INCONCLUSIVE"
    assert "manual review" in by_farm["c"].notes
    assert by_farm["e"].risk_level == "INCONCLUSIVE"
    assert "garbled" in by_farm["e"].notes


def test_every_path_classifies_the_same_loss_the_same_way(tmp_path, monkeypatch):
    import detect_deforestation as dd
    from change_detector import ChangeDetector, classify_forest_loss

    # a farm that is 20% forest and loses half of it: 10% of the image, 50% of its forest
    base = np.zeros((10, 10), dtype=np.uint8)
    base[:2, :] = 1
    now = base.copy()
    now[0, :] = 0
    change = ((base == 1) & (now != 1)).astype(np.uint8)

    assert classify_forest_loss(20, 10)[1].value == "VIOLATION"
    assert dd.compute_deforestation_metrics(base, now)["alert_level"] == "VIOLATION"
    assert ChangeDetector().analyze(base, now).alert_level.value == "VIOLATION"

    masks = {"change.tif": change, "base.tif": base}
    monkeypatch.setattr(dd, "load_mask", lambda p: (masks[p], None, None))

    class Src:
        def __init__(self, p):
            self.p = p
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, band):
            return masks[self.p]
        transform = crs = None

    monkeypatch.setattr(dd.rasterio, "open", lambda p: Src(p))
    m = dd.analyze_farm_change("change.tif", "base.tif", "f")
    assert m["alert_level"] == "VIOLATION" and m["deforestation_percent"] == 50.0


def test_no_baseline_forest_is_the_same_on_the_change_path(monkeypatch):
    import detect_deforestation as dd

    masks = {"c": np.ones((4, 4), np.uint8), "b": np.zeros((4, 4), np.uint8)}
    monkeypatch.setattr(dd, "load_mask", lambda p: (masks[p], None, None))

    class Src:
        def __init__(self, p):
            self.p = p
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, band):
            return masks[self.p]
        transform = crs = None

    monkeypatch.setattr(dd.rasterio, "open", lambda p: Src(p))
    assert dd.analyze_farm_change("c", "b", "f")["alert_level"] == "NO_BASELINE_FOREST"
