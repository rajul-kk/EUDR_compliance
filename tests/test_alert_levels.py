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
