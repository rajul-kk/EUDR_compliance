"""Every alert level is counted in the summary and maps to a defined DDS conclusion; every model path decides the same
loss the same way, under the area rule (default) and the legacy percentage rule."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dds_exporter import CommodityInfo, DDSExporter, OperatorInfo
from detect_deforestation import compute_deforestation_metrics, generate_summary_stats

UTM = SimpleNamespace(is_geographic=False)
WGS84 = SimpleNamespace(is_geographic=True)


def _transform(a, e, f=0.0):
    return SimpleNamespace(a=a, e=e, f=f)


def _exporter():
    return DDSExporter(
        OperatorInfo(name="Op", address="x", country_iso2="GH"),
        CommodityInfo(hs_code="1801", description="cocoa", quantity=1.0, unit="kg",
                      production_start="2024-01-01", production_end="2024-12-31"),
    )


def _report():
    return pd.DataFrame({
        "farm_id": ["a", "b", "c", "d", "e", "f", "g"],
        "alert_level": ["FLAGGED", "CLEAR", "NO_BASELINE_FOREST", "UNCERTAIN", "VIOLATION", "COMPLIANT", "WARNING"],
        "deforestation_percent": [20.0, 0.0, 0.0, 7.0, 20.0, 0.0, 7.0],
        "forest_loss_ha": [3.0, 0.1, 0.0, 0.5, 3.0, 0.0, 1.0],
    })


def _masks():
    # a farm that is 20% forest (20 pixels) and loses half of it (10 pixels)
    base = np.zeros((10, 10), dtype=np.uint8)
    base[:2, :] = 1
    now = base.copy()
    now[0, :] = 0
    return base, now, ((base == 1) & (now != 1)).astype(np.uint8)


class _Src:
    def __init__(self, masks, p, transform=None, crs=None):
        self.masks, self.p, self.transform, self.crs = masks, p, transform, crs

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, band):
        return self.masks[self.p]


def _patch(monkeypatch, dd, masks, transform=None, crs=None):
    monkeypatch.setattr(dd, "load_mask", lambda p: (masks[p], transform, crs))
    monkeypatch.setattr(dd.rasterio, "open", lambda p: _Src(masks, p, transform, crs))


def test_an_empty_baseline_is_its_own_level():
    m = compute_deforestation_metrics(np.zeros((4, 4)), np.zeros((4, 4)), pixel_m2=100.0)
    assert m["alert_level"] == "NO_BASELINE_FOREST"


def test_area_rule_flags_more_than_half_a_hectare():
    base, now, _ = _masks()
    # 10 lost pixels: 10 x 500 m2 = 0.5 ha is not more than 0.5 ha; 10 x 501 m2 is
    assert compute_deforestation_metrics(base, now, pixel_m2=500.0)["alert_level"] == "CLEAR"
    m = compute_deforestation_metrics(base, now, pixel_m2=501.0)
    assert m["alert_level"] == "FLAGGED" and m["forest_loss_ha"] == pytest.approx(0.501)
    assert m["decision_rule"] == "area_0.5ha" and m["deforestation_percent"] == 50.0


def test_area_rule_ignores_the_percentage():
    # 1 % of a large forest can be well over 0.5 ha; 50 % of a tiny one can be under it
    big = np.ones((100, 100), np.uint8)
    lost = big.copy()
    lost[0, :] = 0
    assert compute_deforestation_metrics(big, lost, pixel_m2=100.0)["alert_level"] == "FLAGGED"
    base, now, _ = _masks()
    assert compute_deforestation_metrics(base, now, pixel_m2=100.0)["alert_level"] == "CLEAR"


def test_without_pixel_size_the_area_rule_does_not_guess():
    base, now, _ = _masks()
    m = compute_deforestation_metrics(base, now)
    assert m["alert_level"] == "UNASSESSED" and "pixel size" in m["reason"]


def test_pixel_area():
    from change_detector import pixel_area_m2

    assert pixel_area_m2(_transform(10.0, -10.0), UTM, 100) == 100.0
    assert pixel_area_m2(None, UTM, 100) is None
    eq = pixel_area_m2(_transform(0.0001, -0.0001, 0.005), WGS84, 100)
    assert eq == pytest.approx(11.132 * 11.0574, rel=1e-3)
    assert pixel_area_m2(_transform(0.0001, -0.0001, 60.005), WGS84, 100) == pytest.approx(eq / 2, rel=1e-2)


def test_unknown_rule_is_refused(monkeypatch):
    from change_detector import decision_rule

    monkeypatch.setenv("EUDR_DECISION_RULE", "percent")
    with pytest.raises(ValueError):
        decision_rule()


def test_every_path_decides_the_same_loss_the_same_way(monkeypatch):
    import detect_deforestation as dd
    from change_detector import ChangeDetector

    base, now, change = _masks()
    t = _transform(10.0, -10.0)  # 100 m2 pixels: 10 lost = 0.1 ha
    assert dd.compute_deforestation_metrics(base, now, pixel_m2=100.0)["alert_level"] == "CLEAR"
    assert ChangeDetector().analyze(base, now, pixel_m2=100.0).alert_level.value == "CLEAR"
    _patch(monkeypatch, dd, {"change.tif": change, "base.tif": base}, t, UTM)
    m = dd.analyze_farm_change("change.tif", "base.tif", "f")
    assert m["alert_level"] == "CLEAR" and m["forest_loss_ha"] == pytest.approx(0.1)

    t = _transform(30.0, -30.0)  # 900 m2 pixels: 10 lost = 0.9 ha
    assert ChangeDetector().analyze(base, now, pixel_m2=900.0).alert_level.value == "FLAGGED"
    _patch(monkeypatch, dd, {"change.tif": change, "base.tif": base}, t, UTM)
    assert dd.analyze_farm_change("change.tif", "base.tif", "f")["alert_level"] == "FLAGGED"


def test_legacy_rule_is_kept_behind_its_name(monkeypatch):
    import detect_deforestation as dd
    from change_detector import ChangeDetector, classify_forest_loss

    monkeypatch.setenv("EUDR_DECISION_RULE", "legacy_percent")
    base, now, change = _masks()
    assert classify_forest_loss(20, 10)[1].value == "VIOLATION"
    assert dd.compute_deforestation_metrics(base, now)["alert_level"] == "VIOLATION"
    assert ChangeDetector().analyze(base, now).alert_level.value == "VIOLATION"
    _patch(monkeypatch, dd, {"change.tif": change, "base.tif": base})
    m = dd.analyze_farm_change("change.tif", "base.tif", "f")
    assert m["alert_level"] == "VIOLATION" and m["deforestation_percent"] == 50.0
    assert m["decision_rule"] == "legacy_percent"


def test_no_baseline_forest_is_the_same_on_the_change_path(monkeypatch):
    import detect_deforestation as dd

    _patch(monkeypatch, dd, {"c": np.ones((4, 4), np.uint8), "b": np.zeros((4, 4), np.uint8)},
           _transform(10.0, -10.0), UTM)
    assert dd.analyze_farm_change("c", "b", "f")["alert_level"] == "NO_BASELINE_FOREST"


def test_summary_counts_add_up_to_the_farms(tmp_path):
    s = generate_summary_stats(_report(), str(tmp_path / "s.json"))
    assert (s["flagged"], s["clear"], s["uncertain"], s["no_baseline_forest"]) == (1, 1, 1, 1)
    parts = ("flagged", "clear", "uncertain", "violations", "warnings", "compliant", "no_baseline_forest",
             "unassessed", "other")
    assert sum(s[k] for k in parts) == s["total_farms"]
    assert s["flagged_or_violation"] == 2
    assert json.loads((tmp_path / "s.json").read_text()) == s


def test_dds_never_carries_an_undefined_conclusion(tmp_path):
    report = pd.concat([_report(), pd.DataFrame({"farm_id": ["z"], "alert_level": ["garbled"],
                                                 "deforestation_percent": [0.0]})])
    records = _exporter().from_report(report, farms_csv=str(tmp_path / "missing.csv"))
    by_farm = {r.farm_id: r for r in records}
    assert {r.risk_level for r in records} <= {"FLAGGED", "CLEAR", "UNCERTAIN", "COMPLIANT", "WARNING", "VIOLATION",
                                               "INCONCLUSIVE"}
    assert "3.00 ha" in by_farm["a"].notes and "manual review" in by_farm["a"].notes.lower()
    assert "spans the 0.5 ha threshold" in by_farm["d"].notes
    assert by_farm["c"].risk_level == "INCONCLUSIVE" and "manual review" in by_farm["c"].notes
    assert by_farm["z"].risk_level == "INCONCLUSIVE" and "garbled" in by_farm["z"].notes


def test_farms_that_cannot_be_assessed_stay_in_the_report(tmp_path, monkeypatch):
    import detect_deforestation as dd

    pred, base = tmp_path / "pred", tmp_path / "base"
    pred.mkdir()
    base.mkdir()
    for name in ("way_1_2024_2024-06-01_predicted.tif", "way_2_2024_2024-06-01_predicted.tif",
                 "way_3_2024_2024-06-01_predicted.tif"):
        (pred / name).write_bytes(b"")
    for name in ("way_1_2020_hybrid.tif", "way_3_2020_hybrid.tif", "way_4_2020_hybrid.tif"):
        (base / name).write_bytes(b"")
    ok = {"farm_id": "way_1", "alert_level": "CLEAR", "deforestation_percent": 1.0, "forest_loss_ha": 0.1}
    monkeypatch.setattr(dd, "analyze_farm_pair", lambda b, p, fid, v: ok if fid == "way_1" else None)

    df = dd.batch_detect_deforestation(str(base), str(pred), str(tmp_path / "r" / "report.csv"))
    rows = {r.farm_id: r for r in df.itertuples()}
    assert set(rows) == {"way_1", "way_2", "way_3", "way_4"}
    assert rows["way_2"].reason == "no 2020 baseline mask"
    assert rows["way_3"].reason.startswith("analysis failed")
    assert rows["way_4"].reason == "no 2024 prediction"
    s = dd.generate_summary_stats(df, str(tmp_path / "s.json"))
    assert s["unassessed"] == 3 and s["clear"] == 1 and s["other"] == 0


def test_dds_for_an_unassessed_farm_says_why(tmp_path):
    report = pd.DataFrame({"farm_id": ["a"], "alert_level": ["UNASSESSED"], "deforestation_percent": [float("nan")],
                           "reason": ["no 2024 prediction"]})
    (r,) = _exporter().from_report(report, farms_csv=str(tmp_path / "missing.csv"))
    assert r.risk_level == "INCONCLUSIVE" and "no 2024 prediction" in r.notes and r.deforestation_pct == 0.0
