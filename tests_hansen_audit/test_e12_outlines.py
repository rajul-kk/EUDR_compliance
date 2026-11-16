import importlib.util
import os

import numpy as np
from shapely.geometry import box

from hansen_audit import outlines as O
from hansen_audit.s2 import HANSEN_RES


def _script():
    d = os.path.join(os.path.dirname(__file__), "..", "scripts", "hansen_audit")
    spec = importlib.util.spec_from_file_location("e12_outlines", os.path.join(d, "e12_outlines.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_outline_cells_on_matches_window_version_on_the_hansen_grid():
    origin, shape = np.array([100.0, 1.0]), (8, 10)
    poly = box(100.0005, 0.9990, 100.0021, 0.9999)
    tr = (HANSEN_RES, 0.0, origin[0], 0.0, -HANSEN_RES, origin[1])
    assert (O.outline_cells(origin, shape, poly) == O.outline_cells_on(tr, shape, poly)).all()


def test_outline_cells_on_uses_the_raster_cell_size():
    tr = (0.5, 0.0, 0.0, 0.0, -0.5, 2.0)          # 4 x 4 cells of 0.5 over [0, 2] x [0, 2]
    m = O.outline_cells_on(tr, (4, 4), box(0.0, 1.0, 1.0, 2.0))   # top-left quarter
    assert m.sum() == 4 and m[:2, :2].all()


def test_overpass_query_splits_ways_and_relations():
    q = _script().overpass_query(["way_1", "relation_2", "way_3"])
    assert "way(id:1,3)" in q and "relation(id:2)" in q and "out geom meta" in q


def test_flag_compare_counts_box_flags_lost_inside_outlines():
    m = _script()
    rows = {"a": {"box": {"class2": 10}, "outline": {"class2": 2}},
            "b": {"box": {"class2": 7}, "outline": {"class2": 7}},
            "c": {"box": {"class2": 1}, "outline": {"class2": 1}}}
    r = m.flag_compare(rows, ["a", "b", "c"])
    assert (r["box_flagged"], r["outline_flagged"], r["flagged_outline_not_box"]) == (2, 1, 0)
    assert r["share_box_flags_not_flagged_in_outline"]["est"] == 0.5
    assert m.flag_compare(rows, []) is None


def test_reject_partial_moves_a_reply_with_a_remark_aside(tmp_path):
    import json

    import pytest

    m = _script()
    p = tmp_path / "overpass.json"
    p.write_text(json.dumps({"elements": [], "remark": "runtime error: Query timed out"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="partial"):
        m.reject_partial(str(p))
    assert not p.exists() and (tmp_path / "overpass.json.rejected").exists()
    p.write_text(json.dumps({"elements": [{"type": "way", "id": 1}]}), encoding="utf-8")
    m.reject_partial(str(p))
    assert p.exists()
