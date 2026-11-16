import numpy as np

from hansen_audit import farms_e4 as E


def test_group_windows_respects_span_and_covers_all():
    boxes = np.array([[0, 0, 0.01, 0.01], [0.02, 0, 0.03, 0.01], [0.2, 0.2, 0.21, 0.21], [0.045, 0.0, 0.055, 0.01]])
    groups = E.group_windows(boxes, 0.05)
    assert sorted(j for g in groups for j in g) == [0, 1, 2, 3]
    for g in groups:
        b = boxes[g]
        assert b[:, 2].max() - b[:, 0].min() <= 0.05 + 1e-12
    assert groups[0] == [0, 1]


def test_window_grid_and_farm_cells():
    boxes = np.array([[10.0, -1.01, 10.01, -1.0], [10.005, -1.015, 10.015, -1.005]])
    origin, shape = E.window_grid(boxes)
    assert tuple(origin) == (10.0, -1.0) and shape == (60, 60)
    m0 = E.farm_cells(origin, shape, boxes[0])
    assert m0.sum() == 40 * 40 and m0[0, 0] and not m0[-1, -1]
    m1 = E.farm_cells(origin, shape, boxes[1])
    assert m1.sum() == 40 * 40 and (m0 & m1).sum() == 20 * 20


def test_farm_summary_rule():
    c = np.array([[0, 10, 0], [7, 1, 0], [5, 5, 5], [1, 1, 1]])  # flagged: 0, 1, 2; plantation-driven: 0, 2
    s = E.farm_summary(c, n_boot=200)
    assert (s["n_flagged"], s["n_plantation_driven"]) == (3, 2)
    assert abs(s["share_plantation_driven"]["est"] - 2 / 3) < 1e-12
    assert s["flagged_farms_loss_cells_by_2020_class"] == {"FOREST": 12, "PLANT": 16, "OTHER": 5}


def test_iter_windows_and_read_boxes(tmp_path):
    import json

    (tmp_path / "windows.json").write_text(json.dumps({
        "farms": ["a", "b"], "boxes": [[0, 0, 1, 1], [1, 1, 2, 2]],
        "windows": [{"id": "00", "origin": [0, 2], "shape": [3, 4], "farms": ["a"]},
                    {"id": "01", "origin": [1, 2], "shape": [3, 4], "farms": ["b"]}]}), encoding="utf-8")
    np.savez(tmp_path / "p.npz", win_00=np.zeros((3, 4), int))
    assert E.read_boxes(tmp_path / "windows.json")["b"].tolist() == [1, 1, 2, 2]
    got = list(E.iter_windows(tmp_path / "windows.json", tmp_path / "p.npz"))
    assert [g[0] for g in got] == ["win_00", "win_01"]
    assert got[0][3] == (3, 4) and got[0][4].shape == (3, 4) and got[1][4] is None
