import numpy as np

from hansen_audit import farms_e4 as E
from hansen_audit import outlines as O


def _pts(coords):
    return [{"lon": x, "lat": y} for x, y in coords]


def test_closed_way_and_open_way():
    sq = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01), (0, 0)]
    p = O.element_polygon({"type": "way", "id": 1, "geometry": _pts(sq)})
    assert abs(p.area - 1e-4) < 1e-12
    assert O.element_polygon({"type": "way", "id": 2, "geometry": _pts(sq[:-1])}) is None


def test_relation_with_split_outer_and_inner_hole():
    outer_a = [(0, 0), (0.02, 0), (0.02, 0.02)]
    outer_b = [(0.02, 0.02), (0, 0.02), (0, 0)]
    inner = [(0.005, 0.005), (0.015, 0.005), (0.015, 0.015), (0.005, 0.015), (0.005, 0.005)]
    rel = {"type": "relation", "id": 9, "members": [
        {"type": "way", "role": "outer", "geometry": _pts(outer_a)},
        {"type": "way", "role": "outer", "geometry": _pts(outer_b)},
        {"type": "way", "role": "inner", "geometry": _pts(inner)}]}
    p = O.element_polygon(rel)
    assert abs(p.area - (4e-4 - 1e-4)) < 1e-12
    assert O.farm_key(rel) == "relation_9"


def test_outline_cells_match_box_cells_for_a_box_outline_and_drop_holes():
    from shapely.geometry import box

    origin, shape = np.array([10.0, -1.0]), (60, 60)
    b = (10.0, -1.01, 10.01, -1.0)
    assert (O.outline_cells(origin, shape, box(*b)) == E.farm_cells(origin, shape, b)).all()
    holed = box(*b).difference(box(10.0025, -1.0075, 10.0075, -1.0025))
    m = O.outline_cells(origin, shape, holed)
    assert m.sum() == 40 * 40 - 20 * 20


def test_coverage():
    from shapely.geometry import box

    assert abs(O.coverage(box(0, 0, 0.02, 0.01), (0, 0, 0.01, 0.01)) - 0.5) < 1e-3
    assert O.coverage(box(0, 0, 0.01, 0.01), (0, 0, 0.01, 0.01)) > 0.999
