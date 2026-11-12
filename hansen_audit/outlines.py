"""PROTOCOL_E5: OpenStreetMap plot outlines from Overpass `out geom` JSON, and the cells inside them."""
from __future__ import annotations

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, polygonize, transform, unary_union

from .s2 import HANSEN_RES


def farm_key(element: dict) -> str:
    return f"{element['type']}_{element['id']}"


def _rings(ways):
    lines = [LineString([(p["lon"], p["lat"]) for p in w["geometry"]]) for w in ways if len(w.get("geometry") or []) >= 2]
    if not lines:
        return []
    return list(polygonize(linemerge(lines)))


def element_polygon(element: dict):
    """Closed way -> Polygon; multipolygon relation -> outer rings minus inner rings. None if no closed area results."""
    if element["type"] == "way":
        g = element.get("geometry") or []
        if len(g) < 4 or (g[0]["lat"], g[0]["lon"]) != (g[-1]["lat"], g[-1]["lon"]):
            return None
        poly = Polygon([(p["lon"], p["lat"]) for p in g])
    elif element["type"] == "relation":
        members = [m for m in element.get("members", []) if m.get("type") == "way"]
        outer = _rings([m for m in members if m.get("role") in ("outer", "")])
        inner = _rings([m for m in members if m.get("role") == "inner"])
        if not outer:
            return None
        poly = unary_union(outer)
        if inner:
            poly = poly.difference(unary_union(inner))
    else:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not isinstance(poly, (Polygon, MultiPolygon)):
        return None
    return poly


def equal_area(geom):
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    return transform(t.transform, geom)


def coverage(poly, farm_box) -> float:
    """Share of the outline's area inside the farm box (equal-area)."""
    a = equal_area(poly).area
    return float(equal_area(poly.intersection(box(*farm_box))).area / a) if a > 0 else 0.0


def outline_cells(origin, shape, poly) -> np.ndarray:
    """Bool (H, W): window cells whose centres fall inside the outline (holes excluded)."""
    H, W = shape
    lon = origin[0] + (np.arange(W) + 0.5) * HANSEN_RES
    lat = origin[1] - (np.arange(H) + 0.5) * HANSEN_RES
    X, Y = np.meshgrid(lon, lat)
    return contains_xy(poly, X, Y)
