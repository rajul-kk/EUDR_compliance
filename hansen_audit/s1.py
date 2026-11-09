"""Sentinel-1 RTC annual statistics on the Hansen grid of a reference grid (PROTOCOL_S2C data rules).

Source: Planetary Computer STAC `sentinel-1-rtc`, anonymous SAS token. Each scene's first overview (20 m) is warped straight onto the
Hansen 0.00025 deg grid with average resampling; per-cell statistics over the chosen scenes are taken in dB.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np

from . import s2

COLLECTION = "sentinel-1-rtc"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
N_SCENES = 6
MIN_OBS = 3
OVERVIEW_LEVEL = 0  # first overview, 20 m
CHANNELS = ("VV_med", "VH_med", "RATIO_med", "VV_tstd", "VH_tstd", "NOBS")


class Token(s2.Token):
    def get(self) -> str:
        if self._tok is None or time.time() > self._exp - 300:
            self._tok = json.load(urllib.request.urlopen(TOKEN_URL, timeout=60))["token"]
            self._exp = time.time() + 45 * 60
        return self._tok


def pick_scenes(records):
    """records: list of (datetime_str, orbit_state). Returns indices of the chosen scenes (protocol rule), in date order."""
    if not records:
        return []
    by = {}
    for i, (_, orb) in enumerate(records):
        by.setdefault(orb, []).append(i)
    n_desc, n_asc = len(by.get("descending", [])), len(by.get("ascending", []))
    orb = "descending" if n_desc >= n_asc else "ascending"
    idx = sorted(by.get(orb, []), key=lambda i: records[i][0])
    if len(idx) <= N_SCENES:
        return idx
    pos = np.round(np.linspace(0, len(idx) - 1, N_SCENES)).astype(int)
    return [idx[p] for p in pos]


def stats_to_cells(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """vv, vh (S, H, W) linear power, 0 = no data -> (len(CHANNELS), H, W); S1 features NaN where < MIN_OBS valid observations."""
    ok = (vv > 0) & (vh > 0) & np.isfinite(vv) & np.isfinite(vh)
    with np.errstate(all="ignore"):
        vv_db = np.where(ok, 10 * np.log10(np.where(ok, vv, 1.0)), np.nan)
        vh_db = np.where(ok, 10 * np.log10(np.where(ok, vh, 1.0)), np.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = [np.nanmedian(vv_db, 0), np.nanmedian(vh_db, 0), np.nanmedian(vh_db - vv_db, 0),
               np.nanstd(vv_db, 0), np.nanstd(vh_db, 0)]
    nobs = ok.sum(0)
    out = np.stack(out).astype(np.float32)
    out[:, nobs < MIN_OBS] = np.nan
    return np.concatenate([out, nobs[None].astype(np.float32)])


def hansen_transform(origin):
    from rasterio.transform import from_origin

    return from_origin(float(origin[0]), float(origin[1]), s2.HANSEN_RES, s2.HANSEN_RES)


def warp_overview(href: str, origin, shape, retries=3) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    last = None
    for k in range(retries):
        try:
            with rasterio.open(href, overview_level=OVERVIEW_LEVEL) as src, WarpedVRT(
                src, crs="EPSG:4326", transform=hansen_transform(origin), width=shape[1], height=shape[0],
                resampling=Resampling.average, nodata=0,
            ) as vrt:
                return vrt.read(1).astype(np.float32)
        except Exception as e:  # transient HTTP errors
            last = e
            time.sleep(2 * (k + 1))
    raise last


def build_s1(origin, shape, start: str, end: str, token: Token, log=print):
    from pystac_client import Client

    bounds = s2.grid_bounds(origin, shape)
    items = list(Client.open(s2.STAC_URL).search(collections=[COLLECTION], bbox=list(bounds), datetime=f"{start}/{end}").items())
    recs = [(it.properties["datetime"], it.properties.get("sat:orbit_state")) for it in items]
    chosen = [items[i] for i in pick_scenes(recs)]
    log(f"  {start[:4]}: {len(items)} items, using {len(chosen)} {chosen[0].properties.get('sat:orbit_state') if chosen else ''}")
    if not chosen:
        return None, {"n_items": len(items), "scenes": []}
    vv = np.stack([warp_overview(it.assets["vv"].href + "?" + token.get(), origin, shape) for it in chosen])
    vh = np.stack([warp_overview(it.assets["vh"].href + "?" + token.get(), origin, shape) for it in chosen])
    cells = stats_to_cells(vv, vh)
    meta = {"n_items": len(items), "scenes": [{"id": it.id, "datetime": it.properties["datetime"],
                                                "orbit": it.properties.get("sat:orbit_state")} for it in chosen]}
    return cells, meta
