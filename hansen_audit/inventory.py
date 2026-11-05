"""Compute (never assume) the data inventory. Read-only on the data folder."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

from .io import index_images, index_labels, load_farms


def _meta(path: Path) -> dict:
    with rasterio.open(path) as s:
        return dict(count=s.count, dtypes=list(s.dtypes), w=s.width, h=s.height,
                    crs=str(s.crs), res=[round(r, 12) for r in s.res], nodata=s.nodata,
                    bounds=list(s.bounds))


def inventory(root: Path, band_sample: int = 40, seed: int = 0) -> dict:
    root = Path(root)
    farms = load_farms(root)
    b = index_images(root / "2020_baseline")
    c = index_images(root / "2024_current")
    lab = index_labels(root / "hansen_labels")
    keys = set(farms["key"])
    inv: dict = {}
    inv["n_farms_csv"] = len(farms)
    inv["n_farms_dup_key"] = int(farms["key"].duplicated().sum())
    inv["country"] = farms["country_iso2"].value_counts().to_dict()
    inv["crop_type"] = farms["crop_type"].value_counts().to_dict()
    inv["country_x_crop"] = {f"{k[0]}|{k[1]}": int(v) for k, v in
                             farms.groupby(["country_iso2", "crop_type"]).size().items()}
    inv["n_farms_with_baseline"] = len(b)
    inv["n_farms_with_current"] = len(c)
    inv["n_farms_with_label"] = len(lab)
    inv["n_files"] = dict(baseline=sum(len(v) for v in b.values()),
                          current=sum(len(v) for v in c.values()), labels=len(lab))
    complete = sorted(set(b) & set(c) & set(lab) & keys)
    inv["n_farms_complete_triplet"] = len(complete)
    inv["files_not_in_csv"] = dict(baseline=len(set(b) - keys), current=len(set(c) - keys),
                                   labels=len(set(lab) - keys))
    inv["dates_per_farm_baseline"] = dict(Counter(len(v) for v in b.values()))
    inv["dates_per_farm_current"] = dict(Counter(len(v) for v in c.values()))
    inv["baseline_date_range"] = [min(d for v in b.values() for d, _ in v), max(d for v in b.values() for d, _ in v)]
    inv["current_date_range"] = [min(d for v in c.values() for d, _ in v), max(d for v in c.values() for d, _ in v)]
    inv["baseline_months"] = dict(Counter(d[:7] for v in b.values() for d, _ in v))
    inv["current_months"] = dict(Counter(d[:7] for v in c.values() for d, _ in v))

    # raster metadata: all distinct signatures
    sig_img, sig_lab = Counter(), Counter()
    for fid in complete:
        m = _meta(b[fid][0][1]); sig_img[(m["count"], tuple(m["dtypes"]), m["w"], m["h"], m["crs"], tuple(m["res"]), m["nodata"])] += 1
        m = _meta(lab[fid]); sig_lab[(m["count"], tuple(m["dtypes"]), m["w"], m["h"], m["crs"], tuple(m["res"]), m["nodata"])] += 1
    inv["image_signatures"] = [dict(sig=str(k), n=v) for k, v in sig_img.items()]
    inv["label_signatures"] = [dict(sig=str(k), n=v) for k, v in sig_lab.items()]

    # label values + class balance over all complete labels; footprint ratio
    tot = Counter(); per_farm = {}
    for fid in complete:
        with rasterio.open(lab[fid]) as s:
            a = s.read(1)
        u, n = np.unique(a, return_counts=True)
        for ui, ni in zip(u, n):
            tot[int(ui)] += int(ni)
        per_farm[fid] = {int(ui): int(ni) for ui, ni in zip(u, n)}
    inv["label_pixel_counts"] = dict(tot)
    s_all = sum(tot.values())
    inv["label_pixel_fraction"] = {k: v / s_all for k, v in tot.items()}
    inv["farms_with_any_loss_px"] = sum(1 for d in per_farm.values() if d.get(2, 0) > 0)
    inv["farms_with_any_forest_px"] = sum(1 for d in per_farm.values() if d.get(1, 0) + d.get(2, 0) > 0)
    inv["farms_no_forest_at_all"] = sum(1 for d in per_farm.values() if d.get(1, 0) + d.get(2, 0) == 0)

    # band statistics on a deterministic sample
    rng = np.random.default_rng(seed)
    pick = list(rng.choice(complete, size=min(band_sample, len(complete)), replace=False))
    stats = {i: [] for i in range(5)}
    nan_frac = []
    scl = Counter()
    for fid in pick:
        for y in (b, c):
            with rasterio.open(y[fid][0][1]) as s:
                a = s.read()
            nan_frac.append(float(np.isnan(a).mean()))
            for i in range(5):
                stats[i].append((float(np.nanmin(a[i])), float(np.nanmax(a[i])), float(np.nanmean(a[i]))))
            u, n = np.unique(a[4], return_counts=True)
            for ui, ni in zip(u, n):
                scl[int(ui)] += int(ni)
    inv["band_sample_n_farms"] = len(pick)
    inv["band_stats_min_max_mean_range"] = {
        i: dict(min=min(x[0] for x in v), max=max(x[1] for x in v), mean=float(np.mean([x[2] for x in v])))
        for i, v in stats.items()}
    inv["image_nan_fraction_max"] = max(nan_frac)
    inv["band5_value_counts_sample"] = dict(scl)

    # geographic extent (centroid from csv) vs label centre
    inv["farm_lat_range"] = [float(farms.lat.min()), float(farms.lat.max())]
    inv["farm_lon_range"] = [float(farms.lon.min()), float(farms.lon.max())]
    inv["_complete"] = complete
    inv["_per_farm_labels"] = {k: v for k, v in per_farm.items()}
    return inv


def main(root: str, out: str) -> None:
    inv = inventory(Path(root))
    Path(out).write_text(json.dumps(inv, indent=1, default=str))
    print(json.dumps({k: v for k, v in inv.items() if not k.startswith("_")}, indent=1, default=str))


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
