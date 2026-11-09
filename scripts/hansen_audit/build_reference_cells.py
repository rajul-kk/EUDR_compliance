"""Build results/reference_cells.npz: per-grid Hansen label (re-derived) and reference class group.

usage: python -I build_reference_cells.py <extracted_labels_dir> <out.npz>
Needs internet (public Hansen tiles, no key). Cached per grid under the out directory.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import geopandas as gpd  # noqa: E402

from hansen_audit import reference as R  # noqa: E402


def main(labels_dir, out):
    gs = sorted(glob.glob(os.path.join(labels_dir, "grid_*")))
    res = {}
    for d in gs:
        gid = d[-2:]
        f = glob.glob(os.path.join(d, "user_land_use_*.gpkg"))[0]
        g = gpd.read_file(f).to_crs(4326)
        w, s, e, n = g.total_bounds
        tc, (w0, n0) = R.read_window("treecover2000", w, s, e, n)
        ly, _ = R.read_window("lossyear", w, s, e, n)
        H, W = tc.shape
        cid, cov = R.rasterise_reference(g, w0, n0, (H, W))
        res[f"hansen_{gid}"] = R.hansen_label(tc, ly)
        res[f"classid_{gid}"] = cid
        res[f"cov_{gid}"] = cov.astype(np.float16)
        res[f"origin_{gid}"] = np.array([w0, n0])
        print(gid, tc.shape, "annotated", float((cov >= 0.5).mean()), flush=True)
    np.savez_compressed(out, **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
