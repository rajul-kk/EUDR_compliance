"""Build results/features.npz for all complete (baseline+current+label+csv) farms.
usage: python -I build_features.py <data_root> <out.npz> [--composites]
--composites also reads the undated median composites (E6) and saves each farm's image type in `product`."""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit.io import COMPOSITE, index_images, index_labels, load_farms
from hansen_audit.features import farm_features, CHANNELS


def main(root, out, composites=False):
    root = Path(root)
    farms = load_farms(root).drop_duplicates("key").set_index("key")
    b, c = index_images(root / "2020_baseline", composites), index_images(root / "2024_current", composites)
    l = index_labels(root / "hansen_labels")
    keys = sorted(set(b) & set(c) & set(l) & set(farms.index))
    X, Y, V, K, T = [], [], [], [], []
    for i, k in enumerate(keys):
        x, y, v, cov = farm_features(b[k][0][1], c[k][0][1], l[k])
        assert cov == 1.0, (k, cov)
        X.append(x); Y.append(y); V.append(v); K.append(k)
        T.append("composite" if COMPOSITE in (b[k][0][0], c[k][0][0]) else "scene")
        if i % 50 == 0:
            print(i, len(keys), flush=True)
    np.savez_compressed(out, X=np.stack(X), Y=np.stack(Y), V=np.stack(V), keys=np.array(K),
                        country=farms.loc[K, "country_iso2"].values, crop=farms.loc[K, "crop_type"].values,
                        channels=np.array(CHANNELS), product=np.array(T))
    print("saved", len(K))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--composites" in sys.argv[3:])
