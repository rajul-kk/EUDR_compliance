"""Build results/features.npz for all complete (baseline+current+label+csv) farms.
usage: python -I build_features.py <data_root> <out.npz>"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit.io import index_images, index_labels, load_farms
from hansen_audit.features import farm_features, CHANNELS


def main(root, out):
    root = Path(root)
    farms = load_farms(root).drop_duplicates("key").set_index("key")
    b, c, l = index_images(root / "2020_baseline"), index_images(root / "2024_current"), index_labels(root / "hansen_labels")
    keys = sorted(set(b) & set(c) & set(l) & set(farms.index))
    X, Y, V, K = [], [], [], []
    for i, k in enumerate(keys):
        x, y, v, cov = farm_features(b[k][0][1], c[k][0][1], l[k])
        assert cov == 1.0, (k, cov)
        X.append(x); Y.append(y); V.append(v); K.append(k)
        if i % 50 == 0:
            print(i, len(keys), flush=True)
    np.savez_compressed(out, X=np.stack(X), Y=np.stack(Y), V=np.stack(V), keys=np.array(K),
                        country=farms.loc[K, "country_iso2"].values, crop=farms.loc[K, "crop_type"].values,
                        channels=np.array(CHANNELS))
    print("saved", len(K))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
