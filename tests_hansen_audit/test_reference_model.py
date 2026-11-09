import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from hansen_audit import reference as R, model as M, features as Ft


def test_tile_name():
    assert R.tile_name(102.9, -1.7) == "00N_100E"
    assert R.tile_name(101.5, 1.2) == "10N_100E"
    assert R.tile_name(-60.2, -3.0) == "00N_070W"


def test_hansen_label_matches_generate_labels_recipe():
    tc = np.array([5, 50, 50, 50, 50, 50, 5])
    ly = np.array([0, 0, 21, 24, 15, 25, 22])
    # low cover->0 ; forest no loss->1 ; loss 21,24 ->2 ; loss 2015 ->0 ; lossyear 25 counts as no_pre but not post -> 1
    assert R.hansen_label(tc, ly).tolist() == [0, 1, 2, 2, 0, 1, 0]


def test_class_group():
    assert R.class_group(9) == R.G_FOREST and R.class_group(1) == R.G_PLANT and R.class_group(10) == R.G_OTHER
    assert R.class_group(3) == R.G_OTHER and R.class_group(7) == R.G_PLANT


def test_rasterise_reference_majority():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    # one 2x2-cell window; left cell column class 9 (forest), right column class 1 (palm)
    w, n, res = 100.0, 0.0, R.RES
    g = gpd.GeoDataFrame({"class_ID": [9, 1]}, geometry=[box(w, n - 2 * res, w + res, n), box(w + res, n - 2 * res, w + 2 * res, n)], crs=4326)
    cid, cov = R.rasterise_reference(g, w, n, (2, 2))
    assert cid.tolist() == [[9, 1], [9, 1]] and (cov == 1).all()


def test_patches_and_training_smoke():
    rng = np.random.default_rng(0)
    Fn, C, G = 6, 15, 8
    X = rng.normal(size=(Fn, C, G, G)).astype(np.float32)
    Y = rng.integers(0, 3, size=(Fn, G, G)).astype(np.uint8)
    V = np.ones((Fn, G, G), np.float32)
    # make class separable: loss cells have channel 0 shifted
    X[:, 0] += (Y == 2) * 3.0
    P, y, f = M.make_patches(X, Y, V)
    assert P.shape[1:] == (C, 3, 3) and set(np.unique(y)) <= {0, 1} and len(P) == len(y) == len(f)
    assert len(y) == ((Y >= 1)).sum()
    Pa, = M.standardise(P)[:1]
    m = M.train_model(Pa, y, Pa, y, "ce", epochs=80, bs=64)
    pred = (M.predict_proba(m, Pa)[:, 1] > .5).astype(int)
    assert (pred == y).mean() > 0.8
    assert M.f1_binary(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1])) == pytest.approx(0.5)


def test_make_patches_drops_invalid():
    X = np.zeros((1, 15, 5, 5), np.float32); X[0, :, 2, 2] = np.nan
    Y = np.full((1, 5, 5), 2, np.uint8); V = np.ones((1, 5, 5), np.float32); V[0, 0, 0] = 0.1
    P, y, f = M.make_patches(X, Y, V)
    assert len(y) == 25 - 2


def test_cell_means_masks_clouds():
    img = np.zeros((5, 74, 74), np.float32)
    img[:4] = 0.2; img[3] = 0.4; img[4] = 4
    img[:4, :37, :] = 9.0; img[4, :37, :] = 9          # cloudy top half (SCL 9) with garbage reflectance
    feat, valid = Ft._cell_means(img, grid=2)
    assert valid[0].tolist() == [0.0, 0.0] and valid[1].tolist() == [1.0, 1.0]
    assert feat[0, 1, 0] == pytest.approx(0.2, abs=1e-5)
