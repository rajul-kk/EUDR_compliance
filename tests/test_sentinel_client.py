"""Token refresh, atomic composite writes and manifest status in the Copernicus download client."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import sentinel_client as sc


@pytest.fixture(autouse=True)
def fresh_token_cache():
    sc.ACCESS_TOKEN, sc.TOKEN_EXPIRY = None, 0
    yield
    sc.ACCESS_TOKEN, sc.TOKEN_EXPIRY = None, 0


def _auth_response(token):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"access_token": token, "expires_in": 600})


def test_a_401_retries_with_a_new_token_not_the_cached_one():
    sc.ACCESS_TOKEN, sc.TOKEN_EXPIRY = "stale", 10**12  # still "valid" by the local clock, rejected by the server
    sent = []

    def fake_request(method, url, **kw):
        if url == sc.AUTH_URL:
            return _auth_response("fresh")
        sent.append(kw["headers"]["Authorization"])
        ok = kw["headers"]["Authorization"] == "Bearer fresh"
        return SimpleNamespace(status_code=200 if ok else 401, content=b"tiff", text="")

    with patch.object(sc, "request_with_retry", side_effect=fake_request):
        assert sc._download_scene_bytes([0, 0, 1, 1], "2020-06-01") == b"tiff"
    assert sent == ["Bearer stale", "Bearer fresh"]


def test_concurrent_callers_share_one_refresh():
    calls = []

    def fake_request(method, url, **kw):
        calls.append(url)
        import time

        time.sleep(0.05)  # long enough for every thread to see an empty cache without a lock
        return _auth_response("t")

    import threading

    with patch.object(sc, "request_with_retry", side_effect=fake_request):
        threads = [threading.Thread(target=sc.get_auth_token) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert calls == [sc.AUTH_URL]


def test_a_composite_is_never_left_half_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Crash(Exception):
        pass

    class Writer:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, arr):
            with open(self.path, "wb") as fh:
                fh.write(b"half")
            raise Crash()

    composite = np.zeros((5, 2, 2), dtype=np.float32)
    with patch.object(sc, "build_median_composite", return_value=(composite, {})), \
            patch.object(sc.rasterio, "open", side_effect=lambda path, mode, **kw: Writer(path)):
        with pytest.raises(Crash):
            sc.download_farm_image({"lat": 0, "lon": 0}, 2020, "f1", "2020-05-01", "2020-09-30", {})
    target = os.path.join("data", "raw_satellite", "2020_baseline", "f1_2020.tiff")
    assert not os.path.exists(target)
    assert not sc.glob.glob("data/raw_satellite/2020_baseline/f1_*.tiff")


def test_an_error_after_the_baseline_saved_only_fails_the_missing_year(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = str(tmp_path / "state" / "manifest.csv")

    def fake_download(row, year, farm_id, *a):
        if year == 2024:
            raise RuntimeError("network down")
        out = f"data/raw_satellite/2020_baseline/{farm_id}_2020.tiff"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").close()
        return out

    with patch.object(sc, "download_farm_image", side_effect=fake_download):
        ok = sc.process_single_farm({"farm_id": "f1", "lat": 0, "lon": 0}, 0, 0, sc.DEFAULT_PROFILE, manifest)
    assert ok is False
    df = pd.read_csv(manifest)
    status = {int(r.year): r.status for r in df.itertuples()}
    assert status == {2020: "downloaded", 2024: "failed"}
