"""Download budget state: bytes survive a killed run, failed items are terminal, the stop check reserves an item."""
import json
import time

import pytest

from hansen_audit.budget import DownloadBudget, exclusive_download, run_guarded


class Net:
    def __init__(self):
        self.bytes = 1_000

    def __call__(self):
        return self.bytes


def test_a_killed_run_still_records_its_in_flight_bytes(tmp_path):
    net = Net()
    b = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0.01, net_bytes=net)
    net.bytes += 300_000_000  # mid-download, nothing recorded by the script yet
    time.sleep(0.1)
    # the process dies here: no close(), no record(); the heartbeat's last write is all that remains
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["runs"][-1]["bytes_recv"] == 300_000_000
    b._stop.set()
    nxt = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=net)
    assert nxt.used() == 300_000_000


def test_a_failed_item_is_not_downloaded_again_unless_asked(tmp_path):
    net = Net()
    b = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=net)
    b.record("01_2020", {"bytes_recv": 5}, failed=True)
    b.close()
    out = str(tmp_path / "grid_01_2020.npz")
    assert DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=net).done("01_2020", out)
    assert not DownloadBudget(str(tmp_path), 1.0, "grids", retry_failed=True, heartbeat_s=0,
                              net_bytes=net).done("01_2020", out)


def test_the_budget_check_reserves_room_for_the_next_item(tmp_path):
    net = Net()
    b = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=net)
    net.bytes += 400_000_000
    b.record("01_2020", {"bytes_recv": 400_000_000}, failed=False)
    assert not b.exhausted()  # 0.4 GB used + 0.4 GB expected <= 1 GB
    net.bytes += 400_000_000
    b.record("01_ref", {"bytes_recv": 400_000_000}, failed=False)
    assert b.exhausted()      # 0.8 + 0.4 > 1: stop before overshooting
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["runs"][-1]["stopped_by_budget"]


def test_the_manifest_is_never_half_written(tmp_path):
    b = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=Net())
    b.close()
    assert not (tmp_path / "manifest.json.part").exists()
    json.loads((tmp_path / "manifest.json").read_text())


def test_only_one_guarded_download_at_a_time(tmp_path):
    lock = str(tmp_path / "dl.lock")
    with exclusive_download(lock):
        with pytest.raises(RuntimeError, match="one at a time"):
            with exclusive_download(lock):
                pass
    with exclusive_download(lock):  # released after the first one ends
        pass


def test_run_guarded_records_a_raising_fetch_as_failed_with_the_secret_scrubbed(tmp_path):
    net = Net()
    b = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=net)

    def fetch(item):
        if item == "bad":
            raise ValueError("projects/secret-123 refused")
        return {"bytes_recv": 10}

    n_fail = run_guarded(b, ["ok", "bad"], fetch, workers=2, scrub=lambda s: s.replace("secret-123", "<p>"), log=lambda m: None)
    assert n_fail == 1
    assert b.items["ok"] == {"bytes_recv": 10, "failed": False}
    assert b.items["bad"]["failed"] and "secret" not in b.items["bad"]["error"]


def test_run_guarded_reserves_a_whole_batch_before_starting_it(tmp_path):
    net = Net()
    b = DownloadBudget(str(tmp_path), 1.0, "grids", heartbeat_s=0, net_bytes=net)
    b.record("seen", {"bytes_recv": 200_000_000}, failed=False)   # largest item so far: 0.2 GB
    calls = []
    run_guarded(b, list("abcdefgh"), lambda it: calls.append(it) or {"bytes_recv": 1}, workers=8, log=lambda m: None)
    assert calls == []   # 8 x 0.2 GB would pass the 1 GB budget, so the batch never starts
