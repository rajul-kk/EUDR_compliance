"""Download budget state: bytes survive a killed run, failed items are terminal, the stop check reserves an item."""
import json
import time

from hansen_audit.budget import DownloadBudget


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
