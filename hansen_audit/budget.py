"""Download budget and resume state shared by the S2, S1 and E4 download scripts.

State lives in <out_dir>/manifest.json: one entry per run (system-wide bytes received) and one per item (grid/period or
window). An item is finished when its output file exists, or when its manifest entry says it failed (nothing usable
came back); failed items are skipped on later runs unless retry_failed is set, so they do not spend budget every run.

The budget is the sum of bytes over all runs. A heartbeat thread rewrites the manifest every few seconds, so a run
that is killed mid-item still records (almost) all the bytes it received. Before each item the budget check reserves
room for one more item, estimated as the largest item seen so far.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence

LOCK_PATH = os.path.join(tempfile.gettempdir(), "hansen_audit_download.lock")


def _net_bytes() -> int:
    import psutil

    return psutil.net_io_counters().bytes_recv


@contextmanager
def exclusive_download(path: str = LOCK_PATH) -> Iterator[None]:
    """Hold a machine-wide lock for the length of a guarded download. The byte count is system-wide, so two guarded
    downloads at once would each count the other's traffic; the second one stops here instead."""
    fh = open(path, "a+")  # noqa: SIM115 - the open handle is the lock
    try:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another guarded download holds {path}; run downloads one at a time") from None
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # Windows releases a closed handle's lock lazily
    finally:
        fh.close()


class DownloadBudget:
    def __init__(self, out_dir: str, budget_gb: float, section: str, retry_failed: bool = False,
                 heartbeat_s: float = 15.0, net_bytes: Callable[[], int] = _net_bytes):
        self.path = os.path.join(out_dir, "manifest.json")
        self.section = section
        self.limit = budget_gb * 1e9
        self.retry_failed = retry_failed
        self._net = net_bytes
        self._lock = threading.Lock()
        self.manifest = json.load(open(self.path)) if os.path.exists(self.path) else {"runs": []}
        self.manifest.setdefault("runs", [])
        self.manifest.setdefault(section, {})
        self.used_before = sum(r.get("bytes_recv", 0) for r in self.manifest["runs"])
        self.net0 = self._net()
        self.run = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes_recv": 0, "stopped_by_budget": False}
        self.manifest["runs"].append(self.run)
        self._stop = threading.Event()
        self._beat: Optional[threading.Thread] = None
        if heartbeat_s > 0:
            self._beat = threading.Thread(target=self._heartbeat, args=(heartbeat_s,), daemon=True)
            self._beat.start()

    @property
    def items(self) -> dict:
        return self.manifest[self.section]

    def used(self) -> int:
        return self.used_before + self._net() - self.net0

    def save(self) -> None:
        with self._lock:
            self.run["bytes_recv"] = self._net() - self.net0
            tmp = self.path + ".part"
            with open(tmp, "w") as fh:
                json.dump(self.manifest, fh, indent=1)
            os.replace(tmp, self.path)

    def _heartbeat(self, every: float) -> None:
        while not self._stop.wait(every):
            self.save()

    def done(self, key: str, out_file: str) -> bool:
        """True when the item needs no download: its file exists, or it failed before and retry is off."""
        if os.path.exists(out_file):
            return True
        return bool(self.items.get(key, {}).get("failed")) and not self.retry_failed

    def next_item_estimate(self) -> int:
        return max((int(m.get("bytes_recv", 0)) for m in self.items.values()), default=0)

    def exhausted(self, n_next: int = 1) -> bool:
        """True (and recorded) when `n_next` more items would likely take the total over the budget."""
        if self.used() + n_next * self.next_item_estimate() <= self.limit:
            return False
        self.run["stopped_by_budget"] = True
        self.save()
        return True

    def record(self, key: str, meta: dict, failed: bool) -> None:
        with self._lock:
            self.items[key] = dict(meta, failed=failed)
        self.save()

    def close(self) -> None:
        self._stop.set()
        if self._beat is not None:
            self._beat.join()
        self.save()


def run_guarded(budget: "DownloadBudget", items: Sequence, fetch: Callable[[object], dict], key: Callable[[object], str] = str,
                workers: int = 8, scrub: Callable[[str], str] = str, log: Callable[[str], None] = print) -> int:
    """Fetch `items` in parallel batches of `workers`, stopping before a batch that would likely pass the budget.

    fetch(item) does the download and returns the item's meta (with bytes_recv, the guard's per-item estimate); if it
    raises, the item is recorded as failed with the error passed through `scrub` (to remove secrets) and cut to 300
    characters. Every item is recorded under key(item). Returns the number of failed items; closing the budget is the
    caller's job."""
    def one(item):
        try:
            return item, fetch(item), False
        except Exception as ex:  # noqa: BLE001 - recorded per item, retried with retry_failed
            return item, {"error": scrub(f"{type(ex).__name__}: {ex}")[:300]}, True

    n_fail = 0
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(items), workers):
            if budget.exhausted(n_next=workers):
                log(f"budget reached ({budget.used() / 1e9:.3f} GB), stopping")
                break
            for item, meta, failed in pool.map(one, items[i:i + workers]):
                budget.record(key(item), meta, failed=failed)
                n_fail += failed
            if (i // workers) % 25 == 0:
                log(f"{min(i + workers, len(items))}/{len(items)} done, {n_fail} failed, {budget.used() / 1e9:.3f} GB")
    return n_fail
