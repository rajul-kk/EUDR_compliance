"""
Append-only audit trail for EUDR compliance assessments.

Each pipeline run writes one AuditEntry to audit_log.jsonl.  Entries record
SHA-256 hashes of every input satellite image and every output prediction mask
so any past assessment can be independently verified.  The log file is never
modified in place — only appended — which makes tampering detectable via hash
chaining.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    """Return 'sha256:<hex>' for a file, or 'sha256:missing' if not found."""
    if not os.path.exists(path):
        return "sha256:missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_string(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def _chain_hash(previous_hash: Optional[str], entry_json: str) -> str:
    """Hash of (previous_chain_hash + current_entry_json) for tamper detection."""
    payload = (previous_hash or "") + entry_json
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    run_id: str                          # UUID for this pipeline run
    timestamp: str                       # ISO-8601 UTC
    model_type: str                      # deeplab | tessera | tessera-embed
    model_version: str                   # semver or git SHA
    baseline_year: int                   # 2020
    assessment_year: int                 # 2024
    input_image_hashes: Dict[str, str]   # {filename: sha256:...}
    output_mask_hashes: Dict[str, str]   # {filename: sha256:...}
    report_hash: str                     # sha256 of output CSV report
    summary: Dict                        # {total_farms, violations, warnings, compliant}
    chain_hash: str = ""                 # hash linking to previous entry
    operator_id: Optional[str] = None   # optional: submitting operator


@dataclass
class AuditLog:
    """Append-only JSONL audit log."""

    log_path: str

    def _last_chain_hash(self) -> Optional[str]:
        """Read the chain_hash of the last written entry without loading the whole file."""
        if not os.path.exists(self.log_path):
            return None
        last: Optional[str] = None
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            return None
        try:
            return json.loads(last).get("chain_hash")
        except json.JSONDecodeError:
            return None

    def append(self, entry: AuditEntry) -> AuditEntry:
        """Compute chain hash and append entry to the log. Returns the written entry."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        entry_dict = asdict(entry)
        entry_dict["chain_hash"] = ""  # placeholder
        entry_json = json.dumps(entry_dict, sort_keys=True)

        prev_hash = self._last_chain_hash()
        chain = _chain_hash(prev_hash, entry_json)
        entry.chain_hash = chain
        entry_dict["chain_hash"] = chain

        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry_dict, sort_keys=True) + "\n")

        return entry

    def verify(self) -> List[str]:
        """Walk the log and verify every chain hash. Returns list of violation messages."""
        if not os.path.exists(self.log_path):
            return ["Log file does not exist."]

        violations: List[str] = []
        previous_chain: Optional[str] = None

        with open(self.log_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as exc:
                    violations.append(f"Line {lineno}: JSON parse error — {exc}")
                    continue

                stored_chain = entry.get("chain_hash", "")
                check_dict = {k: v for k, v in entry.items() if k != "chain_hash"}
                check_dict["chain_hash"] = ""
                expected = _chain_hash(previous_chain, json.dumps(check_dict, sort_keys=True))

                if stored_chain != expected:
                    violations.append(
                        f"Line {lineno} (run_id={entry.get('run_id')}): "
                        f"chain hash mismatch — expected {expected}, got {stored_chain}"
                    )
                previous_chain = stored_chain

        return violations

    def history(self, farm_id: Optional[str] = None) -> List[Dict]:
        """Return all entries, optionally filtered to those mentioning a farm_id."""
        if not os.path.exists(self.log_path):
            return []
        results = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if farm_id is None or farm_id in json.dumps(entry):
                    results.append(entry)
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_audit_entry(
    model_type: str,
    model_version: str,
    input_image_dir: str,
    prediction_dir: str,
    report_csv_path: str,
    summary: Dict,
    baseline_year: int = 2020,
    assessment_year: int = 2024,
    operator_id: Optional[str] = None,
) -> AuditEntry:
    """Construct an AuditEntry by hashing all inputs and outputs."""

    def _hash_dir(directory: str, exts: tuple = (".tif", ".tiff")) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        if not os.path.isdir(directory):
            return hashes
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(exts):
                fpath = os.path.join(directory, fname)
                hashes[fname] = _sha256(fpath)
        return hashes

    return AuditEntry(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_type=model_type,
        model_version=model_version,
        baseline_year=baseline_year,
        assessment_year=assessment_year,
        input_image_hashes=_hash_dir(input_image_dir),
        output_mask_hashes=_hash_dir(prediction_dir),
        report_hash=_sha256(report_csv_path),
        summary=summary,
        operator_id=operator_id,
    )
