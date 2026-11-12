"""Hash-chained audit log: what verify() can and cannot detect."""
import pytest

from audit_trail import AuditEntry, AuditLog


def _entry(i):
    return AuditEntry(run_id=f"r{i}", timestamp="t", model_type="deeplab", model_version="v", baseline_year=2020,
                      assessment_year=2024, input_image_hashes={}, output_mask_hashes={}, report_hash="h",
                      summary={"total_farms": i})


def _log(tmp_path, n=3):
    log = AuditLog(str(tmp_path / "audit.jsonl"))
    for i in range(n):
        log.append(_entry(i))
    return log


def test_an_intact_log_verifies(tmp_path):
    log = _log(tmp_path)
    assert log.verify() == []
    assert log.verify(expected_head=log.head()) == []


def test_removing_the_newest_entry_is_caught_only_against_a_recorded_head(tmp_path):
    log = _log(tmp_path)
    head = log.head()
    path = tmp_path / "audit.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    assert log.verify() == []  # a shorter chain is still a valid chain
    assert log.verify(expected_head=head)


def test_an_edited_entry_is_caught(tmp_path):
    log = _log(tmp_path)
    path = tmp_path / "audit.jsonl"
    path.write_text(path.read_text().replace('"total_farms": 1', '"total_farms": 9'))
    assert log.verify()


def test_appending_after_a_cut_off_line_is_refused(tmp_path):
    log = _log(tmp_path)
    path = tmp_path / "audit.jsonl"
    with path.open("a") as fh:
        fh.write('{"run_id": "half')
    with pytest.raises(ValueError, match="not a complete entry"):
        log.append(_entry(9))
