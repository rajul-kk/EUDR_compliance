"""The pipeline's exit status reflects failed stages."""
import subprocess
from unittest.mock import patch

import pytest

import pipeline_runner as pr


def _fail(*a, **kw):
    raise subprocess.CalledProcessError(3, a[0] if a else "cmd")


def test_a_failed_stage_exits_non_zero():
    with patch.object(pr.subprocess, "check_call", side_effect=_fail):
        assert pr.main(["--skip-inference"]) == 1


def test_a_clean_run_exits_zero():
    with patch.object(pr.subprocess, "check_call", return_value=0), \
            patch.object(pr.os.path, "exists", return_value=True):
        assert pr.main(["--skip-inference"]) == 0


def test_dds_export_without_an_operator_is_refused():
    with pytest.raises(SystemExit) as exc:
        pr.parse_args(["--export-dds"])
    assert exc.value.code == 2


def test_embed_change_refuses_one_folder_for_both_years(tmp_path):
    calls = []
    with patch.object(pr.subprocess, "check_call", side_effect=lambda *a, **k: calls.append(a)), \
            patch.object(pr.os.path, "exists", return_value=True):
        same = str(tmp_path)
        assert pr.main(["--skip-inference", "--model-type", "embed-change",
                        "--embeddings-dir-2020", same, "--embeddings-dir-2024", same]) == 1
        assert pr.main(["--skip-inference", "--model-type", "embed-change"]) == 1
    assert calls == []


def test_embed_change_trains_on_the_two_given_years(tmp_path):
    calls = []
    a, b = tmp_path / "e2020", tmp_path / "e2024"
    with patch.object(pr.subprocess, "check_call", side_effect=lambda cmd, **k: calls.append(cmd)), \
            patch.object(pr.os.path, "exists", return_value=True):
        assert pr.main(["--skip-inference", "--model-type", "embed-change",
                        "--embeddings-dir-2020", str(a), "--embeddings-dir-2024", str(b)]) == 0
    train = next(c for c in calls if any("change_embed_train" in str(x) for x in c))
    assert train[train.index("--embeddings-dir-t1") + 1] == str(a)
    assert train[train.index("--embeddings-dir-t2") + 1] == str(b)


def test_change_dataset_refuses_identical_year_folders(tmp_path):
    from change_embed_train import check_year_dirs

    with pytest.raises(ValueError, match="same folder"):
        check_year_dirs(str(tmp_path), str(tmp_path / "." ))
    check_year_dirs(str(tmp_path / "a"), str(tmp_path / "b"))
