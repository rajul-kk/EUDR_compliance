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
