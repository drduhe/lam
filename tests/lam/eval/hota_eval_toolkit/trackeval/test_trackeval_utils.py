"""Tests for ``lam.eval.hota_eval_toolkit.trackeval.utils``."""

from __future__ import annotations

from pathlib import Path

import pytest

from lam.eval.hota_eval_toolkit.trackeval import utils as te_utils


def test_init_config_fills_missing_keys():
    default = {"a": 1, "b": 2, "PRINT_CONFIG": False}
    cfg = te_utils.init_config({"a": 9}, default)
    assert cfg["a"] == 9
    assert cfg["b"] == 2


def test_get_code_path_exists():
    p = te_utils.get_code_path()
    assert Path(p).is_dir()


def test_validate_metrics_list_unique():
    class M:
        def __init__(self, name, fields):
            self._name = name
            self.fields = fields

        def get_name(self):
            return self._name

    names = te_utils.validate_metrics_list([M("m1", ["a"]), M("m2", ["b"])])
    assert names == ["m1", "m2"]


def test_validate_metrics_list_duplicate_name_raises():
    class M:
        fields = ["a"]

        def get_name(self):
            return "dup"

    with pytest.raises(te_utils.TrackEvalException):
        te_utils.validate_metrics_list([M(), M()])


def test_write_summary_results(tmp_path: Path):
    te_utils.write_summary_results([{"HOTA": 0.5, "extra": 1}], "cls", str(tmp_path))
    out = tmp_path / "cls_summary.txt"
    assert out.is_file()
