"""Tests for ``lam.eval.teta_eval_toolkit.utils``."""

from __future__ import annotations

import pytest

from lam.eval.teta_eval_toolkit import utils as teta_utils


def test_validate_metrics_list_teta():
    class M:
        def __init__(self, name, fields):
            self._name = name
            self.fields = fields

        def get_name(self):
            return self._name

    assert teta_utils.validate_metrics_list([M("a", ["x"])]) == ["a"]


def test_validate_metrics_list_duplicate_field_raises():
    class M:
        def __init__(self, name, fields):
            self._name = name
            self.fields = fields

        def get_name(self):
            return self._name

    with pytest.raises(teta_utils.TrackEvalException):
        teta_utils.validate_metrics_list([M("a", ["x"]), M("b", ["x"])])


def test_get_track_id_str_variants():
    assert teta_utils.get_track_id_str({"track_id": 1}) == "track_id"
    assert teta_utils.get_track_id_str({"instance_id": 1}) == "instance_id"
    assert teta_utils.get_track_id_str({"scalabel_id": 1}) == "scalabel_id"


def test_get_track_id_str_missing_raises():
    with pytest.raises(AssertionError):
        teta_utils.get_track_id_str({})
