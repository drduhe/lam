"""Tests for ``lam.train.data.serialization_utils``."""

from lam.train.data.serialization_utils import NumpySerializedList, TorchSerializedList


def test_numpy_serialized_list_roundtrip():
    data = [{"k": i, "nested": {"x": i * 2}} for i in range(5)]
    s = NumpySerializedList(data)
    assert len(s) == 5
    for i in range(5):
        assert s[i] == data[i]


def test_torch_serialized_list_roundtrip():
    data = [{"k": i} for i in range(3)]
    s = TorchSerializedList(data)
    assert len(s) == 3
    assert s[2] == data[2]
