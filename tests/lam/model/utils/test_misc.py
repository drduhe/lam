"""Tests for ``lam.model.utils.misc.copy_data_to_device``."""

from __future__ import annotations

from collections import defaultdict, namedtuple
from dataclasses import dataclass

import torch

from lam.model.utils.misc import copy_data_to_device


def test_copy_nested_structure_to_cpu():
    t = torch.tensor([1.0, 2.0])
    data = {
        "a": [t, {"b": t}],
        "c": defaultdict(list, {"k": (t,)}),
    }
    out = copy_data_to_device(data, torch.device("cpu"))
    torch.testing.assert_close(out["a"][0], t)


def test_copy_named_tuple():
    NT = namedtuple("NT", ["x"])
    t = torch.tensor([3.0])
    nt = NT(x=t)
    out = copy_data_to_device(nt, torch.device("cpu"))
    assert isinstance(out, NT)
    torch.testing.assert_close(out.x, t)


@dataclass
class _DC:
    a: torch.Tensor
    b: torch.Tensor


def test_copy_dataclass():
    t1 = torch.tensor(1.0)
    t2 = torch.tensor(2.0)
    dc = _DC(a=t1, b=t2)
    out = copy_data_to_device(dc, torch.device("cpu"))
    assert isinstance(out, _DC)
    torch.testing.assert_close(out.a, t1)
