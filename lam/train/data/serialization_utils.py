"""Serialize large Python lists for DataLoader workers to reduce copy-on-read RAM use.

See https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
and Detectron2-style serialized list patterns.
"""

from __future__ import annotations

import pickle
from typing import Any, List

import numpy as np


class NumpySerializedList:
    """Store a list of picklable objects in one uint8 numpy buffer (fork-friendly sharing)."""

    def __init__(self, lst: List[Any]):
        serialized = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in lst]
        self._addr = np.cumsum([len(x) for x in serialized], dtype=np.int64)
        self._lst = np.concatenate(serialized)

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx: int):
        start = 0 if idx == 0 else self._addr[idx - 1]
        end = self._addr[idx]
        return pickle.loads(memoryview(self._lst[start:end]))


class TorchSerializedList:
    """Same as NumpySerializedList but stores buffers as torch tensors (spawn/forkserver)."""

    def __init__(self, lst: List[Any]):
        import torch

        serialized = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in lst]
        self._addr = torch.from_numpy(np.cumsum([len(x) for x in serialized], dtype=np.int64))
        self._lst = torch.from_numpy(np.concatenate(serialized))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx: int):
        start = 0 if idx == 0 else self._addr[idx - 1].item()
        end = self._addr[idx].item()
        return pickle.loads(bytes(self._lst[start:end].numpy()))
