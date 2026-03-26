"""Tests for ``lam.cli`` helpers (no full model / pipeline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lam import cli


def test_default_checkpoint_path_env_overrides(tmp_path, monkeypatch):
    p = tmp_path / "w.pt"
    p.write_bytes(b"x")
    monkeypatch.setenv("CHECKPOINT_PATH", str(p))
    assert cli._default_checkpoint_path() == str(p)
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)


def test_default_checkpoint_path_assets_file(monkeypatch):
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    repo_root = Path(cli.__file__).resolve().parent.parent
    weights = repo_root / "assets" / "weights" / "sam3.pt"
    if weights.is_file():
        assert cli._default_checkpoint_path() == str(weights)
    else:
        assert cli._default_checkpoint_path() is None
