"""Tests for ``lam.agent.helpers.color_map``."""

from __future__ import annotations

import numpy as np

from lam.agent.helpers.color_map import colormap, random_color, random_colors


def test_colormap_shapes_and_maximum():
    c255 = colormap(rgb=True, maximum=255)
    assert c255.shape[1] == 3
    assert c255.max() <= 255
    c1 = colormap(rgb=False, maximum=1)
    assert c1.max() <= 1


def test_random_color_deterministic_with_seed():
    np.random.seed(0)
    a = random_color(rgb=True, maximum=1)
    np.random.seed(0)
    b = random_color(rgb=True, maximum=1)
    assert np.allclose(a, b)


def test_random_colors_count():
    np.random.seed(0)
    colors = random_colors(3, rgb=True, maximum=255)
    assert len(colors) == 3
