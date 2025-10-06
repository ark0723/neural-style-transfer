"""
Unit tests for the StyleTransfer model.

These tests use pytest's `monkeypatch` fixture to mock the heavy `torchvision.vgg19`
dependency. This ensures the tests are fast, isolated, and don't require
network access to download pretrained weights.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from model import StyleTransfer
import pytest


class FakeVGG(nn.Module):
    """A fast, lightweight mock of VGG19 that mimics its layer structure."""

    def __init__(self):
        super().__init__()
        # Build a minimal feature stack with indices up to 28.
        self.features = nn.ModuleList([nn.Identity() for _ in range(29)])


def fake_vgg19(pretrained=True):  # signature matches real call site
    return FakeVGG()


def test_style_forward_returns_5_feature_maps(monkeypatch):
    """Verifies the forward pass in 'style' mode returns a list of 5 tensors."""
    # `monkeypatch.setattr` temporarily replaces the `vgg19` function within the `torchvision.models` module
    # with our `fake_vgg19` factory. This change only lasts for the duration of this single test function.
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()
    x = torch.randn(1, 3, 256, 256)
    feats = model(x, mode="style")
    assert isinstance(feats, list)
    assert len(feats) == 5
    for f in feats:
        assert isinstance(f, torch.Tensor)
        # The output shape is the same as the input because our FakeVGG only
        # contains Identity layers, which don't change the tensor shape.
        assert f.shape == x.shape


def test_content_forward_returns_1_feature_map(monkeypatch):
    """
    Tests if the model's forward pass in 'content' mode returns 1 feature map.
    """
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()
    x = torch.randn(1, 3, 256, 256)
    feats = model(x, mode="content")
    assert isinstance(feats, list)
    assert len(feats) == 1
    assert feats[0].shape == x.shape


def test_layer_indices_expected(monkeypatch):
    """
    Tests if the model initializes with the correct hard-coded layer indices.
    """
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()
    assert model.style_layers == [0, 5, 10, 19, 28]
    assert model.content_layers == [21]


def test_invalid_mode_raises(monkeypatch):
    """
    Tests if the model raises a NotImplementedError when given an invalid mode.
    """
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()
    x = torch.randn(1, 3, 64, 64)
    # `pytest.raises` checks that the code inside this block raises the expected error.
    # The test passes if the error is raised, and fails otherwise.
    with pytest.raises(NotImplementedError):
        _ = model(x, mode="invalid")
