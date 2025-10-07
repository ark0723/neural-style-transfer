"""
Unit tests for the StyleTransfer model.

These tests use pytest's `monkeypatch` fixture to mock the heavy `torchvision.vgg19`
dependency. This ensures the tests are fast, isolated, and don't require
network access to download pretrained weights.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import StyleTransfer
from loss import ContentLoss, StyleLoss
import pytest


class FakeVGG(nn.Module):
    """A fast, lightweight mock of VGG19 that mimics its layer structure."""

    def __init__(self):
        super().__init__()
        # Build a feature stack that includes all necessary layers
        layers = []
        in_channels = 3
        out_channels = 64  # Start with 64 channels as in real VGG19

        # Create a simplified version of VGG19 that maintains tensor dimensions better
        for i in range(29):  # Up to index 28 (conv5_1)
            if i in [0, 5, 10, 19, 21, 28]:  # Conv layers at important indices
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
                in_channels = out_channels
                if i < 10:  # Increase channels as we go deeper
                    out_channels *= 2
            elif i % 2 == 0:  # Other conv layers
                layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            else:  # ReLU layers
                layers.append(nn.ReLU(inplace=True))

            # Add pooling layers at appropriate positions
            if i in [1, 6, 11, 20, 29]:  # After early convolutions
                layers.append(nn.MaxPool2d(2, 2, padding=0))

        self.features = nn.Sequential(*layers)

    def eval(self):
        """Match the interface of the real VGG19."""
        return self


def fake_vgg19(pretrained=True):  # signature matches real call site
    return FakeVGG()


def test_layer_indices_match_expected_names(monkeypatch):
    """Tests if the model initializes with the correct layer indices and names."""
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()

    # Check content layer
    assert model.content_layers_indices == {"conv4_2": 21}

    # Check style layers
    expected_style_layers = {
        "conv1_1": 0,
        "conv2_1": 5,
        "conv3_1": 10,
        "conv4_1": 19,
        "conv5_1": 28,
    }
    assert model.style_layers_indices == expected_style_layers


def test_build_model_structure(monkeypatch):
    """Tests if build_model returns the correct components and structure."""
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()

    # Create dummy inputs
    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)

    # Build the model
    final_model, content_losses, style_losses = model.build_model(
        content_img, style_img
    )

    # Check if we got the correct number of loss modules
    assert len(content_losses) == 1
    assert len(style_losses) == 5

    # Check if the model is a Sequential
    assert isinstance(final_model, nn.Sequential)

    # Verify loss module types
    assert all(isinstance(m, ContentLoss) for m in content_losses)
    assert all(isinstance(m, StyleLoss) for m in style_losses)


def test_model_layer_modifications(monkeypatch):
    """Tests if ReLU and MaxPool layers are properly modified during model building."""
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()

    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)

    final_model, _, _ = model.build_model(content_img, style_img)

    # Check layer modifications
    for module in final_model:
        if isinstance(module, nn.ReLU):
            # All ReLU layers should be non-inplace
            assert module.inplace == False
        elif isinstance(module, nn.MaxPool2d):
            # All MaxPool layers should have been converted to AvgPool
            assert isinstance(module, nn.AvgPool2d)


def test_feature_extraction(monkeypatch):
    """Tests if _extract_features correctly extracts features from specified layers."""
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()

    # Test with a sample input - use larger input size to prevent size issues
    x = torch.randn(1, 3, 256, 256)  # Increased from 64x64 to 256x256
    content_features = model._extract_features(x, model.content_layers_indices.values())
    style_features = model._extract_features(x, model.style_layers_indices.values())

    # Check if we got the correct number of features
    assert len(content_features) == 1
    assert len(style_features) == 5

    # Check if all outputs are tensors
    assert all(isinstance(f, torch.Tensor) for f in content_features)
    assert all(isinstance(f, torch.Tensor) for f in style_features)

    # Check if feature maps have valid dimensions
    for f in content_features + style_features:
        assert f.dim() == 4  # (batch, channels, height, width)
        assert f.size(0) == 1  # batch size
        assert all(s > 0 for s in f.shape)  # all dimensions should be positive


def test_model_truncation(monkeypatch):
    """Tests if the model is correctly truncated after the last loss layer."""
    monkeypatch.setattr(models, "vgg19", fake_vgg19)
    model = StyleTransfer()

    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)

    final_model, _, _ = model.build_model(content_img, style_img)

    # The last layer should be either a ContentLoss or StyleLoss
    assert isinstance(final_model[-1], (ContentLoss, StyleLoss))

    # Find the last loss layer
    last_loss_idx = -1
    for i, module in enumerate(final_model):
        if isinstance(module, (ContentLoss, StyleLoss)):
            last_loss_idx = i

    # Check that there are no layers after the last loss layer
    assert (
        last_loss_idx == len(final_model) - 1
    ), "Model should be truncated after last loss layer"
