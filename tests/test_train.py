"""
Unit tests for the training utility functions.
Tests image loading, tensor conversion, and device handling.
"""

import os
import torch
import pytest
from PIL import Image
import numpy as np
from torchvision import transforms

# Add parent directory to Python path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_image_tensor, tensor_to_image


@pytest.fixture
def device():
    """Returns the appropriate device for the current environment."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a temporary test image file."""
    # Create a simple RGB test image
    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_tensor(device):
    """Creates a sample image tensor on the appropriate device."""
    # Create a simple tensor with recognizable pattern
    # Red channel full, others empty
    tensor = torch.zeros(1, 3, 100, 100, device=device)
    tensor[0, 0, :, :] = 1.0  # Red channel = 1
    return tensor


class TestLoadImageTensor:
    """Tests for the load_image_tensor function."""

    def test_basic_loading(self, sample_image_path, device):
        """Tests if image loading works with basic parameters."""
        size = 128
        tensor = load_image_tensor(sample_image_path, size, device)

        # Check tensor properties
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4  # (batch, channel, height, width)
        assert tensor.shape == (1, 3, size, size)
        # Compare device types instead of full device objects
        assert tensor.device.type == device.type
        assert tensor.dtype == torch.float32

    def test_size_parameter(self, sample_image_path, device):
        """Tests if resizing works correctly."""
        sizes = [64, 128, 256]

        for size in sizes:
            tensor = load_image_tensor(sample_image_path, size, device)
            assert tensor.shape == (1, 3, size, size)

    def test_value_range(self, sample_image_path, device):
        """Tests if output tensor values are in [0, 1] range."""
        size = 128
        tensor = load_image_tensor(sample_image_path, size, device)
        assert torch.all(tensor >= 0)
        assert torch.all(tensor <= 1)

    def test_invalid_path(self, device):
        """Tests if function handles invalid file paths appropriately."""
        with pytest.raises(FileNotFoundError):
            load_image_tensor("nonexistent_image.jpg", 128, device)


class TestTensorToImage:
    """Tests for the tensor_to_image function."""

    def test_basic_conversion(self, sample_tensor):
        """Tests basic tensor to image conversion."""
        image = tensor_to_image(sample_tensor)

        # Check if result is PIL Image
        assert isinstance(image, Image.Image)
        # Check if dimensions are correct (removing batch dimension)
        assert image.size == (100, 100)
        # Check if it's RGB
        assert image.mode == "RGB"

    def test_value_clamping(self, device):
        """Tests if values outside [0,1] are properly clamped."""
        # Create tensor with values outside [0,1]
        tensor = torch.randn(1, 3, 50, 50, device=device)
        image = tensor_to_image(tensor)

        # Convert back to numpy array for checking
        arr = np.array(image)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_shape_preservation(self, device):
        """Tests if spatial dimensions are preserved."""
        shapes = [(1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)]

        for shape in shapes:
            tensor = torch.rand(*shape, device=device)
            image = tensor_to_image(tensor)
            assert image.size == shape[2:]  # Height and width should match

    def test_color_preservation(self, sample_tensor):
        """Tests if color information is preserved correctly."""
        image = tensor_to_image(sample_tensor)
        # Convert image back to numpy array
        arr = np.array(image)

        # Check if red channel is maximum (our sample tensor has only red channel = 1)
        assert np.all(arr[:, :, 0] > arr[:, :, 1])  # Red > Green
        assert np.all(arr[:, :, 0] > arr[:, :, 2])  # Red > Blue

    def test_batch_dimension_handling(self, device):
        """Tests if function properly handles batch dimension removal."""
        # Test with single batch dimension
        tensor = torch.rand(1, 3, 50, 50, device=device)
        image = tensor_to_image(tensor)
        assert isinstance(image, Image.Image)
        assert image.size == (50, 50)

        # Should raise error for batch size > 1
        with pytest.raises(ValueError):
            tensor = torch.rand(2, 3, 50, 50, device=device)  # Batch size of 2
            tensor_to_image(tensor)
