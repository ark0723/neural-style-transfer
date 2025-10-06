import unittest
import torch
import torch.nn as nn
from loss import ContentLoss, StyleLoss


class TestContentLoss(unittest.TestCase):
    def setUp(self):
        """Set up a new ContentLoss instance before each test."""
        self.content_loss = ContentLoss()

    def test_content_loss_forward(self):
        """Tests the basic forward pass of ContentLoss."""
        # Create sample tensors with random data
        generated = torch.randn(1, 64, 128, 128)
        target = torch.randn(1, 64, 128, 128)

        # Calculate loss
        loss = self.content_loss(generated, target)

        # --- Assertions ---
        # 1. The output should be a torch tensor.
        self.assertTrue(torch.is_tensor(loss))
        # 2. The loss should be a scalar (a 0-dimensional tensor).
        self.assertEqual(loss.dim(), 0)
        # 3. The loss value should be non-negative.
        self.assertGreaterEqual(loss.item(), 0)

    def test_content_loss_is_zero_for_identical_tensors(self):
        """Tests if the loss is zero when inputs are identical, which is the expected behavior."""
        # Create identical tensors
        tensor = torch.randn(1, 64, 128, 128)

        # Use .clone() to ensure it's a copy, not the same object in memory
        loss = self.content_loss(tensor, tensor.clone())

        # Check if loss is exactly 0
        self.assertEqual(loss.item(), 0)


class TestStyleLoss(unittest.TestCase):
    def setUp(self):
        """Creates a new StyleLoss object for each test to ensure test independence."""
        self.style_loss = StyleLoss()

    def test_gram_matrix_shape(self):
        """Tests if the gram_matrix method returns the correct shape (batch, channels, channels)."""
        feature_map = torch.randn(1, 64, 128, 128)  # (batch, channels, height, width)
        gram = self.style_loss.gram_matrix(feature_map)

        # Expected output shape for a single batch item is (batch_size, num_channels, num_channels)
        self.assertEqual(gram.size(), torch.Size([1, 64, 64]))

    def test_gram_matrix_values(self):
        """Tests the gram_matrix calculation with a known, simple input."""
        # Create a simple test case with a predictable outcome
        feature_map = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
        gram = self.style_loss.gram_matrix(feature_map)

        # Manually calculate the expected result:
        # The feature map reshapes to (1, 1, 4).
        # Gram matrix = feature * feature_transpose = [[1*1 + 2*2 + 3*3 + 4*4]] = [[30]]
        # Normalization divides by the number of elements in the feature map (1*2*2 = 4)
        # Expected value is 30 / 4 = 7.5
        # The expected shape is (batch, channels, channels) -> (1, 1, 1)
        expected = torch.tensor([[[30.0]]]) / 4

        # Check if the calculated gram matrix is close to the expected value
        self.assertTrue(torch.allclose(gram, expected, rtol=1e-5))

    def test_gram_matrix_batch_invariance(self):
        """Tests if the Gram matrix shape (excluding batch dim) is independent of batch size."""
        feature_map1 = torch.randn(1, 32, 64, 64)  # Batch size 1
        feature_map2 = torch.randn(2, 32, 64, 64)  # Batch size 2

        gram1 = self.style_loss.gram_matrix(feature_map1)  # Expected shape: (1, 32, 32)
        gram2 = self.style_loss.gram_matrix(feature_map2)  # Expected shape: (2, 32, 32)

        # The shape of the gram matrix for each item in the batch should be (channels, channels).
        # We compare the shapes starting from the second dimension (index 1).
        self.assertEqual(gram1.size()[1:], gram2.size()[1:])

    def test_style_loss_forward(self):
        """Tests the basic forward pass of StyleLoss."""
        generated = torch.randn(1, 64, 128, 128)
        target = torch.randn(1, 64, 128, 128)

        # 1. Run the forward function. The return value is the 'generated' tensor and is ignored here.
        self.style_loss(generated, target)

        # 2. The calculated loss is stored as an attribute of the object.
        loss = self.style_loss.loss

        # --- Assertions ---
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_style_loss_is_zero_for_identical_tensors(self):
        """Tests if the style loss is zero for identical inputs."""
        tensor = torch.randn(1, 64, 128, 128)

        # Run the forward pass with identical tensors
        self.style_loss(tensor, tensor.clone())
        loss = self.style_loss.loss

        # Use assertAlmostEqual as minor precision errors can occur during Gram matrix calculation
        self.assertAlmostEqual(loss.item(), 0, places=6)
