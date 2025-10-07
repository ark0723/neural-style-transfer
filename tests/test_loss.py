import unittest
import torch
import torch.nn as nn

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss import ContentLoss, StyleLoss


class TestContentLoss(unittest.TestCase):
    def setUp(self):
        """Set up test data that will be used across multiple tests."""
        self.target = torch.randn(1, 64, 128, 128)
        self.content_loss = ContentLoss(self.target)

    def test_content_loss_forward(self):
        """Tests the basic forward pass of ContentLoss."""
        # Create a generated tensor with random data
        generated = torch.randn(1, 64, 128, 128)

        # The forward pass should return the input tensor unchanged
        output = self.content_loss(generated)
        self.assertTrue(torch.equal(output, generated))

        # --- Loss Value Assertions ---
        # 1. The loss should be a torch tensor
        self.assertTrue(torch.is_tensor(self.content_loss.loss))
        # 2. The loss should be a scalar (a 0-dimensional tensor)
        self.assertEqual(self.content_loss.loss.dim(), 0)
        # 3. The loss value should be non-negative
        self.assertGreaterEqual(self.content_loss.loss.item(), 0)

    def test_content_loss_is_zero_for_identical_tensors(self):
        """Tests if the loss is zero when input matches target exactly."""
        # Use the target tensor as the generated tensor
        output = self.content_loss(self.target)

        # Check if loss is exactly 0
        self.assertEqual(self.content_loss.loss.item(), 0)
        # Verify that the output tensor is unchanged
        self.assertTrue(torch.equal(output, self.target))

    def test_target_detached(self):
        """Tests if the target tensor is properly detached from the computation graph."""
        self.assertFalse(self.content_loss.target.requires_grad)


class TestStyleLoss(unittest.TestCase):
    def setUp(self):
        """Creates test data and StyleLoss instance for each test."""
        self.target = torch.randn(1, 64, 128, 128)
        self.style_loss = StyleLoss(self.target)

    def test_gram_matrix_shape(self):
        """Tests if the gram_matrix method returns the correct shape."""
        feature_map = torch.randn(2, 64, 128, 128)  # batch_size=2
        gram = self.style_loss.gram_matrix(feature_map)

        # Expected shape should be (batch_size, channels, channels)
        expected_shape = torch.Size([2, 64, 64])
        self.assertEqual(gram.size(), expected_shape)

    def test_gram_matrix_values(self):
        """Tests the gram_matrix calculation with a known, simple input."""
        # Create a simple test case with predictable outcome
        feature_map = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
        gram = self.style_loss.gram_matrix(feature_map)

        # Manual calculation for verification:
        # Features reshaped to (1, 1, 4)
        # After bmm and normalization by (1 * 2 * 2), result should be
        # [[[30.0]]] / 4 = [[[7.5]]]
        expected = torch.tensor([[[7.5]]])

        # Check if the calculated gram matrix matches expected value
        self.assertTrue(torch.allclose(gram, expected, rtol=1e-5))

    def test_gram_matrix_batch_processing(self):
        """Tests if gram matrix computation works correctly with multiple batches."""
        # Create two identical feature maps in a batch
        single_map = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        feature_map = torch.cat([single_map, single_map], dim=0)  # Shape: (2, 1, 2, 2)

        gram = self.style_loss.gram_matrix(feature_map)

        # Each item in the batch should produce the same gram matrix
        expected = torch.tensor([[[7.5]], [[7.5]]])  # Shape: (2, 1, 1)
        self.assertTrue(torch.allclose(gram, expected, rtol=1e-5))

    def test_style_loss_forward(self):
        """Tests the basic forward pass of StyleLoss."""
        generated = torch.randn(1, 64, 128, 128)

        # 1. The forward pass should return the input tensor unchanged
        output = self.style_loss(generated)
        self.assertTrue(torch.equal(output, generated))

        # 2. Check the computed loss
        self.assertTrue(torch.is_tensor(self.style_loss.loss))
        self.assertEqual(self.style_loss.loss.dim(), 0)
        self.assertGreaterEqual(self.style_loss.loss.item(), 0)

    def test_style_loss_is_zero_for_identical_tensors(self):
        """Tests if the style loss is zero for identical inputs."""
        # The forward pass with the target tensor should produce zero loss
        output = self.style_loss(self.target)

        # Check if loss is very close to zero (allowing for floating point precision)
        self.assertAlmostEqual(self.style_loss.loss.item(), 0, places=6)
        # Verify that the output tensor is unchanged
        self.assertTrue(torch.equal(output, self.target))

    def test_target_detached(self):
        """Tests if the target Gram matrix is properly detached."""
        self.assertFalse(self.style_loss.target.requires_grad)
