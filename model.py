"""Neural style transfer feature extractor using pretrained VGG19.

This module exposes a `StyleTransfer` wrapper around VGG19 that returns
feature maps from canonical style and content layers, following Gatys et al.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from loss import ContentLoss, StyleLoss


class StyleTransfer(nn.Module):
    """
    Builds a style transfer model by injecting loss modules into a VGG19 network.
    This class handles loading the VGG19 model once and constructing the final
    sequential model used for training.
    """

    def __init__(self, device: torch.device):
        super(StyleTransfer, self).__init__()
        # Load the VGG19 model and its feature layers just once.
        # Load VGG19 features and immediately move them to the correct device.
        # Use the modern 'weights' API to avoid deprecation warnings.
        vgg19_features = (
            models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            .features.to(device)
            .eval()
        )
        # Define the canonical content and style layers by their indices in the VGG19 model.
        self.content_layers_indices = {"conv4_2": 21}
        self.style_layers_indices = {
            "conv1_1": 0,
            "conv2_1": 5,
            "conv3_1": 10,
            "conv4_1": 19,
            "conv5_1": 28,
        }

        self.vgg_features = vgg19_features

    def _extract_features(self, x: torch.Tensor, layer_indices: list[int]):
        """Helper function to extract features from specified layer indices."""
        features = []
        for idx, layer in enumerate(self.vgg_features):
            x = layer(x)
            if idx in layer_indices:
                features.append(x)
        return features

    def build_model(self, content_img: torch.Tensor, style_img: torch.Tensor):
        """
        Builds the final style transfer model with loss layers.

        Args:
            content_img (torch.Tensor): The content image tensor.
            style_img (torch.Tensor): The style image tensor.

        Returns:
            tuple: A tuple containing:
                - nn.Sequential: The final model with VGG and loss layers.
                - list: A list of the content loss modules.
                - list: A list of the style loss modules.
        """
        model = nn.Sequential()
        content_loss_modules = []
        style_loss_modules = []

        # --- Step 1: Get target features for loss calculation ---
        # No gradients needed for this part. only once implemented in the beginning
        with torch.no_grad():
            content_target_features = self._extract_features(
                content_img, self.content_layers_indices.values()
            )
            style_target_features = self._extract_features(
                style_img, self.style_layers_indices.values()
            )

        # --- Step 2: Build the new model by iterating through VGG19 layers ---
        for name, layer in self.vgg_features.named_children():
            # convert the layer's original name (string) back to an integer index
            layer_idx = int(name)

            # --- Layer Modifications ---
            if isinstance(layer, nn.ReLU):
                # Use a non-inplace ReLU to prevent errors during backpropagation
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                # Replace MaxPool2d with AvgPool2d for better style transfer results
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )

            # Add layers to the new model
            model.add_module(name, layer)

            # --- Add Loss Modules ---
            # Add loss module after the specifided content layer
            if layer_idx == self.content_layers_indices["conv4_2"]:
                target = content_target_features.pop(0)
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{layer_idx}", content_loss)
                content_loss_modules.append(content_loss)

            if layer_idx in self.style_layers_indices.values():
                target = style_target_features.pop(0)
                style_loss = StyleLoss(target)
                model.add_module(f"style_loss_{layer_idx}", style_loss)
                style_loss_modules.append(style_loss)

        # Truncate the model after the last loss layer to save computation
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLoss, StyleLoss)):
                model = model[: i + 1]
                break

        return model, content_loss_modules, style_loss_modules
