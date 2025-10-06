"""Neural style transfer feature extractor using pretrained VGG19.

This module exposes a `StyleTransfer` wrapper around VGG19 that returns
feature maps from canonical style and content layers, following Gatys et al.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StyleTransfer(nn.Module):
    """Wrapper around VGG19 to extract style/content feature maps.

    This class loads a pretrained VGG19 and provides indices for:
    - style: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    - content: conv4_2

    Use `forward(x, mode)` to collect the corresponding feature maps.
    """

    def __init__(
        self,
    ):
        super(StyleTransfer, self).__init__()
        # to do : load vgg19 pre train model
        self.vgg19 = models.vgg19(pretrained=True)
        # print(self.vgg19)

        # to do : conv layer seperate
        # Style: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, Content: conv4_2 (see the results section in the paper)
        self.layers = self.vgg19.features

        CONV = {
            "conv1_1": 0,
            "conv2_1": 5,
            "conv3_1": 10,
            "conv4_1": 19,
            "conv5_1": 28,
            "conv4_2": 21,
        }
        self.style_layers = [
            CONV[layer]
            for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        ]
        self.content_layers = [CONV["conv4_2"]]

    def forward(self, x, mode="style"):
        """Extract feature maps from VGG19 for style or content loss.

        Args:
            x (torch.Tensor): Input tensor shaped (N, 3, H, W).
            mode (str): "style" to collect style layers, "content" for content layers.

        Returns:
            list[torch.Tensor]: Collected feature maps, in forward order.

        Raises:
            NotImplementedError: If `mode` is neither "style" nor "content".
        """
        # extract feature maps for style and content
        features = []

        if mode.lower() not in ["style", "content"]:
            raise NotImplementedError("Invalid mode")

        for i, layer in enumerate(self.layers):
            x = layer(x)  # pass the input through the layer
            if mode.lower() == "style" and i in self.style_layers:
                features.append(x)
            elif mode.lower() == "content" and i in self.content_layers:
                features.append(x)
        return features


if __name__ == "__main__":
    model = StyleTransfer()
    x = torch.randn(1, 3, 256, 256)  # (batch_size, channel, height, width)

    # We call `model(x, ...)` instead of `model.forward(x, ...)`.
    # This is the standard PyTorch practice because it uses the `nn.Module.__call__`
    # method, which handles important background tasks (hooks) before calling
    # our defined `.forward()` method.
    y = model(x, mode="style")
    print(y[4].shape)
