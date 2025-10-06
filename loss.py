# content loss : vgg19 feature map
# style loss : gram matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import StyleTransfer


class ContentLoss(nn.Module):
    """
    ContentLoss computes the mean squared error between the feature maps of the
    generated image and the original content image. This loss encourages the
    generated image to retain the "content" of the original image.
    """

    def __init__(
        self,
    ):
        super(ContentLoss, self).__init__()
        self.loss = torch.tensor(0.0)  # Initialize loss as a tensor

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the content loss.
        Args:
            generated (torch.Tensor): The feature map of the generated image.
            target (torch.Tensor): The feature map of the content image.
        Returns:
            generated: The 'generated' tensor is passed through unmodified.
            The calculated loss is stored in self.loss.
        """
        # 1. Compute the mean squared error between the feature maps of the generated image and the target image.
        self.loss = F.mse_loss(generated, target)

        # 2. Return the original 'generated' tensor to allow chaining in a nn.Sequential model.
        return generated


class StyleLoss(nn.Module):
    """
    StyleLoss computes the mean squared error between the Gram matrices of the
    feature maps from the generated image and the style image. This encourages
    the generated image to adopt the textural and stylistic patterns of the
    style image.
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        # We store the target Gram matrix to avoid recomputing it on every forward pass.
        # It's detached from the computation graph as it's a fixed target and doesn't require gradients.
        self.target = None
        self.loss = torch.tensor(0.0)  # Initialize loss as a tensor

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gram matrix for a given batch of feature maps.
        The Gram matrix captures the correlations between different filter responses,
        representing the texture and style of an image layer.

        Args:
            x (torch.Tensor): The feature map tensor.
        Returns:
            torch.Tensor: The Gram matrix tensor.
        """
        # Assuming 'x' is a tensor from a convolutional layer
        # a=batch size, b=num_feature_maps, (c,d)=dimensions of a feature map
        a, b, c, d = x.size()

        # Reshape the tensor to combine the spatial dimensions (height and width)
        # into one dimension, while keeping batch and channel dimensions separate.
        # Reshape to (batch_size, num_feature_maps, N) where N = c*d
        features = x.view(a, b, c * d)

        # Use batch matrix multiplication (bmm) to compute the Gram matrix for each image in the batch.
        # features: (a, b, c*d), features.transpose(1, 2): (a, c*d, b)
        # The result 'G' will have a shape of (a, b, b)
        G = torch.bmm(features, features.transpose(1, 2))

        # Normalize the Gram matrix by dividing by the total number of elements
        # in each feature map. This makes the loss independent of image size.
        return G.div(b * c * d)

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the style loss. In style transfer, this module is often used as a
        transparent layer that computes the loss and passes the input through.

        Args:
            generated (torch.Tensor): The feature map of the generated image.
            target (torch.Tensor): The feature map of the style image.

        Returns:
            torch.Tensor: The 'generated' tensor is passed through unmodified.
                          The calculated loss is stored in self.loss.
        """
        # The Gram matrix of the target (style) image only needs to be computed once.
        # We compute and store it on the first forward pass.
        if self.target is None:
            self.target = self.gram_matrix(target).detach()

        # Compute the Gram matrix for the generated image on every pass.
        G = self.gram_matrix(generated)

        # Calculate the mean squared error between the two Gram matrices.
        self.loss = F.mse_loss(G, self.target)

        # Return the original 'generated' tensor to allow chaining in a nn.Sequential model.
        return generated
