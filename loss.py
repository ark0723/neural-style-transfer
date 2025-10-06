# content loss : vgg19 feature map
# style loss : gram matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import StyleTransfer


class ContentLoss(nn.Module):
    def __init__(
        self,
    ):
        super(ContentLoss, self).__init__()

    def forward(self, generated, target):
        return F.mse_loss(generated, target)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        # detach the target gram matrix from the computation graph
        self.target = None
        self.loss = 0

    def gram_matrix(self, x):
        # Assuming 'x' is a tensor from a convolutional layer
        # a=batch size, b=num_feature_maps, (c,d)=dimensions of a feature map
        a, b, c, d = x.size()
        features = x.view(
            a, b, c * d
        )  # Reshape to (batch_size, num_feature_maps, N) where N = c*d

        # bmm : batch matrix multiplication
        # features: (a, b, c*d), features.transpose(1, 2): (a, c*d, b)
        G = torch.bmm(features, features.transpose(1, 2))  # Gram matrix : (a, b, b)
        # Normalize by the number of elements in the feature map
        return G.div(b * c * d)

    def forward(self, generated, target):
        # target_gram은 한 번만 계산하고 재사용하는 것이 효율적입니다.
        if self.target is None:
            self.target = self.gram_matrix(target).detach()
        # gram matrix
        G = self.gram_matrix(generated)
        self.loss = F.mse_loss(G, self.target)
        return generated
