import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import StyleTransfer


def load_image_tensor(
    image_path: str | Path, size: int, device: torch.device
) -> torch.Tensor:
    """Load an image from a file path and convert it to a tensor."""
    image = Image.open(image_path)

    # ToTensor() : normalize to [0, 1] and convert to tensor (h, w, c) -> (c, h, w)
    # add: Normalize with ImageNet's mean and standard deviation (https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    # add batch dimension : (c, h, w) -> (1, c, h, w)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float32)  # move to device and convert to float32


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Denormalize a tensor using global constants and convert it to a PIL image."""

    # Ensure tensor is 4D with batch size 1 before squeezing
    if tensor.dim() == 4 and tensor.size(0) != 1:
        raise ValueError(
            "Expected tensor with batch size 1, got size {}".format(tensor.size(0))
        )

    # remove batch dimension : (1, c, h, w) -> (c, h, w) and move to CPU for processing
    tensor = tensor.squeeze(0).cpu().clone()

    # --- Denormalization part --
    # Reshape constants for broadcasting: (C) -> (C, 1, 1) for broadcasting to (C, H, W)
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)

    # Apply the inverse transformation: (tensor * std) + mean
    tensor = tensor * std + mean

    # ensure pixel values are in [0, 1] : tensor.clamp(min, max)
    tensor = tensor.clamp(0, 1)
    # (c, h, w ) -> RGB (0-255)
    image = transforms.ToPILImage()(tensor)
    return image


if __name__ == "__main__":
    # --- 1. Setup ---

    # Use CUDA if available, otherwise fall back to MPS for Apple Silicon or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    DATA_PATH = Path("data")
    CONTENT_IMAGE_PATH = DATA_PATH / "content.jpg"
    STYLE_IMAGE_PATH = DATA_PATH / "style.jpg"

    # ImageNet statistics for normalization, defined once as a single source of truth.
    # These will be used for both normalizing input images and denormalizing the output image.
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    # load original images
    content_image = load_image_tensor(CONTENT_IMAGE_PATH, 256, device)
    style_image = load_image_tensor(STYLE_IMAGE_PATH, 256, device)

    # --- 2. Build the Model ---
    style_transfer = StyleTransfer(device)
    # model : nn.Sequential, content_loss_modules: list[ContentLoss], style_loss_modules: list[StyleLoss]
    model, content_loss_modules, style_loss_modules = style_transfer.build_model(
        content_image, style_image
    )
    # The image to be optimized is initialized from random noise.
    # It has the same size and is on the same device as the content image.
    generated_image = torch.randn(content_image.size(), device=device).requires_grad_(
        True
    )

    # --- 3. Training Loop ---
    # Optimizer targets the generated image itself, not the model weights.
    optimizer = torch.optim.Adam([generated_image], lr=0.01)

    # hyperparameters
    style_weight = 1000
    content_weight = 1
    epochs = 3000

    print("Starting style transfer training...")
    for epoch in range(epochs):

        def closure():
            optimizer.zero_grad()  # reset gradients

            # pass the generated image through the model to compute the losses internally
            model(generated_image)

            # sum up the losses from the loss modules
            style_loss = sum(style_module.loss for style_module in style_loss_modules)
            content_loss = sum(
                content_module.loss for content_module in content_loss_modules
            )

            # calculate the total loss
            total_loss = style_weight * style_loss + content_weight * content_loss
            # backpropagation
            total_loss.backward()

            if (epoch + 1) % 100 == 0:  # print progress every 100 epochs
                # loss 의 데이터 구조: 예 - tensor(1502.7270, device='cuda:0') -> .item() 으로 순수한 값 1502.7270 (float 또는 int)만
                print(
                    f"Epoch {epoch +1} / {epochs} : Style Loss: {style_loss.item():.4f}, Content Loss: {content_loss.item():.4f}"
                )

        optimizer.step(closure)

    # --- 4. Save the Generated Image ---
    OUTPUT_PATH = Path("result")
    OUTPUT_IMAGE_PATH = OUTPUT_PATH / "output.png"

    generated_image = generated_image.detach()
    generated_image = tensor_to_image(generated_image)
    generated_image.save(OUTPUT_IMAGE_PATH)
    print("Generated image saved to generated_image.png")
