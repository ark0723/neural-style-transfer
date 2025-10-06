import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image


def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # normalize to [0,1] & (H,W,C) -> (C,H,W)
        ]
    )
    image = transform(image).unsqueeze(0)  # add batch dimension : (c,H,W) -> (1,C,H,W)
    return image.to(
        device, torch.float
    )  # move to device and set the dtype to float to ensure compatibility with model weights.


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for GPU Acceleration")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")
