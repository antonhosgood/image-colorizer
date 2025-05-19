import argparse
from os import PathLike
from pathlib import Path
from typing import AnyStr, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from src.models.unet import UNet
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def preprocess_image(
    image_path: PathLike[AnyStr] | AnyStr, resize: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Loads a grayscale image from the specified path, optionally resizes it, and transforms it into a normalized tensor.

    Args:
        image_path: Path to the input image.
        resize: Optional (width, height) tuple to resize the image.

    Returns:
        torch.Tensor: A 4D tensor representing the image with shape (1, 1, H, W).
    """
    image = Image.open(image_path).convert("L")
    transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if resize is not None:
        transforms += [v2.Resize(resize)]
    transform = v2.Compose(transforms)
    return transform(image).unsqueeze(0)


def postprocess_and_save(
    output_tensor: torch.Tensor, output_path: PathLike[AnyStr] | AnyStr
) -> None:
    """
    Converts a model output tensor to an image and saves it to disk.

    Args:
        output_tensor: The tensor to convert and save (assumed shape (1, 3, H, W)).
        output_path: The path where the image will be saved.
    """
    output = output_tensor.squeeze(0).detach().cpu()
    output_image = v2.ToPILImage()(output.clamp(0, 1))
    output_image.save(output_path)


def show_side_by_side(
    grayscale_tensor: torch.Tensor, color_tensor: torch.Tensor
) -> None:
    """
    Displays the input grayscale and output colorized images side by side.

    Args:
        grayscale_tensor: Grayscale image tensor of shape (1, 1, H, W).
        color_tensor: Colorized image tensor of shape (1, 3, H, W).
    """
    gray_img = v2.ToPILImage()(grayscale_tensor.squeeze(0).cpu())
    color_img = v2.ToPILImage()(color_tensor.squeeze(0).cpu().clamp(0, 1))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(gray_img, cmap="gray")
    axs[0].set_title("Input (Grayscale)")
    axs[1].imshow(color_img)
    axs[1].set_title("Output (Colorized)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def inference(
    model: nn.Module,
    device: torch.device,
    input_path: PathLike[AnyStr] | AnyStr,
    output_path: Optional[PathLike[AnyStr] | AnyStr] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Performs inference using a model on a grayscale image, and either saves or displays the result.

    Args:
        model: The PyTorch model used for inference.
        device: The device to run the model on.
        input_path: Path to the input grayscale image.
        output_path: Path to save the colorized output image. If None, display the result.
        resize: Optional (width, height) tuple to resize the image.
    """
    input_tensor = preprocess_image(input_path, resize).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        postprocess_and_save(output, output_path)
    else:
        show_side_by_side(input_tensor, output)


def main():
    parser = argparse.ArgumentParser(
        description="Colorize a grayscale image using a trained model."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file.")
    parser.add_argument("input", type=str, help="Path to grayscale image.")
    parser.add_argument(
        "checkpoint", type=str, help="Path to trained model checkpoint (.pth)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the colorized image.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    resize = tuple(config["image_size"]) if "image_size" in config else None

    model = UNet().to(device)
    load_checkpoint(model, args.checkpoint)

    inference(
        model=model,
        device=device,
        input_path=args.input,
        output_path=args.output,
        resize=resize,
    )


if __name__ == "__main__":
    main()
