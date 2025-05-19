import argparse
from os import PathLike
from typing import AnyStr

import torch
from torch import nn, optim
from torchvision.transforms import v2

from src.data.stock_image_dataset import StockImageDataset
from src.data.utils import create_dataloaders
from src.models.unet import UNet
from src.train.trainer import train_one_epoch, validate
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def train(config_path: PathLike[AnyStr] | AnyStr) -> None:
    """Trains an image-to-image model using parameters defined in an external file."""
    config = load_config(config_path)

    device = get_device()
    print(f"Using device: {device}")

    torch.manual_seed(config["seed"])

    transform = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])

    dataset = StockImageDataset(
        color_dir=config["color_dir"],
        grayscale_dir=config["grayscale_dir"],
        transform=transform,
    )

    train_loader, val_loader = create_dataloaders(
        dataset, config["val_ratio"], config["batch_size"]
    )

    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )

    for epoch in range(config["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if (epoch + 1) % config["checkpoint"]["interval"] == 0:
            save_checkpoint(model, config["checkpoint"]["dir"], epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    train(args.config)
