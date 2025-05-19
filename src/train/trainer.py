from typing import Any, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
) -> float:
    """
    Trains the model for one epoch on the training dataset.

    Args:
        model: The neural network to train.
        train_loader: DataLoader for the training set.
        optimizer: Optimizer used for updating model weights.
        criterion: Loss function.
        device: Device on which to perform training.
        config: Configuration dictionary with training options.

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    total_loss = 0

    for color_img, grayscale_img in tqdm(train_loader, desc="Training"):
        color_img, grayscale_img = color_img.to(device), grayscale_img.to(device)

        optimizer.zero_grad()
        output = model(grayscale_img)
        loss = criterion(output, color_img)
        loss.backward()

        if config["clip_grad"]["enabled"]:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["clip_grad"]["max_norm"]
            )

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """
    Evaluates the model on the validation dataset.

    Args:
        model: The trained model to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device on which to perform validation.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for color_img, grayscale_img in tqdm(val_loader, desc="Validation"):
            color_img, grayscale_img = color_img.to(device), grayscale_img.to(device)
            output = model(grayscale_img)
            loss = criterion(output, color_img)
            total_loss += loss.item()

    return total_loss / len(val_loader)
