from os import PathLike
from pathlib import Path
from typing import AnyStr

import torch
from torch import nn

from src.utils.device import get_device


def save_checkpoint(
    model: nn.Module, checkpoint_dir: PathLike[AnyStr] | AnyStr, epoch: int
):
    """Saves the model's state dictionary to a checkpoint file."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: PathLike[AnyStr] | AnyStr,
    device: torch.device = get_device(),
):
    """Loads the model's weights from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}")
