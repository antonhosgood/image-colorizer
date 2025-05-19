from os import PathLike
from pathlib import Path
from typing import AnyStr, Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class StockImageDataset(Dataset):
    """
    A Dataset for loading pairs of color and grayscale images with matching filenames from two directories.

    Each item is a tuple of (color_image_tensor, grayscale_image_tensor), optionally transformed using a provided
    transform.
    """

    def __init__(
        self,
        color_dir: PathLike[AnyStr] | AnyStr,
        grayscale_dir: PathLike[AnyStr] | AnyStr,
        transform: Optional[Callable] = None,
    ) -> None:
        self.color_dir = Path(color_dir)
        self.grayscale_dir = Path(grayscale_dir)

        self.filenames = sorted(
            [
                f.name
                for f in self.color_dir.iterdir()
                if f.is_file() and (self.grayscale_dir / f.name).is_file()
            ]
        )

        self.base_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        color_path = self.color_dir / filename
        grayscale_path = self.grayscale_dir / filename

        color_img = Image.open(color_path).convert("RGB")
        gray_img = Image.open(grayscale_path).convert("L")

        color_tensor = self.base_transform(color_img)
        gray_tensor = self.base_transform(gray_img)

        if self.transform:
            color_tensor = self.transform(color_tensor)
            gray_tensor = self.transform(gray_tensor)

        return color_tensor, gray_tensor
