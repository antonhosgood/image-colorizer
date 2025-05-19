from copy import deepcopy
from typing import Callable, Iterable, Tuple

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


def augment_dataset(dataset: Dataset, transforms: Iterable[Callable]) -> Dataset:
    """
    Creates a new dataset that includes the original dataset plus augmented versions using the provided transforms.

    Assumes that the dataset uses an attribute `transform` to apply the transformations.

    Args:
        dataset: Original dataset.
        transforms: List of transforms to apply.

    Returns:
        Dataset: A ConcatDataset containing the original and augmented datasets.
    """
    datasets = [dataset]

    for transform in transforms:
        augmented = deepcopy(dataset)
        augmented.transform = transform
        datasets.append(augmented)

    return ConcatDataset(datasets)


def create_dataloaders(
    dataset: Dataset, val_ratio: float, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a dataset into training and validation sets and returns their respective DataLoaders.

    Args:
        dataset: The full dataset to be split.
        val_ratio: The proportion of the dataset to use for validation (between 0 and 1).
        batch_size: Number of samples per batch to load.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and validation DataLoader.
    """

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
