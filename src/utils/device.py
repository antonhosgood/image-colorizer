import torch


def get_device() -> torch.device:
    """Returns the most suitable available device ('cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
