# src/utils/seed.py
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Base seed value.
    deterministic : bool, optional
        If True, enable deterministic mode in PyTorch (may slow down training).
    """
    # Python's built-in RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch CPU & CUDA RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Some additional env flags for other libs that might use PYTHONHASHSEED
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # This may reduce performance, but improves reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # default (potentially faster) behavior
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
