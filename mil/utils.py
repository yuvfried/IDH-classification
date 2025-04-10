import os
import random
from typing import Union

import numpy as np
import torch


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def min_max(arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Agnostic to Pytorch or Numpy arrays.
    Note: Does not handle NaN values.
    """
    return (arr - arr.min()) / (arr.max() - arr.min())