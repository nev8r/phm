"""
Random seed helper.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
