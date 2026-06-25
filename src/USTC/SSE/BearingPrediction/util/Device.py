"""
Device utility module

Purpose: provide utility helpers used by the bearing PHM framework
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import torch


def select_torch_device() -> torch.device:
    """
    Select the best available PyTorch device.
    Preference order: CUDA, Apple MPS, CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
