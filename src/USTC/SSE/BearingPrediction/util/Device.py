"""
Device utility module

this file is for selecting available PyTorch execution devices

created by zy

copyright USTC

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
