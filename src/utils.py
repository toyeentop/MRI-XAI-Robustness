"""
utils.py

General reproducibility utilities for the
MRI XAI robustness project.

The project uses random seed 42 for reproducible
data splitting and model-development experiments.

The number 42 has no scientific significance.
It is simply a fixed pseudorandom starting state.
"""

import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42


def set_random_seed(
    seed=DEFAULT_SEED
):
    """
    Set common Python, NumPy, and PyTorch random seeds.

    Parameters
    ----------
    seed : int
        Fixed pseudorandom seed.

    Returns
    -------
    int
        Seed that was applied.
    """

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    return seed


def get_device():
    """
    Return the preferred PyTorch computation device.

    CUDA is used when available; otherwise CPU is used.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def count_parameters(
    model
):
    """
    Return total and trainable parameter counts.

    Returns
    -------
    dict
        total_parameters
        trainable_parameters
        frozen_parameters
        trainable_percentage
    """

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    frozen = (
        total - trainable
    )

    percentage = (
        (trainable / total) * 100
        if total > 0
        else 0.0
    )

    return {
        "total_parameters":
            int(total),

        "trainable_parameters":
            int(trainable),

        "frozen_parameters":
            int(frozen),

        "trainable_percentage":
            float(percentage),
    }
