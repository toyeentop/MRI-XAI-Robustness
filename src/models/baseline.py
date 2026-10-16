"""
baseline.py

Simple CNN baseline for binary Normal vs. AD Dementia
classification using 2.5D MRI inputs.

Expected model input:
    [batch, 3, 224, 224]

Output:
    One logit per stack.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Lightweight CNN baseline.

    Total parameters:
        23,649
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.classifier(x)

        return x.squeeze(1)
