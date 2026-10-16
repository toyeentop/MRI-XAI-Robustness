"""
dataset.py

PyTorch dataset utilities for the MRI XAI robustness project.

Canonical derived MRI input:
    182 x 182 x 3

Each participant contributes five overlapping 2.5D stacks.

Model input:
    3 x 224 x 224

The resize from 182 x 182 to 224 x 224 is performed dynamically
during loading. Resized copies are not saved to disk.

No ImageNet RGB normalization is applied because the three channels
represent adjacent MRI slices rather than RGB channels. The MRI data
have already undergone participant-level brain-mask z-score
normalization during preprocessing.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NACC25DDataset(Dataset):
    """
    Participant-stack dataset for NACC 2.5D MRI classification.

    Expected stack filenames:
        <NACCID>_stack1.npy
        ...
        <NACCID>_stack5.npy

    Expected canonical stack shape:
        (182, 182, 3)

    Labels:
        Normal       -> 0
        AD_Dementia  -> 1
    """

    LABEL_MAP = {
        "Normal": 0,
        "AD_Dementia": 1,
    }

    def __init__(
        self,
        manifest_df,
        stack_root,
        output_size=(224, 224),
    ):
        self.manifest_df = (
            manifest_df
            .reset_index(drop=True)
            .copy()
        )

        self.stack_root = Path(stack_root)

        self.output_size = output_size

        self.samples = []

        # ----------------------------------------------------
        # Create one dataset sample per participant x stack
        # ----------------------------------------------------

        for _, row in self.manifest_df.iterrows():

            participant_id = str(
                row["NACCID"]
            )

            label_name = str(
                row["label"]
            )

            if label_name not in self.LABEL_MAP:
                raise ValueError(
                    f"Unknown label '{label_name}' "
                    f"for participant {participant_id}"
                )

            label = self.LABEL_MAP[
                label_name
            ]

            participant_folder = (
                self.stack_root
                / participant_id
            )

            for stack_number in range(
                1,
                6
            ):

                stack_path = (
                    participant_folder
                    / (
                        f"{participant_id}"
                        f"_stack{stack_number}.npy"
                    )
                )

                self.samples.append(
                    {
                        "participant_id":
                            participant_id,

                        "stack_number":
                            stack_number,

                        "stack_path":
                            stack_path,

                        "label":
                            label,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        stack_path = sample[
            "stack_path"
        ]

        if not stack_path.exists():
            raise FileNotFoundError(
                f"Missing stack: {stack_path}"
            )

        # ----------------------------------------------------
        # Load canonical 182 x 182 x 3 stack
        # ----------------------------------------------------

        stack = np.load(
            stack_path
        ).astype(
            np.float32,
            copy=False
        )

        if stack.shape != (
            182,
            182,
            3
        ):
            raise ValueError(
                f"Unexpected stack shape "
                f"{stack.shape} for {stack_path}. "
                f"Expected (182, 182, 3)."
            )

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image = torch.from_numpy(
            stack
        ).permute(
            2,
            0,
            1
        )

        # ----------------------------------------------------
        # Dynamic bicubic resize:
        # 3 x 182 x 182 -> 3 x 224 x 224
        # ----------------------------------------------------

        image = F.interpolate(
            image.unsqueeze(0),
            size=self.output_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        image = image.float()

        return {
            "image": image,

            "label": torch.tensor(
                sample["label"],
                dtype=torch.long,
            ),

            "participant_id":
                sample["participant_id"],

            "stack_number":
                sample["stack_number"],
        }
