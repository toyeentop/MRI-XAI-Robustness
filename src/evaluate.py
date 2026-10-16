"""
evaluate.py

Reusable evaluation utilities for the MRI XAI robustness
project.

Each participant contributes five overlapping 2.5D MRI stacks.

The model produces one logit for each stack.

Stack-level logits are converted to probabilities using the
sigmoid function. The five probabilities belonging to the
same participant are then averaged to produce one
participant-level AD probability.

Participant-level probability:

    mean(stack_probability_1, ..., stack_probability_5)

The participant, rather than the individual stack, is the
primary unit of evaluation.
"""

import pandas as pd
import torch


def collect_participant_predictions(
    model,
    data_loader,
    device
):
    """
    Generate stack-level predictions and aggregate them to
    participant-level probabilities.

    Parameters
    ----------
    model : torch.nn.Module
        Binary classification model returning one logit
        per stack.

    data_loader : torch.utils.data.DataLoader
        DataLoader created from NACC25DDataset or another
        compatible dataset.

    device : torch.device
        CPU or GPU device used for inference.

    Returns
    -------
    stack_df : pandas.DataFrame
        One row per 2.5D stack.

    participant_df : pandas.DataFrame
        One row per participant. The AD probability is the
        arithmetic mean of that participant's stack
        probabilities.
    """

    model.eval()

    stack_records = []

    with torch.no_grad():

        for batch in data_loader:

            images = batch["image"].to(
                device
            )

            labels = batch["label"]

            participant_ids = (
                batch["participant_id"]
            )

            stack_numbers = (
                batch["stack_number"]
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).detach().cpu()

            labels = labels.detach().cpu()

            if torch.is_tensor(stack_numbers):
                stack_numbers = (
                    stack_numbers
                    .detach()
                    .cpu()
                    .tolist()
                )

            for i in range(
                len(participant_ids)
            ):

                stack_records.append({
                    "participant_id":
                        participant_ids[i],

                    "stack_number":
                        int(stack_numbers[i]),

                    "true_label":
                        int(labels[i].item()),

                    "ad_probability":
                        float(
                            probabilities[i].item()
                        ),
                })

    stack_df = pd.DataFrame(
        stack_records
    )

    # --------------------------------------------------------
    # Participant-level aggregation
    #
    # Each participant should contribute five stack
    # probabilities. Their arithmetic mean becomes the
    # participant-level AD probability.
    # --------------------------------------------------------

    participant_df = (
        stack_df
        .groupby(
            "participant_id",
            as_index=False
        )
        .agg(
            true_label=(
                "true_label",
                "first"
            ),

            ad_probability=(
                "ad_probability",
                "mean"
            ),

            number_of_stacks=(
                "stack_number",
                "count"
            ),
        )
    )

    return (
        stack_df,
        participant_df
    )
