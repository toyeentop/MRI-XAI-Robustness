"""
split_data.py

Participant-level stratified splitting utilities for the
MRI XAI robustness project.

Official NACC split design:
    Training:   70%
    Validation: 15%
    Test:       15%

Random seed:
    42

The official frozen NACC split contains:

    Training:
        2759 participants
        2394 Normal
        365 AD Dementia

    Validation:
        591 participants
        513 Normal
        78 AD Dementia

    Test:
        592 participants
        513 Normal
        79 AD Dementia

Important:
    Once the official split has been created and frozen,
    it should NOT be regenerated for model-performance
    optimization.

    Validation is used for model selection, early stopping,
    and final decision-threshold selection.

    The internal test cohort remains untouched until the
    model-development procedure is finalized.

    ADNI is not included in this splitting process.
    It is reserved for external validation only.
"""

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)


DEFAULT_RANDOM_STATE = 42


def create_stratified_70_15_15_split(
    cohort_df,
    id_column="NACCID",
    label_column="label",
    random_state=DEFAULT_RANDOM_STATE
):
    """
    Create participant-level stratified 70/15/15
    train/validation/test partitions.

    Parameters
    ----------
    cohort_df : pandas.DataFrame
        One row per participant.

    id_column : str
        Participant identifier column.

    label_column : str
        Binary diagnostic label column.

    random_state : int
        Fixed pseudorandom seed for reproducibility.

    Returns
    -------
    train_df : pandas.DataFrame

    validation_df : pandas.DataFrame

    test_df : pandas.DataFrame

    combined_df : pandas.DataFrame
        Full cohort with an added 'split' column.
    """

    if cohort_df[id_column].duplicated().any():

        raise ValueError(
            "Participant IDs must be unique before splitting."
        )

    if cohort_df[label_column].isna().any():

        raise ValueError(
            "Missing labels detected."
        )


    # --------------------------------------------------------
    # First split:
    # 70% training
    # 30% temporary validation/test pool
    # --------------------------------------------------------

    train_df, temporary_df = (
        train_test_split(
            cohort_df,
            test_size=0.30,
            stratify=cohort_df[label_column],
            random_state=random_state
        )
    )


    # --------------------------------------------------------
    # Second split:
    # Divide the remaining 30% equally:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    validation_df, test_df = (
        train_test_split(
            temporary_df,
            test_size=0.50,
            stratify=temporary_df[label_column],
            random_state=random_state
        )
    )


    # --------------------------------------------------------
    # Add explicit split labels
    # --------------------------------------------------------

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"

    validation_df["split"] = (
        "validation"
    )

    test_df["split"] = "test"


    # --------------------------------------------------------
    # Combine for convenient auditing
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            train_df,
            validation_df,
            test_df
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    train_ids = set(
        train_df[id_column]
    )

    validation_ids = set(
        validation_df[id_column]
    )

    test_ids = set(
        test_df[id_column]
    )

    if not train_ids.isdisjoint(
        validation_ids
    ):

        raise ValueError(
            "Training and validation participants overlap."
        )

    if not train_ids.isdisjoint(
        test_ids
    ):

        raise ValueError(
            "Training and test participants overlap."
        )

    if not validation_ids.isdisjoint(
        test_ids
    ):

        raise ValueError(
            "Validation and test participants overlap."
        )

    if len(combined_df) != len(
        cohort_df
    ):

        raise ValueError(
            "Split participant count does not match "
            "the original cohort."
        )


    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        combined_df.reset_index(drop=True)
    )


def summarize_split(
    dataframe,
    label_column="label"
):
    """
    Return participant counts by split and diagnostic label.
    """

    if "split" not in dataframe.columns:

        raise ValueError(
            "DataFrame must contain a 'split' column."
        )

    summary = (
        dataframe
        .groupby(
            [
                "split",
                label_column
            ]
        )
        .size()
        .reset_index(
            name="participants"
        )
    )

    return summary
