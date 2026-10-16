"""
metrics.py

Participant-level binary classification metrics for the
MRI XAI robustness project.

Positive class:
    AD Dementia = 1

Negative class:
    Normal = 0

Final study threshold:
    Must be selected using the frozen NACC validation cohort
    and then held fixed for NACC test and ADNI external
    validation.

A threshold of 0.5 may be used for development diagnostics
but is not automatically the final study threshold.
"""

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    matthews_corrcoef,
)


def calculate_participant_metrics(
    participant_df,
    threshold=0.5
):
    """
    Calculate participant-level binary classification metrics.

    Expected participant_df columns:
        true_label
        ad_probability

    Parameters
    ----------
    participant_df : pandas.DataFrame
        One row per participant.

    threshold : float
        Probability threshold used to convert AD probabilities
        into binary predictions.

    Returns
    -------
    dict
        Participant-level evaluation metrics.
    """

    y_true = (
        participant_df["true_label"]
        .astype(int)
        .to_numpy()
    )

    y_prob = (
        participant_df["ad_probability"]
        .astype(float)
        .to_numpy()
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Confusion matrix
    # Force labels [0, 1] so output is always 2 x 2
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Threshold-independent metrics
    # --------------------------------------------------------

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

        average_precision = (
            average_precision_score(
                y_true,
                y_prob
            )
        )

    else:

        roc_auc = np.nan
        average_precision = np.nan

    # --------------------------------------------------------
    # Threshold-dependent metrics
    # --------------------------------------------------------

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    return {
        "threshold": float(threshold),

        "roc_auc": float(roc_auc)
        if not np.isnan(roc_auc)
        else np.nan,

        "average_precision":
            float(average_precision)
            if not np.isnan(
                average_precision
            )
            else np.nan,

        "balanced_accuracy":
            float(balanced_accuracy),

        "sensitivity":
            float(sensitivity)
            if not np.isnan(sensitivity)
            else np.nan,

        "specificity":
            float(specificity)
            if not np.isnan(specificity)
            else np.nan,

        "precision":
            float(precision),

        "f1":
            float(f1),

        "mcc":
            float(mcc),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
