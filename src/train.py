"""
train.py

Reusable training utilities for binary MRI classification.

The same training framework can be used for:
    - Baseline CNN
    - ConvNeXt V2 Tiny

The loss function is supplied by the calling script so that
class weighting can be configured explicitly.

For the NACC training cohort, the positive-class weight is
derived from the frozen training partition:

    Normal = 2394
    AD Dementia = 365

    pos_weight = 2394 / 365
               = 6.5589

Checkpoint selection is based on validation loss rather than
development-set classification metrics.

This is particularly important when validation subsets are
small or imbalanced.
"""

from pathlib import Path
import time

import pandas as pd
import torch


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device
):
    """
    Train a binary classification model for one epoch.

    Returns
    -------
    mean_loss : float
        Sample-weighted mean training loss.

    elapsed_seconds : float
        Training time for the epoch.
    """

    model.train()

    start_time = time.time()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:

        images = batch["image"].to(
            device
        )

        labels = (
            batch["label"]
            .float()
            .to(device)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    mean_loss = (
        total_loss
        / total_samples
    )

    elapsed_seconds = (
        time.time()
        - start_time
    )

    return (
        mean_loss,
        elapsed_seconds
    )


def evaluate_validation_loss(
    model,
    data_loader,
    criterion,
    device
):
    """
    Calculate sample-weighted validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in data_loader:

            images = batch["image"].to(
                device
            )

            labels = (
                batch["label"]
                .float()
                .to(device)
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


def train_binary_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    checkpoint_path,
    log_path,
    max_epochs=20,
    patience=5
):
    """
    Train a binary MRI classification model with validation
    loss checkpointing and early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.

    train_loader : DataLoader
        Training data.

    validation_loader : DataLoader
        Validation data.

    criterion : torch loss
        Binary classification loss.

    optimizer : torch optimizer
        Model optimizer.

    device : torch.device
        CPU or GPU.

    checkpoint_path : str or pathlib.Path
        Location for the best checkpoint.

    log_path : str or pathlib.Path
        Location for the epoch-level CSV log.

    max_epochs : int
        Maximum number of epochs.

    patience : int
        Number of consecutive non-improving epochs allowed
        before early stopping.

    Returns
    -------
    history_df : pandas.DataFrame
        Epoch-level training history.

    best_epoch : int
        Epoch with the lowest validation loss.

    best_validation_loss : float
        Lowest validation loss observed.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    log_path = Path(
        log_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_validation_loss = float(
        "inf"
    )

    best_epoch = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        max_epochs + 1
    ):

        train_loss, train_seconds = (
            train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
        )

        validation_loss = (
            evaluate_validation_loss(
                model=model,
                data_loader=validation_loader,
                criterion=criterion,
                device=device
            )
        )

        improved = (
            validation_loss
            < best_validation_loss
        )

        if improved:

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "train_loss":
                        train_loss,

                    "validation_loss":
                        validation_loss,
                },
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "train_seconds":
                    train_seconds,

                "improved":
                    improved,
            }
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            log_path,
            index=False
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f} | "
            f"Best: {best_validation_loss:.6f}"
        )

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                f"Early stopping after "
                f"{patience} consecutive "
                f"non-improving epochs."
            )

            break

    history_df = pd.DataFrame(
        history
    )

    return (
        history_df,
        best_epoch,
        best_validation_loss
    )
