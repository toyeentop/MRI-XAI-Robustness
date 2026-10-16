"""
main_model.py

ConvNeXt V2 Tiny main model for the MRI XAI robustness project.

Exact pretrained checkpoint:
    convnextv2_tiny.fcmae_ft_in22k_in1k

Expected model input:
    [batch, 3, 224, 224]

Output:
    One binary-classification logit per stack.
"""

import torch.nn as nn
import timm


MAIN_MODEL_NAME = (
    "convnextv2_tiny.fcmae_ft_in22k_in1k"
)


class BinaryModelWrapper(nn.Module):
    """
    Ensures a binary timm model returns logits with shape:

        [batch]

    rather than:

        [batch, 1]
    """

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):

        logits = self.model(x)

        return logits.squeeze(1)


def create_convnextv2_tiny(
    pretrained=True
):
    """
    Create ConvNeXt V2 Tiny for binary classification.

    Parameters
    ----------
    pretrained : bool
        If True, load the pretrained timm checkpoint.

    Returns
    -------
    BinaryModelWrapper
        ConvNeXt V2 Tiny with a one-output classification head.
    """

    model = timm.create_model(
        MAIN_MODEL_NAME,
        pretrained=pretrained,
        num_classes=1
    )

    return BinaryModelWrapper(
        model
    )


def configure_stage1(model):
    """
    Stage 1 transfer learning:

    Freeze the entire ConvNeXt V2 backbone and train only
    the classification head.
    """

    for parameter in model.model.parameters():
        parameter.requires_grad = False

    for parameter in model.model.head.parameters():
        parameter.requires_grad = True

    return model


def configure_stage2(model):
    """
    Stage 2 controlled fine-tuning:

    Keep ConvNeXt V2 architectural stages 0-2 frozen.

    Unfreeze:
        - architectural stage 3
        - classification head
    """

    for parameter in model.model.parameters():
        parameter.requires_grad = False

    for parameter in (
        model.model.stages[3].parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.model.head.parameters()
    ):
        parameter.requires_grad = True

    return model
