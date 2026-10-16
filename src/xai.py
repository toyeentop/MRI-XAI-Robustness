"""
Explainability utilities for the MRI XAI robustness project.

This module provides reusable post-hoc explanation methods for
binary MRI dementia classification.

Current implementation:
- Grad-CAM for ConvNeXt V2 Tiny

The Grad-CAM target is the depthwise convolution of the final
ConvNeXt V2 architectural block:

    model.model.stages[3].blocks[-1].conv_dw

The implementation returns a normalized 2D heatmap in [0, 1].
It does not modify the input image or model parameters.
"""

import torch
import torch.nn.functional as F


class GradCAM:
    """Generate Grad-CAM explanations for a binary classifier."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):
        """
        Capture spatial activations and attach a gradient hook
        directly to the target-layer output tensor.

        Tensor-level gradient capture is used because it is more
        reliable for the ConvNeXt V2 computation graph than a
        module-level full backward hook.
        """

        self.activations = output.detach()

        if output.requires_grad:
            output.register_hook(
                self._save_gradient
            )

    def _save_gradient(self, gradient):
        """Store gradients of the target-layer output."""

        self.gradients = gradient.detach()

    def generate(self, image):
        """
        Generate a Grad-CAM heatmap for one MRI 2.5D stack.

        Parameters
        ----------
        image : torch.Tensor
            Tensor with shape [1, 3, H, W].

        Returns
        -------
        heatmap : torch.Tensor
            Normalized heatmap with shape [H, W].

        probability : float
            Predicted AD probability from the binary classifier.
        """

        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "Grad-CAM expects one image with shape [1, 3, H, W]."
            )

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Grad-CAM requires a gradient path through the feature
        # extractor. During Stage-1 transfer learning the ConvNeXt V2
        # backbone is frozen, so its activations do not require
        # gradients when the input also has requires_grad=False.
        #
        # Create an explanation-only copy with input gradients enabled.
        # This does not unfreeze, update, or retrain model parameters.
        image = image.detach().clone().requires_grad_(True)

        logits = self.model(image)

        if logits.numel() != 1:
            raise ValueError(
                "Grad-CAM currently expects a single binary logit."
            )

        score = logits.reshape(-1)[0]
        probability = torch.sigmoid(score).item()

        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach(), probability

    def remove_hooks(self):
        """Remove registered PyTorch hooks."""

        self.forward_handle.remove()


def get_convnextv2_gradcam_target(model):
    """Return the validated Grad-CAM target layer for ConvNeXt V2 Tiny."""

    return model.model.stages[3].blocks[-1].conv_dw


# ============================================================
# SHAP
# ============================================================

class SHAPModelWrapper(torch.nn.Module):
    """
    Wrap a binary PyTorch model so that its output has shape
    [batch_size, 1], as expected by SHAP GradientExplainer.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        logits = self.model(x)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        return logits


class SHAPExplainer:
    """
    Gradient-based SHAP explanation for the binary
    ConvNeXt V2 MRI classifier.

    The background tensor should contain samples drawn only
    from the training data.

    Raw SHAP values are preserved for the three MRI channels.
    A 2D magnitude map is also produced by averaging the
    absolute attribution magnitude across the three channels.
    """

    def __init__(
        self,
        model,
        background,
    ):

        import shap

        self.model = SHAPModelWrapper(model)
        self.model.eval()

        self.background = background.detach()

        self.explainer = shap.GradientExplainer(
            self.model,
            self.background,
        )


    def generate(
        self,
        image,
        nsamples=20,
    ):
        """
        Generate SHAP attribution values for one MRI stack.

        Parameters
        ----------
        image : torch.Tensor
            Shape [1, 3, H, W].

        nsamples : int
            Number of samples used by GradientExplainer.

        Returns
        -------
        raw_values : torch.Tensor
            Raw channel-wise SHAP values with shape [3, H, W].

        magnitude_map : torch.Tensor
            Normalized 2D attribution-magnitude map [H, W].

        probability : float
            Predicted AD probability.
        """

        import numpy as np

        self.model.eval()

        with torch.no_grad():

            logits = self.model(image)

            probability = torch.sigmoid(
                logits
            ).item()


        shap_values = self.explainer.shap_values(
            image,
            nsamples=nsamples,
        )


        values = np.asarray(
            shap_values
        )


        # SHAP 0.52 returns:
        # [batch, channel, height, width, output]
        if values.ndim == 5:

            values = values[
                0, :, :, :, 0
            ]

        elif values.ndim == 4:

            values = values[0]

        else:

            raise RuntimeError(
                "Unexpected SHAP output shape: "
                f"{values.shape}"
            )


        raw_values = torch.from_numpy(
            values
        ).float()


        # Aggregate the absolute attribution magnitude
        # across the three adjacent MRI slices.
        magnitude_map = (
            raw_values.abs().mean(dim=0)
        )


        # Normalize only for visualization/comparison.
        map_min = magnitude_map.min()
        map_max = magnitude_map.max()

        if map_max > map_min:

            magnitude_map = (
                magnitude_map - map_min
            ) / (
                map_max - map_min
            )

        else:

            magnitude_map = torch.zeros_like(
                magnitude_map
            )


        return (
            raw_values,
            magnitude_map,
            probability,
        )

