# Robustness of Explainable Deep Learning for MRI-Based Dementia Classification Under Natural and Adversarial Perturbations

## Project Overview

This project investigates the robustness of post-hoc explainable artificial
intelligence (XAI) methods for MRI-based dementia classification.

The primary objective is not simply to maximize classification accuracy.
Instead, the study examines whether model explanations remain stable when
small natural or adversarial perturbations are introduced into MRI inputs,
particularly when the model's predicted class remains unchanged.

The project uses a reproducible 2.5D MRI pipeline, a simple convolutional
neural network (CNN) baseline, and ConvNeXt V2 Tiny as the main deep learning
architecture.

Grad-CAM and SHAP are implemented as the primary post-hoc explanation methods.

---

## Research Questions

### RQ1 — Natural Perturbations

How stable are post-hoc explanations for MRI-based dementia classification
under natural perturbations when the predicted class remains unchanged?

Natural perturbations include controlled image modifications such as noise,
blur, and small rotations.

### RQ2 — Adversarial Perturbations

How stable are post-hoc explanations under adversarial perturbations when
the predicted class remains unchanged?

Planned adversarial methods include FGSM and PGD.

### RQ3 — Explanation Method Comparison

Does explanation robustness differ between Grad-CAM and SHAP under natural
and adversarial perturbations?

---

## Data Sources

### NACC

The National Alzheimer's Coordinating Center (NACC) dataset is used for
model development and internal evaluation.

The finalized NACC cohort contains 3,942 participants:

- 3,420 cognitively normal participants
- 522 participants with Alzheimer's disease dementia

A fixed participant-level stratified 70/15/15 split with random seed 42 is used:

| Partition | Normal | AD Dementia | Total |
| --- | ---: | ---: | ---: |
| Training | 2,394 | 365 | 2,759 |
| Validation | 513 | 78 | 591 |
| Test | 513 | 79 | 592 |
| **Total** | **3,420** | **522** | **3,942** |

The split is performed at the participant level so that MRI stacks from the
same participant cannot appear in different partitions.

The natural class distribution is retained rather than downsampling the normal
group. Class imbalance is handled during training using a positive-class weight
derived from the frozen NACC training partition.

### ADNI

The Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort is reserved
for external validation only.

The finalized ADNI external-validation cohort contains 993 participants:

- 862 cognitively normal participants
- 131 participants with Alzheimer's disease

ADNI data are not used for model training, hyperparameter selection, threshold
selection, or internal model development.

---

## MRI Input Design

Each participant contributes one selected 3D structural T1-weighted MRI acquisition.

The project uses a 2.5D representation. Following standardized preprocessing,
an atlas-guided hippocampal/medial-temporal coronal region is represented by
11 consecutive coronal slices (107–117).

These slices form five overlapping three-slice stacks:

```text
Stack 1: 107, 108, 109
Stack 2: 109, 110, 111
Stack 3: 111, 112, 113
Stack 4: 113, 114, 115
Stack 5: 115, 116, 117
```

Each stack is treated as a three-channel model input.

Canonical saved stack size:

```text
182 × 182 × 3
```

At model-loading time, stacks are dynamically resized using bicubic interpolation to:

```text
3 × 224 × 224
```

The five stack-level probabilities are averaged to obtain the participant-level
dementia probability.

---

## MRI Preprocessing Pipeline

The implemented preprocessing workflow is:

```text
DICOM
  ↓
dcm2niix conversion
  ↓
NIfTI
  ↓
N4 bias-field correction
  ↓
HD-BET brain extraction
  ↓
Affine registration to MNI152NLin6Asym 1 mm template
  ↓
Nearest-neighbor transformation of the brain mask
  ↓
Participant-level brain-mask z-score normalization
  ↓
Atlas-guided coronal slices 107–117
  ↓
Five overlapping 2.5D stacks
```

Normalization is performed independently for each participant using voxels
inside the transformed brain mask:

```text
z = (I - mean_brain) / std_brain
```

Background voxels are set to zero.

The currently validated preprocessing function uses affine registration and
does not perform a separate explicit reorientation operation.

---

## Baseline Model

A small CNN is implemented as the end-to-end baseline:

```text
Conv2D: 3 → 16
ReLU
MaxPool

Conv2D: 16 → 32
ReLU
MaxPool

Conv2D: 32 → 64
ReLU
MaxPool

Adaptive Average Pool
Linear: 64 → 1
```

Total parameters: **23,649**.

---

## Main Model — ConvNeXt V2 Tiny

The primary model is:

```text
timm/convnextv2_tiny.fcmae_ft_in22k_in1k
```

Total parameters: **27,867,265**.

### Transfer-Learning Stage 1

The ConvNeXt V2 backbone is frozen and only the binary classification head
is trained. This configuration has **2,305 trainable parameters**.

### Transfer-Learning Stage 2

The final ConvNeXt V2 architectural stage and classification head can be
unfrozen for controlled fine-tuning with differential learning rates.

The reusable model module supports both configurations.

---

## Training Strategy

Binary dementia classification uses `BCEWithLogitsLoss`.

Class imbalance is handled using the frozen NACC training partition:

```text
positive class weight = 2394 / 365 ≈ 6.5589
```

The reusable training pipeline supports weighted binary loss, training and
validation loss calculation, model checkpointing, early stopping, CSV training
logs, and reproducible random seeds.

Model selection is based on NACC validation data. The official NACC test
partition remains separate from model development.

---

## Participant-Level Evaluation

Each participant generates five stack-level predictions. The primary
participant-level dementia probability is the mean of those five probabilities:

```text
P_i = mean(p_i1, p_i2, p_i3, p_i4, p_i5)
```

Implemented metrics include sensitivity, specificity, ROC-AUC, average precision,
balanced accuracy, precision, F1 score, and Matthews correlation coefficient (MCC).

The final classification threshold will be selected using the NACC validation
partition and frozen before evaluation on the NACC test and ADNI external-validation
cohorts.

---

## Explainability and Robustness

Two post-hoc explanation methods are currently implemented:

- **Grad-CAM**
- **SHAP GradientExplainer**

Grad-CAM targets the depthwise convolution in the final ConvNeXt V2 block.
Because the Stage-1 backbone is frozen, the explanation pipeline enables
gradients on an explanation-only copy of the input. This provides the gradient
path required for Grad-CAM without unfreezing or updating model parameters.

SHAP uses a background set drawn from training data only. Raw signed
channel-wise SHAP values are preserved for the three MRI slices. A normalized
2D attribution-magnitude map is also produced by averaging the absolute SHAP
magnitude across the three channels for visualization and future spatial
comparison.

Both methods have passed functional checks on development data. These checks
verify that valid, non-constant spatial explanation maps can be generated and
are not treated as experimental robustness results.

The later robustness experiments will evaluate explanation stability under:

- natural perturbations: noise, blur, and rotation
- adversarial perturbations: FGSM and PGD

The robustness analysis will focus on cases in which perturbations do not change
the predicted class, allowing explanation stability to be examined separately
from prediction instability.

---

## Repository Structure

```text
MRI_XAI_Robustness/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── split_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── utils.py
│   ├── xai.py
│   └── models/
│       ├── __init__.py
│       ├── baseline.py
│       └── main_model.py
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_main_model.ipynb
│   └── 04_xai_check.ipynb
├── data/
│   └── manifests/
├── checkpoints/
└── results/
    ├── logs/
    ├── figures/
    └── metrics/
```

Controlled MRI data and participant-level metadata are intentionally excluded
from version control.

---

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The preprocessing pipeline also requires the external `dcm2niix` command-line
tool. GPU acceleration is recommended for HD-BET and deep learning.

Development testing was performed in Google Colab with an NVIDIA Tesla T4 GPU.

---

## Reproducibility

The project uses a fixed default random seed of **42**.

The number has no scientific significance. It is simply a fixed pseudorandom
starting state so that data splitting and development experiments can be reproduced.

The frozen NACC train/validation/test split can be recreated exactly using the
reusable split module and seed 42.

---

## Data Availability and Privacy

NACC and ADNI MRI data are controlled research datasets and are not distributed
with this repository.

The repository intentionally excludes raw MRI data, DICOM files, NIfTI files,
derived 2.5D arrays, participant-level manifests, participant-level QC files,
participant-level predictions, model checkpoints, and credentials.

Researchers wishing to reproduce the study must obtain the appropriate datasets
directly from the authorized data providers and comply with their respective
data-use requirements.

---

## Current Development Status

The following components have been implemented and validated:

- MRI preprocessing pipeline
- 2.5D stack generation
- participant-level stratified splitting
- PyTorch dataset pipeline
- baseline CNN
- ConvNeXt V2 Tiny
- transfer-learning configuration
- weighted binary training
- checkpointing and early stopping
- stack-to-participant prediction aggregation
- participant-level evaluation metrics
- reproducibility utilities
- Grad-CAM explanation pipeline
- SHAP GradientExplainer pipeline
- XAI functionality checks

The current repository represents the code-development stage of the project.
Development experiments are used to verify that the pipeline functions correctly
and are not presented as final scientific results.

The Grad-CAM and SHAP implementation stage is complete at the functional-check
level. Full perturbation and explanation-robustness experiments, final NACC
evaluation, and ADNI external-validation experiments are subsequent stages of
the study.

---

## AI-Assisted Development Disclosure

Generative AI tools were used during code development to assist with code
organization, debugging, documentation, and explanation of programming concepts.

All project design decisions, data-selection criteria, preprocessing choices,
model configurations, and experimental procedures remain subject to researcher
review and validation. AI-generated code was tested before being incorporated
into the reusable project pipeline.
