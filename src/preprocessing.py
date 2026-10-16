"""
preprocessing.py

Validated NACC MRI preprocessing pipeline for the
MRI XAI robustness project.

Pipeline implemented by the production function:

    verified T1 DICOM ZIP
        -> DICOM extraction
        -> dcm2niix conversion
        -> N4 bias-field correction
        -> HD-BET brain extraction
        -> affine registration to MNI152NLin6Asym
        -> nearest-neighbor transformation of brain mask
        -> participant-specific brain-mask z-score normalization
        -> fixed coronal window 107-117
        -> five overlapping 3-slice 2.5D stacks

Canonical saved stack shape:
    (182, 182, 3)

The model-specific resize from 182 x 182 to 224 x 224 is
performed dynamically by dataset.py and is not baked into
preprocessing.

Important:
    This module preserves the validated production behavior.
    The validated 50-participant production function did not
    perform a separate explicit reorientation operation.
"""

import gc
import os
import shutil
import subprocess
import time
import zipfile

from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import torch


def create_25d_stacks(
    participant,
    registered_path,
    output_root,
    start_slice=107,
    end_slice=117
):
    """
    Create five overlapping 3-channel 2.5D stacks
    from an 11-slice coronal MRI window.

    Window:
        107-117

    Stacks:
        1: 107,108,109
        2: 109,110,111
        3: 111,112,113
        4: 113,114,115
        5: 115,116,117

    Input:
        Affine-registered 3D T1 MRI in MNI152NLin6Asym space.

    Output:
        Five NumPy arrays with shape (H, W, 3).

    No intensity normalization or resizing is performed here.
    """

    start_time = time.perf_counter()

    # -------------------------------
    # Load registered MRI
    # -------------------------------
    img = nib.load(registered_path)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"{participant}: expected 3D MRI, got shape {data.shape}"
        )

    # -------------------------------
    # Extract 11 coronal slices
    # -------------------------------
    slice_indices = list(
        range(start_slice, end_slice + 1)
    )

    if len(slice_indices) != 11:
        raise ValueError(
            "The selected window must contain exactly 11 slices."
        )

    window = data[:, slice_indices, :]

    # Shape should now be:
    # X × 11 × Z

    # -------------------------------
    # Construct five overlapping
    # three-slice stacks
    # -------------------------------
    stack_positions = [
        (0, 1, 2),
        (2, 3, 4),
        (4, 5, 6),
        (6, 7, 8),
        (8, 9, 10)
    ]

    participant_output = os.path.join(
        output_root,
        participant
    )

    os.makedirs(
        participant_output,
        exist_ok=True
    )

    stack_records = []

    for stack_number, positions in enumerate(
        stack_positions,
        start=1
    ):

        actual_slices = [
            slice_indices[p]
            for p in positions
        ]

        # Extract the three coronal slices
        slices = [
            data[:, s, :]
            for s in actual_slices
        ]

        # Combine as channels
        stack = np.stack(
            slices,
            axis=-1
        ).astype(np.float32)

        output_file = os.path.join(
            participant_output,
            f"{participant}_stack{stack_number}.npy"
        )

        np.save(
            output_file,
            stack
        )

        stack_records.append({
            "participant": participant,
            "stack": stack_number,
            "slice_1": actual_slices[0],
            "slice_2": actual_slices[1],
            "slice_3": actual_slices[2],
            "shape": str(stack.shape),
            "file": output_file
        })

    elapsed = time.perf_counter() - start_time

    return stack_records, elapsed


def preprocess_nacc_participant_production(
    participant,
    raw_zip_path,
    template_path,
    local_work_root,
    drive_derived_root,
    drive_stack_root
):
    """
    Production preprocessing for one NACC participant.

    Pipeline
    --------
    Raw MRI ZIP
        -> DICOM extraction
        -> DICOM to NIfTI
        -> N4 bias correction
        -> HD-BET brain extraction
        -> affine registration to MNI152NLin6Asym
        -> transform HD-BET mask with nearest-neighbor interpolation
        -> participant-specific brain-mask z-score normalization
        -> fixed coronal window 107-117
        -> five overlapping 3-slice 2.5D stacks

    Important
    ---------
    Outputs are progressively saved to Google Drive.
    Timing and QC information are returned for logging.
    """

    total_start = time.perf_counter()

    participant_work = Path(local_work_root) / participant
    dicom_dir = participant_work / "dicom"
    nifti_dir = participant_work / "nifti"
    prep_dir = participant_work / "preprocessed"

    drive_participant = (
        Path(drive_derived_root) / participant
    )

    drive_stack_participant = (
        Path(drive_stack_root) / participant
    )

    # ------------------------------------------------
    # Restart protection
    # ------------------------------------------------

    expected_stacks = [
        drive_stack_participant /
        f"{participant}_stack{i}.npy"
        for i in range(1, 6)
    ]

    normalized_drive_path = (
        drive_participant /
        f"{participant}_T1_N4_brain_MNI_affine_zscore.nii.gz"
    )

    if (
        normalized_drive_path.exists()
        and all(path.exists() for path in expected_stacks)
    ):
        return {
            "participant": participant,
            "status": "SKIPPED_ALREADY_COMPLETE",
            "total_sec": 0.0
        }

    # Fresh local workspace for this participant
    if participant_work.exists():
        shutil.rmtree(participant_work)

    for folder in [
        dicom_dir,
        nifti_dir,
        prep_dir,
        drive_participant,
        drive_stack_participant
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    record = {
        "participant": participant,
        "status": "FAILED"
    }

    try:

        # ============================================================
        # 1. Extract DICOM files
        # ============================================================

        start = time.perf_counter()

        with zipfile.ZipFile(raw_zip_path, "r") as z:
            z.extractall(dicom_dir)

        dicom_files = list(
            dicom_dir.rglob("*.dcm")
        )

        record["dicom_files"] = len(dicom_files)
        record["extraction_sec"] = (
            time.perf_counter() - start
        )

        if len(dicom_files) == 0:
            raise ValueError(
                "No DICOM files found after extraction."
            )

        # ============================================================
        # 2. DICOM -> NIfTI
        # ============================================================

        start = time.perf_counter()

        subprocess.run(
            [
                "dcm2niix",
                "-z", "y",
                "-f", f"{participant}_T1",
                "-o", str(nifti_dir),
                str(dicom_dir)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        record["dcm2niix_sec"] = (
            time.perf_counter() - start
        )

        nifti_candidates = sorted(
            nifti_dir.glob(
                f"{participant}_T1*.nii.gz"
            )
        )

        if len(nifti_candidates) == 0:
            raise FileNotFoundError(
                "dcm2niix did not produce a NIfTI file."
            )

        # Our verified raw ZIP represents one selected T1 series.
        nifti_path = nifti_candidates[0]

        # ============================================================
        # 3. N4 bias correction
        # ============================================================

        start = time.perf_counter()

        image = ants.image_read(
            str(nifti_path)
        )

        n4 = ants.n4_bias_field_correction(
            image
        )

        n4_path = (
            prep_dir /
            f"{participant}_T1_N4.nii.gz"
        )

        ants.image_write(
            n4,
            str(n4_path)
        )

        record["n4_sec"] = (
            time.perf_counter() - start
        )

        # ============================================================
        # 4. HD-BET
        # ============================================================

        brain_path = (
            prep_dir /
            f"{participant}_T1_N4_brain.nii.gz"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = time.perf_counter()

        subprocess.run(
            [
                "hd-bet",
                "-i", str(n4_path),
                "-o", str(brain_path),
                "-device", "cuda",
                "--save_bet_mask"
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        record["hd_bet_sec"] = (
            time.perf_counter() - start
        )

        # HD-BET mask produced by the validated pilot pipeline
        original_mask_path = (
            prep_dir /
            f"{participant}_T1_N4_brain_bet.nii.gz"
        )

        if not original_mask_path.exists():
            raise FileNotFoundError(
                f"HD-BET mask not found: "
                f"{original_mask_path.name}"
            )

        # ============================================================
        # 5. Affine MNI registration
        # ============================================================

        start = time.perf_counter()

        fixed = ants.image_read(
            str(template_path)
        )

        moving = ants.image_read(
            str(brain_path)
        )

        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        registered = registration[
            "warpedmovout"
        ]

        registered_path = (
            prep_dir /
            f"{participant}_T1_N4_brain_MNI_affine.nii.gz"
        )

        ants.image_write(
            registered,
            str(registered_path)
        )

        affine_path = (
            prep_dir /
            f"{participant}_to_MNI_affine.mat"
        )

        shutil.copy2(
            registration["fwdtransforms"][0],
            affine_path
        )

        record["registration_sec"] = (
            time.perf_counter() - start
        )

        # ============================================================
        # 6. Transform HD-BET mask to MNI space
        # ============================================================

        start = time.perf_counter()

        moving_mask = ants.image_read(
            str(original_mask_path)
        )

        registered_mask_ants = ants.apply_transforms(
            fixed=fixed,
            moving=moving_mask,
            transformlist=[str(affine_path)],
            interpolator="nearestNeighbor"
        )

        mask_array = (
            registered_mask_ants.numpy() > 0.5
        ).astype(np.uint8)

        registered_mask_ants = ants.from_numpy(
            mask_array,
            origin=registered_mask_ants.origin,
            spacing=registered_mask_ants.spacing,
            direction=registered_mask_ants.direction
        )

        registered_mask_path = (
            prep_dir /
            f"{participant}_brain_mask_MNI_affine.nii.gz"
        )

        ants.image_write(
            registered_mask_ants,
            str(registered_mask_path)
        )

        record["mask_transform_sec"] = (
            time.perf_counter() - start
        )

        # ============================================================
        # 7. Participant-specific z-score normalization
        # ============================================================

        start = time.perf_counter()

        mri_img = nib.load(
            str(registered_path)
        )

        mri = mri_img.get_fdata(
            dtype=np.float32
        )

        mask = (
            nib.load(
                str(registered_mask_path)
            ).get_fdata() > 0.5
        )

        if mri.shape != mask.shape:
            raise ValueError(
                f"MRI/mask shape mismatch: "
                f"{mri.shape} vs {mask.shape}"
            )

        brain_values = mri[mask]

        if brain_values.size == 0:
            raise ValueError(
                "Registered brain mask is empty."
            )

        brain_mean = float(
            brain_values.mean()
        )

        brain_std = float(
            brain_values.std()
        )

        if brain_std <= 0:
            raise ValueError(
                "Brain intensity SD is zero."
            )

        normalized = np.zeros_like(
            mri,
            dtype=np.float32
        )

        normalized[mask] = (
            mri[mask] - brain_mean
        ) / brain_std

        normalized_brain = normalized[mask]

        normalized_path = (
            prep_dir /
            f"{participant}_T1_N4_brain_MNI_affine_zscore.nii.gz"
        )

        normalized_img = nib.Nifti1Image(
            normalized,
            mri_img.affine,
            mri_img.header
        )

        nib.save(
            normalized_img,
            str(normalized_path)
        )

        record["brain_voxels"] = int(
            mask.sum()
        )

        record["original_mean"] = (
            brain_mean
        )

        record["original_std"] = (
            brain_std
        )

        record["normalized_mean"] = float(
            normalized_brain.mean()
        )

        record["normalized_std"] = float(
            normalized_brain.std()
        )

        record["normalization_sec"] = (
            time.perf_counter() - start
        )

        # ============================================================
        # 8. Create five normalized 2.5D stacks
        # ============================================================

        stack_records, stack_sec = (
            create_25d_stacks(
                participant=participant,
                registered_path=str(
                    normalized_path
                ),
                output_root=str(
                    drive_stack_root
                ),
                start_slice=107,
                end_slice=117
            )
        )

        record["stack_generation_sec"] = (
            stack_sec
        )

        record["stacks_created"] = len(
            stack_records
        )

        if len(stack_records) != 5:
            raise ValueError(
                f"Expected 5 stacks, "
                f"created {len(stack_records)}."
            )

        # ============================================================
        # 9. Progressively save important derived outputs
        #    to Google Drive
        # ============================================================

        start = time.perf_counter()

        files_to_save = [
            n4_path,
            brain_path,
            original_mask_path,
            registered_path,
            affine_path,
            registered_mask_path,
            normalized_path
        ]

        for src in files_to_save:
            shutil.copy2(
                src,
                drive_participant /
                src.name
            )

        record["save_sec"] = (
            time.perf_counter() - start
        )

        # ============================================================
        # 10. Final verification
        # ============================================================

        final_stack_files = [
            drive_stack_participant /
            f"{participant}_stack{i}.npy"
            for i in range(1, 6)
        ]

        if not all(
            path.exists()
            for path in final_stack_files
        ):
            raise FileNotFoundError(
                "One or more final 2.5D stacks are missing."
            )

        record["status"] = "SUCCESS"

    except Exception as e:

        record["error"] = str(e)

    finally:

        record["total_sec"] = (
            time.perf_counter() - total_start
        )

        # Release memory before next participant
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return record
