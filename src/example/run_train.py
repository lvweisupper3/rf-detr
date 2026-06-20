# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Example script demonstrating how to train RF-DETR on a custom dataset.

This script shows the most common training workflows:

1. **Basic training** — train a detection model on a custom YOLO-format dataset
2. **COCO / Roboflow format** — train on a dataset exported from Roboflow
3. **Resuming training** — resume from a checkpoint
4. **Fine-tuning** — fine-tune with a lower learning rate and more epochs

Dataset Formats
---------------

**YOLO format** (``dataset_file="yolo"``)::

    dataset_dir/
    ├── data.yaml          # contains: names, nc, train, val paths
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img1.txt
    │       └── ...
    └── valid/
        ├── images/
        │   └── ...
        └── labels/
            └── ...

**COCO / Roboflow format** (``dataset_file="roboflow"`` or ``"coco"``)::

    dataset_dir/
    └── train/
        └── _annotations.coco.json

Usage
-----

Set ``DATASET_DIR`` below to the path of your dataset, then run::

    python src/example/run_train.py

Or import and call the individual functions from your own script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from rfdetr import RFDETRMedium, RFDETRSegMedium

# =========================================================================
# Configuration — edit these paths to match your setup
# =========================================================================

# Path to your custom dataset directory.  See the docstring above for the
# expected directory layout per format.
DATASET_DIR: str = "datasets/defect.v2i.yolov8"

# Directory where training outputs (checkpoints, logs, config) are saved.
OUTPUT_DIR: str = "output/training"


# =========================================================================
# Basic training — YOLO format
# =========================================================================

def train_detection_yolo(
    dataset_dir: str = DATASET_DIR,
    output_dir: str = OUTPUT_DIR,
    *,
    epochs: int = 100,
    batch_size: int | str = "auto",
    lr: float = 1e-4,
    resolution: int = 512,
    device: str = "cuda",
    num_workers: int = 2,
    resume: Optional[str] = None,
):
    """Train an object detection model on a YOLO-format dataset.

    This is the recommended starting point for training on a custom dataset
    that follows the YOLO directory layout (``data.yaml`` + ``train/`` +
    ``valid/`` with ``images/`` and ``labels/`` subdirectories).

    Args:
        dataset_dir: Path to the YOLO-format dataset root directory.
        output_dir: Directory for saving checkpoints, logs and config.
        epochs: Total number of training epochs.
        batch_size: Per-device batch size, or ``"auto"`` for automatic selection.
        lr: Base learning rate.
        resolution: Input image resolution in pixels. Must be divisible by
            ``patch_size * num_windows`` for the model variant.
        device: PyTorch device specifier (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
        num_workers: Number of DataLoader worker processes.
        resume: Optional path to a checkpoint from which to resume training.
    """

    model = RFDETRMedium()

    model.train(
        dataset_dir=dataset_dir,
        dataset_file="yolo",
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        # resolution=resolution,
        device=device,
        num_workers=num_workers,
        resume=resume,
        progress_bar="tqdm",
    )

    print(f"Training complete. Checkpoints saved to: {output_dir}")

    return model


# =========================================================================
# COCO / Roboflow format training
# =========================================================================

def train_detection_coco(
    dataset_dir: str = DATASET_DIR,
    output_dir: str = OUTPUT_DIR,
    *,
    epochs: int = 100,
    batch_size: int | str = 4,
    lr: float = 1e-4,
    resolution: int = 560,
    device: str = "cuda",
    num_workers: int = 2,
):
    """Train an object detection model on a COCO-format (Roboflow) dataset.

    Roboflow exports datasets in this format automatically.  The directory
    must contain ``train/_annotations.coco.json``.

    Args:
        dataset_dir: Path to the COCO-format dataset root directory.
        output_dir: Directory for saving checkpoints, logs and config.
        epochs: Total number of training epochs.
        batch_size: Per-device batch size, or ``"auto"`` for automatic selection.
        lr: Base learning rate.
        resolution: Input image resolution in pixels.
        device: PyTorch device specifier.
        num_workers: Number of DataLoader worker processes.
    """

    model = RFDETRMedium()

    model.train(
        dataset_dir=dataset_dir,
        dataset_file="roboflow",  # or "coco"
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resolution=resolution,
        device=device,
        num_workers=num_workers,
        progress_bar="tqdm",
    )

    print(f"Training complete. Checkpoints saved to: {output_dir}")

    return model


# =========================================================================
# Resuming from a checkpoint
# =========================================================================

def resume_training(
    checkpoint_path: str,
    dataset_dir: str = DATASET_DIR,
    output_dir: str = OUTPUT_DIR,
    *,
    epochs: int = 100,
    device: str = "cuda",
):
    """Resume training from a previously saved checkpoint.

    When ``resume`` is set, the optimizer state, learning rate scheduler state,
    and current epoch are all restored so training continues exactly where
    it left off.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g.
            ``output/training/checkpoint-epoch=050.ckpt``) or ``"last"``
            to automatically pick up the latest checkpoint.
        dataset_dir: Path to the dataset root directory.
        output_dir: Directory for saving new checkpoints and logs.
        epochs: Total number of epochs (should be > the checkpoint's epoch).
        device: PyTorch device specifier.
    """
    model = RFDETRMedium()

    model.train(
        dataset_dir=dataset_dir,
        dataset_file="yolo",
        output_dir=output_dir,
        epochs=epochs,
        resume=checkpoint_path,
        device=device,
        progress_bar="tqdm",
    )

    print(f"Resumed training complete. Checkpoints saved to: {output_dir}")

    return model


# =========================================================================
# Fine-tuning with lower learning rate
# =========================================================================

def finetune_detection(
    dataset_dir: str = DATASET_DIR,
    output_dir: str = OUTPUT_DIR,
    *,
    epochs: int = 50,
    batch_size: int | str = 4,
    lr: float = 1e-5,
    resolution: int = 560,
    device: str = "cuda",
):
    """Fine-tune a detection model with a conservative learning rate.

    Fine-tuning is useful when the custom dataset is small or when you want
    to adapt a pre-trained model to a new domain without catastrophic
    forgetting.  A lower learning rate and fewer epochs help preserve the
    pre-trained backbone features.

    Args:
        dataset_dir: Path to the dataset root directory.
        output_dir: Directory for saving checkpoints, logs and config.
        epochs: Total number of fine-tuning epochs.
        batch_size: Per-device batch size, or ``"auto"``.
        lr: Learning rate (lower than the default 1e-4 for fine-tuning).
        resolution: Input image resolution in pixels.
        device: PyTorch device specifier.
    """

    model = RFDETRMedium()

    model.train(
        dataset_dir=dataset_dir,
        dataset_file="yolo",
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resolution=resolution,
        device=device,
        progress_bar="tqdm",
        # Freeze the backbone at a lower LR for fine-tuning stability
        lr_encoder=lr * 0.1,
    )

    print(f"Fine-tuning complete. Checkpoints saved to: {output_dir}")

    return model


# =========================================================================
# Advanced: training with custom config
# =========================================================================

def train_with_full_config(
    dataset_dir: str = DATASET_DIR,
    output_dir: str = OUTPUT_DIR,
    *,
    device: str = "cuda",
):
    """Train with explicit control over every training hyperparameter.

    All keyword arguments passed to ``model.train()`` are forwarded to
    :class:`~rfdetr.config.TrainConfig`.  This function shows the
    full set of commonly used parameters.

    Args:
        dataset_dir: Path to the dataset root directory.
        output_dir: Directory for saving checkpoints, logs and config.
        device: PyTorch device specifier.
    """

    model = RFDETRMedium()

    model.train(
        # ---- Dataset ----
        dataset_dir=dataset_dir,
        dataset_file="yolo",  # "yolo", "coco", "roboflow", "o365"
        # ---- Training hyperparameters ----
        epochs=100,
        batch_size="auto",  # or "auto" for automatic batch size selection
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_scheduler="step",  # "step" or "cosine"
        lr_drop=100,
        warmup_epochs=0.0,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        # ---- EMA (Exponential Moving Average) ----
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        # ---- Data augmentation ----
        multi_scale=True,
        expanded_scales=True,
        augmentation_backend="cpu",  # "cpu", "auto", or "gpu"
        # ---- Regularization ----
        drop_path=0.0,
        # ---- Output & logging ----
        output_dir=output_dir,
        checkpoint_interval=10,
        eval_interval=1,
        tensorboard=True,
        wandb=False,
        mlflow=False,
        # ---- Early stopping ----
        early_stopping=False,
        early_stopping_patience=10,
        # ---- Hardware ----
        device=device,
        num_workers=2,
        # ---- Misc ----
        seed=42,
        progress_bar="tqdm",
        class_names=None,  # auto-detected from dataset if None
    )

    print(f"Training complete. Checkpoints saved to: {output_dir}")
    return model


# =========================================================================
# Predict with the trained model
# =========================================================================

def predict_trained_model(
    model,
    image_path: str,
    threshold: float = 0.5,
):
    """Run inference with a trained model and visualize results.

    Args:
        model: A trained :class:`RFDETR` instance (returned by any of the
            training functions above).
        image_path: Path to an image file.
        threshold: Confidence threshold for detections.
    """
    import cv2
    import numpy as np
    import supervision as sv

    detections = model.predict(image_path, threshold=threshold)

    class_names = getattr(model.model, "class_names", None)
    if class_names and detections.class_id is not None:
        labels = [f"{class_names[cid]}" for cid in detections.class_id]
    else:
        labels = [f"class {cid}" for cid in (detections.class_id or [])]

    annotated = sv.BoxAnnotator().annotate(
        detections.metadata["source_image"], detections
    )
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
    annotated = np.array(annotated)

    cv2.imshow("Trained Model Predictions", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Choose a workflow by uncommenting one of the lines below.
    # Make sure DATASET_DIR points to a valid dataset directory.
    # ------------------------------------------------------------------

    if not Path(DATASET_DIR).exists():
        print(
            f"Dataset directory '{DATASET_DIR}' does not exist.\n"
            f"Please update DATASET_DIR at the top of this file to point "
            f"to a valid dataset.",
        )
        raise SystemExit(1)

    # --- YOLO format ---
    model = train_detection_yolo()

    # --- COCO / Roboflow format ---
    # model = train_detection_coco()

    # --- Resume from checkpoint ---
    # model = resume_training(checkpoint_path="output/training/checkpoint-epoch=050.ckpt")

    # --- Fine-tune ---
    # model = finetune_detection()

    # --- Full config ---
    # model = train_with_full_config()

    # --- Predict with the trained model ---
    # predict_trained_model(model, "path/to/test_image.jpg")
