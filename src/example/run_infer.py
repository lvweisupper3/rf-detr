# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Example script demonstrating how to run inference with RF-DETR on a test dataset.

This script shows common inference workflows:

1. **Single image prediction** — predict on one image and visualize results
2. **Batch prediction on a directory** — iterate over a folder of images
3. **Prediction from checkpoint** — load a trained model and run inference
4. **Prediction with COCO evaluation** — evaluate a trained model on a COCO-format test dataset

Usage
-----

Set the configuration variables below, then run::

    python src/example/run_infer.py

Or import and call the individual functions from your own script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm.auto import tqdm

from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

# =========================================================================
# Configuration — edit these paths to match your setup
# =========================================================================

# Path to a single test image for quick testing.
TEST_IMAGE_PATH: str = "path/to/test_image.jpg"

# Directory containing test images for batch prediction.
TEST_IMAGES_DIR: str = "path/to/test_images"

# Directory to save annotated prediction results.
OUTPUT_DIR: str = "output/predictions"

# Path to a trained model checkpoint (set to None to use pretrained weights).
CHECKPOINT_PATH: Optional[str] = None

# Confidence threshold for filtering detections.
CONFIDENCE_THRESHOLD: float = 0.5

# Device for inference ("cuda", "cpu", etc.).
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================
# Single image prediction
# =========================================================================


def predict_single_image(
    image_path: str = TEST_IMAGE_PATH,
    threshold: float = CONFIDENCE_THRESHOLD,
    output_dir: Optional[str] = OUTPUT_DIR,
    *,
    show: bool = True,
) -> sv.Detections:
    """Run inference on a single image and optionally display/save the result.

    Args:
        image_path: Path to the input image file.
        threshold: Confidence threshold for filtering detections.
        output_dir: Directory to save the annotated image. If None, the image is not saved.
        show: Whether to display the annotated image in a window.

    Returns:
        A supervision :class:`~supervision.Detections` object containing the predictions.
    """
    model = RFDETRMedium(device=DEVICE)

    detections: sv.Detections | Any = model.predict(image_path, threshold=threshold)

    # Build label strings from class IDs.
    class_ids = detections.class_id if detections.class_id is not None else []
    labels = [
        f"{COCO_CLASSES[cid]} {conf:.2f}"
        for cid, conf in zip(class_ids, detections.confidence or [])
    ]

    # Annotate the source image with bounding boxes and labels.
    annotated_image = sv.BoxAnnotator().annotate(
        detections.metadata["source_image"], detections
    )
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    annotated_image = np.array(annotated_image)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved to: {output_path}")

    if show:
        cv2.imshow(
            "RF-DETR Prediction", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


# =========================================================================
# Batch prediction on a directory of images
# =========================================================================


def predict_directory(
    images_dir: str = TEST_IMAGES_DIR,
    threshold: float = CONFIDENCE_THRESHOLD,
    output_dir: str = OUTPUT_DIR,
    *,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> list[sv.Detections]:
    """Run inference on all images in a directory and save annotated results.

    Args:
        images_dir: Path to the directory containing test images.
        threshold: Confidence threshold for filtering detections.
        output_dir: Directory to save annotated images.
        image_extensions: File extensions to consider as images.

    Returns:
        A list of supervision :class:`~supervision.Detections` objects, one per image.
    """
    model = RFDETRMedium(device=DEVICE)

    # Collect all image paths from the directory.
    image_paths: list[str] = []
    for ext in image_extensions:
        image_paths.extend(str(p) for p in Path(images_dir).rglob(f"*{ext}"))
        image_paths.extend(str(p) for p in Path(images_dir).rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in: {images_dir}")
        return []

    print(f"Found {len(image_paths)} images in {images_dir}")

    os.makedirs(output_dir, exist_ok=True)
    all_detections: list[sv.Detections] = []

    for image_path in tqdm(image_paths, desc="Predicting"):
        detections: sv.Detections | Any = model.predict(image_path, threshold=threshold)
        all_detections.append(detections)

        # Build labels.
        class_ids = detections.class_id if detections.class_id is not None else []
        labels = [
            f"{COCO_CLASSES[cid]} {conf:.2f}"
            for cid, conf in zip(class_ids, detections.confidence or [])
        ]

        # Annotate and save.
        annotated_image = sv.BoxAnnotator().annotate(
            detections.metadata["source_image"], detections
        )
        annotated_image = sv.LabelAnnotator().annotate(
            annotated_image, detections, labels
        )
        annotated_image = np.array(annotated_image)

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    print(f"Annotated images saved to: {output_dir}")
    return all_detections


# =========================================================================
# Prediction from a trained checkpoint
# =========================================================================


def predict_from_checkpoint(
    checkpoint_path: str,
    images_dir: str = TEST_IMAGES_DIR,
    threshold: float = CONFIDENCE_THRESHOLD,
    output_dir: str = OUTPUT_DIR,
) -> list[sv.Detections]:
    """Load a trained model from a checkpoint and run inference on a directory of images.

    The model class is automatically inferred from the checkpoint metadata via
    :meth:`RFDETR.from_checkpoint`.  Custom class names stored in the checkpoint are
    used for labeling when available.

    Args:
        checkpoint_path: Path to a training checkpoint file (e.g.
            ``output/training/checkpoint-epoch=050.ckpt``).
        images_dir: Path to the directory containing test images.
        threshold: Confidence threshold for filtering detections.
        output_dir: Directory to save annotated images.

    Returns:
        A list of supervision :class:`~supervision.Detections` objects, one per image.
    """
    from rfdetr import RFDETR

    model = RFDETR.from_checkpoint(checkpoint_path, device=DEVICE)
    class_names = model.class_names

    # Collect image paths.
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_paths: list[str] = []
    for ext in image_extensions:
        image_paths.extend(str(p) for p in Path(images_dir).rglob(f"*{ext}"))
        image_paths.extend(str(p) for p in Path(images_dir).rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in: {images_dir}")
        return []

    print(f"Found {len(image_paths)} images in {images_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_detections: list[sv.Detections] = []

    for image_path in tqdm(image_paths, desc="Predicting"):
        detections: sv.Detections | Any = model.predict(image_path, threshold=threshold)
        all_detections.append(detections)

        # Use custom class names from the checkpoint when available.
        class_ids = detections.class_id if detections.class_id is not None else []
        confidences = detections.confidence if detections.confidence is not None else []
        
        labels = [
            f"{class_names[cid] if cid < len(class_names) else f'class {cid}'} {conf:.2f}"
            for cid, conf in zip(class_ids, confidences)
        ]

        annotated_image = sv.BoxAnnotator().annotate(
            detections.metadata["source_image"], detections
        )
        annotated_image = sv.LabelAnnotator().annotate(
            annotated_image, detections, labels
        )
        annotated_image = np.array(annotated_image)

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    print(f"Annotated images saved to: {output_dir}")
    return all_detections


# =========================================================================
# COCO evaluation on a test dataset
# =========================================================================


def evaluate_coco_dataset(
    checkpoint_path: str,
    dataset_dir: str,
    output_dir: str = OUTPUT_DIR,
    *,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate a trained model on a COCO-format test dataset.

    The dataset directory must contain a ``test/`` folder with
    ``_annotations.coco.json`` and the corresponding images.

    Args:
        checkpoint_path: Path to a training checkpoint file.
        dataset_dir: Path to the COCO-format dataset root directory.
        output_dir: Directory to save evaluation results and visualizations.
        threshold: Confidence threshold for filtering detections.

    Returns:
        A dictionary of COCO evaluation metrics.

    Raises:
        FileNotFoundError: If the COCO annotation file is not found.
    """
    from rfdetr import RFDETR

    coco_json_path = os.path.join(dataset_dir, "test", "_annotations.coco.json")
    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(
            f"COCO annotation file not found at: {coco_json_path}. "
            f"Expected directory layout: {dataset_dir}/test/_annotations.coco.json"
        )

    model = RFDETR.from_checkpoint(checkpoint_path, device=DEVICE)
    class_names = model.class_names

    # Load COCO ground truth.
    from pycocotools.coco import COCO

    coco_gt = COCO(coco_json_path)
    image_ids = coco_gt.getImgIds()

    os.makedirs(output_dir, exist_ok=True)
    predictions: list[dict[str, Any]] = []

    for img_id in tqdm(image_ids, desc="Evaluating"):
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(dataset_dir, "test", img_info["file_name"])

        if not os.path.isfile(image_path):
            print(f"Warning: image not found, skipping: {image_path}")
            continue

        detections: sv.Detections | Any = model.predict(image_path, threshold=threshold)

        if detections.xyxy is None or len(detections.xyxy) == 0:
            continue

        # Convert to COCO prediction format.
        for box, score, class_id in zip(
            detections.xyxy, detections.confidence or [], detections.class_id or []
        ):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": class_id + 1,  # COCO uses 1-indexed category IDs
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    # Save predictions to file.
    pred_path = os.path.join(output_dir, "coco_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"Predictions saved to: {pred_path}")

    # Run COCO evaluation.
    if not predictions:
        print("No predictions generated. Skipping evaluation.")
        return {}

    coco_dt = coco_gt.loadRes(pred_path)

    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP@0.50:0.95": coco_eval.stats[0],
        "AP@0.50": coco_eval.stats[1],
        "AP@0.75": coco_eval.stats[2],
        "AP@0.50:0.95 (small)": coco_eval.stats[3],
        "AP@0.50:0.95 (medium)": coco_eval.stats[4],
        "AP@0.50:0.95 (large)": coco_eval.stats[5],
        "AR@0.50:0.95 (max 1)": coco_eval.stats[6],
        "AR@0.50:0.95 (max 10)": coco_eval.stats[7],
        "AR@0.50:0.95 (max 100)": coco_eval.stats[8],
    }

    # Save metrics to file.
    metrics_path = os.path.join(output_dir, "coco_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return metrics


# =========================================================================
# Batch prediction with optimized model (lower latency)
# =========================================================================


def predict_optimized(
    images_dir: str = TEST_IMAGES_DIR,
    threshold: float = CONFIDENCE_THRESHOLD,
    output_dir: str = OUTPUT_DIR,
    *,
    compile_model: bool = True,
    dtype: str = "float16",
) -> list[sv.Detections]:
    """Run inference with an optimized model for lower latency.

    Calls :meth:`RFDETR.optimize_for_inference` before prediction to JIT-compile
    the model and cast to the specified dtype.  This reduces per-image latency at
    the cost of a one-time compilation step.

    Args:
        images_dir: Path to the directory containing test images.
        threshold: Confidence threshold for filtering detections.
        output_dir: Directory to save annotated images.
        compile_model: Whether to JIT-compile the model with ``torch.jit.trace``.
        dtype: Target dtype for the optimized model (``"float16"`` or ``"float32"``).

    Returns:
        A list of supervision :class:`~supervision.Detections` objects, one per image.
    """
    model = RFDETRMedium(device=DEVICE)

    # Optimize the model for faster inference.
    model.optimize_for_inference(compile=compile_model, dtype=dtype)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_paths: list[str] = []
    for ext in image_extensions:
        image_paths.extend(str(p) for p in Path(images_dir).rglob(f"*{ext}"))
        image_paths.extend(str(p) for p in Path(images_dir).rglob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in: {images_dir}")
        return []

    print(f"Found {len(image_paths)} images in {images_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_detections: list[sv.Detections] = []

    for image_path in tqdm(image_paths, desc="Predicting (optimized)"):
        detections: sv.Detections | Any = model.predict(image_path, threshold=threshold)
        all_detections.append(detections)

        class_ids = detections.class_id if detections.class_id is not None else []
        labels = [
            f"{COCO_CLASSES[cid]} {conf:.2f}"
            for cid, conf in zip(class_ids, detections.confidence or [])
        ]

        annotated_image = sv.BoxAnnotator().annotate(
            detections.metadata["source_image"], detections
        )
        annotated_image = sv.LabelAnnotator().annotate(
            annotated_image, detections, labels
        )
        annotated_image = np.array(annotated_image)

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    print(f"Annotated images saved to: {output_dir}")

    # Clean up the optimized model to free GPU memory.
    model.remove_optimized_model()

    return all_detections


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Choose a workflow by uncommenting one of the lines below.
    # Make sure the paths at the top of this file point to valid data.
    # ------------------------------------------------------------------

    # --- Single image prediction (pretrained model) ---
    # predict_single_image()

    # --- Batch prediction on a directory (pretrained model) ---
    # predict_directory()

    # --- Prediction from a trained checkpoint ---
    predict_from_checkpoint(
        checkpoint_path="output/training/checkpoint_best_total.pth",
        images_dir="datasets/defect.v2i.yolov8/test/images",
    )

    # --- COCO evaluation on a test dataset ---
    # metrics = evaluate_coco_dataset(
    #     checkpoint_path="output/training/checkpoint-epoch=050.ckpt",
    #     dataset_dir="path/to/coco_dataset",
    # )

    # --- Optimized batch prediction (lower latency) ---
    # predict_optimized()

    print(
        "No workflow selected. Please edit the __main__ section to uncomment "
        "a workflow, and update the paths at the top of this file."
    )
