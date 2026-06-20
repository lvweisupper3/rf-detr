# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Example: train RF-DETR on a custom dataset.

This script demonstrates three common workflows for training RF-DETR on your own
data, in order of increasing flexibility:

1. **Quick-start (high-level API)** — ``model.train(...)``: auto-detects the
   dataset format and class count, runs training end-to-end, and syncs the
   trained weights back so ``model.predict()`` / ``model.export()`` just work.
   This is the recommended path for most users.

2. **Low-level PTL API** — ``RFDETRModelModule`` + ``RFDETRDataModule`` +
   ``build_trainer`` + ``trainer.fit(...)``: gives you full control over
   PyTorch Lightning callbacks, checkpointing, and the training loop.

3. **Fine-tuning from a pre-trained checkpoint** — ``RFDETR.from_checkpoint()``
   loads an existing ``.pth``, adapts the detection head to your custom class
   count, and resumes training.

**Supported dataset formats (set via ``dataset_file`` in ``TrainConfig``):**

+---------------+--------------------------------------------------+
| ``"roboflow"``| Auto-detect COCO or YOLO format (default).       |
+---------------+--------------------------------------------------+
| ``"coco"``    | COCO JSON format (``_annotations.coco.json``).   |
+---------------+--------------------------------------------------+
| ``"yolo"``    | YOLO format (``data.yaml`` + images/labels dirs).|
+---------------+--------------------------------------------------+

**Expected dataset directory layouts:**

COCO format (``dataset_file="roboflow"`` or ``"coco"``)::

    dataset_root/
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── valid/
    │   ├── _annotations.coco.json
    │   ├── image3.jpg
    │   └── ...
    └── test/                          # optional
        ├── _annotations.coco.json
        └── ...

YOLO format (``dataset_file="roboflow"`` or ``"yolo"``)::

    dataset_root/
    ├── data.yaml                      # names, nc, train/val paths
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── image1.txt
    │       └── ...
    └── valid/
        ├── images/
        │   └── ...
        └── labels/
            └── ...

Requirements:
    Install training extras before running this script::

        pip install "rfdetr[train,loggers]"

Usage:
    python -m src.example.train_custom_data --dataset-root /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall
from rfdetr.config import RFDETRBaseConfig, RFDETRNanoConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """Return a torch-style device string for the best available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _inspect_dataset(dataset_root: Path) -> Dict[str, Any]:
    """Inspect a dataset directory and return summary information.

    Args:
        dataset_root: Path to the dataset root directory.

    Returns:
        A dict with keys ``format``, ``num_classes``, and ``class_names``.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
        ValueError: If the dataset format cannot be detected.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # COCO format detection
    coco_train = dataset_root / "train" / "_annotations.coco.json"
    if coco_train.exists():
        with open(coco_train, encoding="utf-8") as f:
            coco_data = json.load(f)
        categories: List[Dict[str, Any]] = coco_data.get("categories", [])
        cat_by_id = {cat["id"]: cat for cat in categories}
        class_names = [cat_by_id[cid]["name"] for cid in sorted(cat_by_id)]
        return {
            "format": "coco",
            "num_classes": len(cat_by_id),
            "class_names": class_names,
        }

    # YOLO format detection
    for yaml_name in ("data.yaml", "data.yml"):
        yaml_path = dataset_root / yaml_name
        if yaml_path.exists():
            # Parse the simple YAML subset that YOLO datasets use.
            import yaml

            with open(yaml_path, encoding="utf-8") as f:
                yolo_cfg = yaml.safe_load(f)
            names: List[str] = yolo_cfg.get("names", [])
            if isinstance(names, dict):
                names = [names[int(i)] for i in sorted(names.keys())]
            num_classes = yolo_cfg.get("nc", len(names))
            return {
                "format": "yolo",
                "num_classes": num_classes if num_classes else len(names),
                "class_names": names,
            }

    raise ValueError(
        f"Could not detect dataset format in {dataset_root}. "
        f"Expected either COCO format (train/_annotations.coco.json) "
        f"or YOLO format (data.yaml + train/images/)."
    )


# ---------------------------------------------------------------------------
# Workflow 1 — Quick-start with the high-level API
# ---------------------------------------------------------------------------


def train_quickstart(
    dataset_root: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 4,
    model_variant: str = "nano",
    device: Optional[str] = None,
    resolution: Optional[int] = None,
) -> RFDETR:
    """Train an RF-DETR model using the high-level ``model.train()`` API.

    This is the simplest way to get started.  The method auto-detects the dataset
    format and class count, builds the correct model head, trains, and syncs the
    trained weights back — ``model.predict()`` works immediately afterwards.

    Args:
        dataset_root: Path to the dataset directory (COCO or YOLO format).
        output_dir: Directory where checkpoints and logs will be written.
        epochs: Number of training epochs.
        batch_size: Per-device batch size (micro-batch).
        model_variant: One of ``"nano"``, ``"small"``, or ``"base"``.
        device: Torch device string (``"cuda"``, ``"cpu"``, ``"mps"``).
            Auto-detected when omitted.
        resolution: Optional input resolution override.  Must be divisible
            by ``patch_size * num_windows``.  When omitted the model default
            is used.

    Returns:
        The trained :class:`RFDETR` model, ready for inference.
    """
    device = device or _get_device()
    info = _inspect_dataset(dataset_root)
    print(f"Dataset: {info['format']} format, {info['num_classes']} classes: {info['class_names']}")

    # Create the model.  num_classes is auto-detected from the dataset during
    # training when not explicitly set, but providing it here ensures the model
    # head is built at the correct width from the start.
    variant_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
    }
    model_cls = variant_map.get(model_variant, RFDETRNano)
    model = model_cls(num_classes=info["num_classes"])

    # All keyword arguments are forwarded to TrainConfig.
    train_kwargs: Dict[str, Any] = {
        "dataset_dir": str(dataset_root),
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        # Use class names from the dataset for logs and evaluation.
        "class_names": info["class_names"],
        # Optional: produce dataset grid visualizations for debugging.
        "save_dataset_grids": True,
    }
    if resolution is not None:
        train_kwargs["resolution"] = resolution

    print(f"Starting training on {device} for {epochs} epochs...")
    model.train(**train_kwargs)

    print(f"Training complete. Checkpoint saved to {output_dir}")
    return model


# ---------------------------------------------------------------------------
# Workflow 2 — Low-level PyTorch Lightning API
# ---------------------------------------------------------------------------


def train_low_level(
    dataset_root: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 4,
    model_variant: str = "nano",
    device: Optional[str] = None,
    resume_checkpoint: Optional[Path] = None,
) -> RFDETRModelModule:
    """Train using the low-level PTL stack for full control.

    Use this approach when you need custom PyTorch Lightning callbacks, gradient
    accumulation tuning, mixed-precision settings, or custom trainer
    configuration that the high-level API doesn't expose directly.

    Args:
        dataset_root: Path to the dataset directory.
        output_dir: Directory for checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Per-device micro-batch size.
        model_variant: ``"nano"``, ``"small"``, or ``"base"``.
        device: Torch device string.  Auto-detected when omitted.
        resume_checkpoint: Optional path to a ``.ckpt`` file to resume from.

    Returns:
        The trained :class:`RFDETRModelModule`.
    """
    device = device or _get_device()
    info = _inspect_dataset(dataset_root)
    print(f"Dataset: {info['format']} format, {info['num_classes']} classes: {info['class_names']}")

    # Map device string to PTL accelerator.
    if device == "cuda":
        accelerator = "gpu"
    elif device == "mps":
        accelerator = "mps"
    else:
        accelerator = "cpu"

    # --- Build configs ---
    config_cls_map = {
        "nano": RFDETRNanoConfig,
        "small": RFDETRSmallConfig,
        "base": RFDETRBaseConfig,
    }
    config_cls = config_cls_map.get(model_variant, RFDETRNanoConfig)
    model_config = config_cls(num_classes=info["num_classes"], pretrain_weights=None)
    train_config = TrainConfig(
        dataset_dir=str(dataset_root),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        lr_encoder=1.5e-4,
        warmup_epochs=1.0,
        grad_accum_steps=1,  # set > 1 when batch_size is small
        class_names=info["class_names"],
        use_ema=True,
        num_workers=2,
        tensorboard=True,
        # Disable multi-scale for faster initial experiments; re-enable for
        # final training runs to improve robustness.
        multi_scale=False,
    )

    # --- Build module and datamodule ---
    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)

    # --- Build trainer and fit ---
    trainer = build_trainer(
        train_config,
        model_config,
        accelerator=accelerator,
        devices=1,
    )
    trainer.fit(module, datamodule, ckpt_path=str(resume_checkpoint) if resume_checkpoint else None)

    # --- Validate after training ---
    if not train_config.run_test:
        val_results = trainer.validate(module, datamodule)
        print(f"Validation results: {val_results}")

    print(f"Training complete. Best checkpoint at {output_dir}")
    return module


# ---------------------------------------------------------------------------
# Workflow 3 — Fine-tuning from a checkpoint
# ---------------------------------------------------------------------------


def train_from_checkpoint(
    checkpoint_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> RFDETR:
    """Fine-tune an RF-DETR model from an existing training checkpoint.

    ``RFDETR.from_checkpoint()`` infers the model class and architecture from the
    checkpoint file, loads the trained weights, and adapts the detection head to
    the new dataset's class count.  This is ideal for transfer learning: start
    from a model already trained on one domain and adapt it to another.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint file.
        dataset_root: Path to the new dataset directory.
        output_dir: Directory for the fine-tuned checkpoints.
        epochs: Number of fine-tuning epochs.
        batch_size: Per-device micro-batch size.
        device: Torch device string.  Auto-detected when omitted.

    Returns:
        The fine-tuned :class:`RFDETR` model, ready for inference.
    """
    device = device or _get_device()
    info = _inspect_dataset(dataset_root)
    print(f"Target dataset: {info['format']} format, {info['num_classes']} classes: {info['class_names']}")

    # Load model from checkpoint — class is inferred automatically.
    # Pass num_classes to adapt the detection head, or omit it to let train()
    # auto-detect the class count from the new dataset.
    model = RFDETR.from_checkpoint(str(checkpoint_path), num_classes=info["num_classes"])
    print(f"Loaded checkpoint: {type(model).__name__}")

    model.train(
        dataset_dir=str(dataset_root),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        class_names=info["class_names"],
        # Use a lower learning rate for fine-tuning.
        lr=5e-5,
        lr_encoder=7.5e-5,
    )

    print(f"Fine-tuning complete. Checkpoints saved to {output_dir}")
    return model


# ---------------------------------------------------------------------------
# Workflow 4 — Post-training inference
# ---------------------------------------------------------------------------


def run_inference(model: RFDETR, image_paths: List[Path]) -> None:
    """Run object detection inference with a trained model.

    Args:
        model: A trained :class:`RFDETR` instance.
        image_paths: List of paths to images for inference.
    """
    print(f"\nRunning inference on {len(image_paths)} image(s)...")

    for image_path in image_paths:
        if not image_path.exists():
            print(f"  [SKIP] {image_path}: file not found")
            continue

        detections = model.predict(str(image_path))
        print(f"  {image_path.name}: {len(detections.xyxy)} detections")
        for i, (box, class_id, confidence) in enumerate(
            zip(detections.xyxy, detections.class_id, detections.confidence)
        ):
            class_name = detections.data.get("class_name", [None])[i] if hasattr(detections, "data") else f"cls_{class_id}"
            print(f"    [{class_name}] confidence={confidence:.2%} box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on a custom dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick-start training on a COCO-format dataset
  python -m src.example.train_custom_data --dataset-root ./my_data

  # Train on a YOLO-format dataset with more epochs
  python -m src.example.train_custom_data --dataset-root ./my_yolo_data --epochs 200

  # Fine-tune from a checkpoint
  python -m src.example.train_custom_data --from-checkpoint ./checkpoint.pth --dataset-root ./new_data

  # Run inference with a trained model
  python -m src.example.train_custom_data --predict ./checkpoint_best.pth --images ./img1.jpg ./img2.jpg
        """,
    )

    # Dataset / model inputs
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to the dataset root directory (COCO or YOLO format).",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Fine-tune from an existing training checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory for training outputs (default: ./output).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per device (default: 4).")
    parser.add_argument(
        "--model",
        choices=["nano", "small", "base"],
        default="nano",
        help="Model variant (default: nano).",
    )
    parser.add_argument("--device", default=None, help="Torch device: 'cuda', 'cpu', or 'mps' (auto-detected).")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input resolution override (must be divisible by patch_size * num_windows).",
    )
    parser.add_argument(
        "--low-level",
        action="store_true",
        help="Use the low-level PTL API instead of the high-level train() method.",
    )

    # Inference mode
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Path to a trained .pth checkpoint for inference (skips training).",
    )
    parser.add_argument("--images", type=Path, nargs="*", default=None, help="Image files for inference.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args and dispatch to the appropriate workflow."""
    args = _parse_args()

    # Inference-only mode
    if args.predict is not None:
        if not args.images:
            print("Error: --images is required when --predict is used.")
            return
        model = RFDETR.from_checkpoint(str(args.predict))
        run_inference(model, args.images)
        return

    # Training mode requires a dataset root
    if args.dataset_root is None:
        print("Error: --dataset-root is required for training.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Choose workflow based on flags
    if args.from_checkpoint is not None:
        # Workflow 3 — fine-tune from checkpoint
        model = train_from_checkpoint(
            checkpoint_path=args.from_checkpoint,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.low_level:
        # Workflow 2 — low-level PTL API
        train_low_level(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_variant=args.model,
            device=args.device,
        )
        return
    else:
        # Workflow 1 — high-level API (recommended)
        model = train_quickstart(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_variant=args.model,
            device=args.device,
            resolution=args.resolution,
        )

    # Run inference on provided images (or skip if none provided)
    if args.images:
        run_inference(model, args.images)


if __name__ == "__main__":
    main()
