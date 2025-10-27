# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DepthMetrics
from ultralytics.utils.plotting import plot_images


class DepthValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a depth estimation model.

    This validator handles the evaluation of depth estimation models, processing depth map predictions
    to compute metrics such as MAE, RMSE, AbsRel, and depth accuracy thresholds.

    Attributes:
        metrics (DepthMetrics): Metrics calculator for depth estimation tasks.
        args (SimpleNamespace): Configuration arguments with task set to 'depth'.

    Methods:
        preprocess: Preprocess batch by normalizing images and moving depth to device.
        postprocess: Postprocess depth predictions (no NMS needed for regression).
        init_metrics: Initialize depth estimation metrics.
        update_metrics: Update metrics based on depth predictions and ground truth.
        finalize_metrics: Finalize and compute all depth metrics.
        get_stats: Return computed depth statistics.
        get_desc: Get description of depth estimation metrics.
        print_results: Print depth estimation evaluation results.
        plot_val_samples: Plot validation samples with ground truth depth maps.
        plot_predictions: Plot predicted depth maps.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthValidator
        >>> args = dict(model='yolo11n-depth.pt', data='depth.yaml')
        >>> validator = DepthValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize DepthValidator with depth-specific configurations.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"
        self.metrics = DepthMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess batch for depth validation.

        Normalizes images from [0,255] to [0,1] and moves depth data to device.
        Note: Depth is already Per-image normalized to [0,1] by Format class.

        Args:
            batch (dict[str, Any]): Batch containing images and depth maps (already normalized to [0,1]).

        Returns:
            (dict[str, Any]): Preprocessed batch with normalized images and depth on device.
        """
        # Determine the dtype: use half if specified in args, otherwise float32
        dtype = torch.half if self.args.half else torch.float32
        
        # Convert image to the correct device and dtype
        batch["img"] = batch["img"].to(self.device, non_blocking=True).to(dtype) / 255.0
        
        if "depth" in batch:
            # Depth is already Per-image normalized to [0, 1] by Format class
            batch["depth"] = batch["depth"].to(self.device).float()  # Keep depth as float32 for metrics
            
            # Store per-image min/max for denormalization during metric computation
            # These are already in the batch from Format class
            if "depth_min" not in batch:
                batch["depth_min"] = 0.0  # Fallback
            if "depth_max" not in batch:
                batch["depth_max"] = 1.0  # Fallback
        
        return batch

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Postprocess depth predictions.

        Keep predictions in normalized [0,1] scale for metric computation.
        Conversion to [0,255] happens during visualization only.

        Args:
            preds (torch.Tensor): Raw depth predictions from model (normalized to [0, 1]).

        Returns:
            (torch.Tensor): Depth predictions in normalized [0,1] scale for metrics.
        """
        # preds shape: (batch_size, 1, height, width)
        # Model outputs are already in [0, 1] range (Per-image normalized)
        # Keep in [0,1] for metrics computation
        if isinstance(preds, list):
            preds = preds[0]
        
        preds = preds.detach()
        
        # Keep in [0, 1] range for metric computation
        # Visualization will convert to [0, 255] when needed
        return preds

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize depth estimation metrics.

        Args:
            model (torch.nn.Module): Model being validated.
        """
        self.metrics = DepthMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """
        Update depth estimation metrics with predictions and ground truth.

        Args:
            preds (torch.Tensor): Model predictions with shape (B, 1, H, W), already in [0,1] normalized range.
            batch (dict[str, Any]): Batch containing ground truth depth (normalized [0,1]) and image info.
        """
        if "depth" not in batch:
            LOGGER.warning("No depth ground truth in batch, skipping metrics update")
            return

        # Ensure predictions and ground truth have compatible shapes
        pred_depth = preds  # (B, 1, H, W) - normalized [0, 1]
        gt_depth = batch["depth"]  # (B, 1, H, W) or (B, H, W) - normalized [0, 1]

        # Handle dimension mismatches
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(1)

        # Resize predictions to match ground truth if needed
        if pred_depth.shape != gt_depth.shape:
            pred_depth = F.interpolate(
                pred_depth,
                size=gt_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Process each sample in batch
        # Both pred and gt are in normalized [0, 1] scale (Per-image normalized)
        # Compute metrics in normalized scale for consistency
        # This makes MAE/RMSE values directly interpretable as relative errors
        
        for i in range(pred_depth.shape[0]):
            # Both are already in [0, 1] range from Per-image normalization
            pred = pred_depth[i].squeeze().cpu().numpy()  # (H, W) - [0, 1] normalized
            gt = gt_depth[i].squeeze().cpu().numpy()  # (H, W) - [0, 1] normalized
            
            # Compute metrics directly on normalized scale [0, 1]
            self.metrics.process(pred, gt)

    def finalize_metrics(self) -> None:
        """
        Finalize depth estimation metrics and prepare for output.

        Computes final statistics and generates plots if configured.
        """
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, float]:
        """
        Return computed depth estimation statistics.

        Returns:
            (dict[str, float]): Dictionary containing depth metrics (MAE, RMSE, AbsRel, delta thresholds).
        """
        return self.metrics.results_dict

    def get_desc(self) -> str:
        """
        Return formatted description of depth estimation metrics.

        Returns:
            (str): Formatted string with metric names and widths.
        """
        return ("%22s" + "%11s" * 6) % ("Metric", "MAE", "RMSE", "AbsRel", "δ<1.25", "δ<1.25²", "δ<1.25³")

    def print_results(self) -> None:
        """Print depth estimation validation results."""
        stats = self.get_stats()

        # Print in compact single-line format matching the header from get_desc()
        pf = "%22s" + "%11.4f" * 6  # print format
        LOGGER.info(pf % ("all", stats.get('mae', 0), stats.get('rmse', 0), stats.get('abs_rel', 0),
                          stats.get('delta1', 0), stats.get('delta2', 0), stats.get('delta3', 0)))

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset for depth validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode.
            batch (int, optional): Size of batches for rectangular training.

        Returns:
            (Dataset): YOLO dataset for depth estimation.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader for depth validation.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        Plot validation samples with ground truth depth maps.

        Args:
            batch (dict[str, Any]): Batch containing images and depth ground truth (normalized [0,1]).
            ni (int): Batch index for file naming.
        """
        if "depth" in batch:
            # GT depth is already Per-image normalized to [0, 1]
            # Convert to [0, 255] for visualization
            gt_depth = batch["depth"]  # normalized [0, 1]
            if isinstance(gt_depth, torch.Tensor):
                gt_depth = gt_depth.clone()
            else:
                gt_depth = torch.from_numpy(gt_depth).float()
            
            # Convert [0, 1] to [0, 255] for visualization
            gt_depth_vis = gt_depth * 255.0
            gt_depth_vis = torch.clamp(gt_depth_vis, 0.0, 255.0)
            
            plot_images(
                labels={
                    "img": batch["img"],
                    "depths": gt_depth_vis,
                    "im_file": batch.get("im_file", []),
                },
                paths=batch.get("im_file"),
                fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                names={0: "ground_truth_depth"},
                on_plot=self.on_plot,
                depth_range=(0.0, 255.0),  # Visualization range [0, 255]
            )

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """
        Plot predicted depth maps for visualization.

        Args:
            batch (dict[str, Any]): Batch containing images.
            preds (torch.Tensor): Predicted depth maps with shape (B, 1, H, W).
                                 In normalized [0,1] scale (for metrics computation).
            ni (int): Batch index for file naming.
        """
        # Convert predictions from [0,1] to [0,255] for visualization
        preds_vis = preds * 255.0
        preds_vis = torch.clamp(preds_vis, 0.0, 255.0)
        
        plot_images(
            labels={
                "img": batch["img"],
                "depths": preds_vis,
                "im_file": batch.get("im_file", []),
            },
            paths=batch.get("im_file"),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names={0: "predicted_depth"},
            on_plot=self.on_plot,
            depth_range=(0.0, 255.0),  # Visualization range [0, 255]
        )