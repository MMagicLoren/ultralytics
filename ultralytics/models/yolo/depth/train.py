# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.utils.plotting import plot_images, plot_results


class DepthTrainer(BaseTrainer):
    """
    A trainer class for depth estimation models based on YOLO.
    
    This trainer is specifically designed for monocular depth estimation tasks, 
    extending BaseTrainer to handle depth prediction where outputs are continuous
    depth maps rather than discrete class predictions or bounding boxes.
    
    Attributes:
        model (DepthModel): The YOLO depth estimation model being trained.
        data (dict): Dictionary containing dataset information.
        loss_names (tuple): Names of the loss components (l1_loss, si_loss, grad_loss).
        validator (DepthValidator): Validator instance for model evaluation.
    
    Example:
    ```python
        from ultralytics.models.yolo.depth import DepthTrainer
        
        args = dict(model='yolo11n-depth.yaml', data='depth.yaml', epochs=100)
        trainer = DepthTrainer(overrides=args)
        trainer.train()
    ```
    """

    def __init__(self, cfg: dict | str = DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """
        Initialize DepthTrainer object with custom configurations.
        
        Args:
            cfg (dict | str): Configuration file path or dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "depth"
        super().__init__(cfg, overrides, _callbacks)
    
    def set_model_attributes(self):
        """Set the model attributes like input channels, output depth channels, etc."""
        self.model.names = {0: "depth"}  # Depth has single output channel

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO dataset.
        
        Args:
            img_path (str): Path to dataset directory.
            mode (str): Dataset mode ('train' or 'val').
            batch (int, optional): Batch size for DataLoader.
        
        Returns:
            (Dataset): Built dataset instance.
        """
        gs = max(int(self.model.stride.max()), 32)  # grid size
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            stride=gs,
        )
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader.
        
        Args:
            dataset_path (str): Path to dataset.
            batch_size (int): Batch size for DataLoader.
            rank (int): Rank for distributed training.
            mode (str): Dataset mode ('train' or 'val').
        
        Returns:
            (DataLoader): PyTorch DataLoader instance.
        """
        assert mode in ["train", "val"], f"Mode should be 'train' or 'val', not {mode}"
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once, before DDP spawning
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        shuffle = mode == "train"
        if self.args.rect and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        loader = build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=shuffle,
            rank=rank,
        )
        return loader
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch by moving data to device and applying dtype conversion.
        
        Note: Depth normalization is already done in Format class (Per-image normalization to [0,1]).
        
        Args:
            batch (dict): Batch data containing images and depth maps (already normalized to [0,1]).
        
        Returns:
            (dict): Preprocessed batch data.
        """
        # Determine the dtype based on AMP setting
        dtype = torch.half if self.args.half else torch.float32
        
        # Convert image to the correct device and dtype
        batch["img"] = batch["img"].to(self.device).to(dtype) / 255.0  # normalize images to [0, 1]
        
        if "depth" in batch:
            # Depth is already normalized to [0, 1] by Format class (Per-image normalization)
            batch["depth"] = batch["depth"].to(self.device).float()  # Keep depth as float32 for loss computation
        
        return batch

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """
        Return DepthModel initialized with specified config and weights.
        
        Args:
            cfg (dict | str | None): Model configuration.
            weights (str | Path | None): Pre-trained weights path.
            verbose (bool): Whether to print model info.
        
        Returns:
            (DepthModel): Initialized depth model.
        """
        model = DepthModel(
            cfg or self.args.model,
            ch=3,
            nc=self.data.get("nc", 1),
            verbose=verbose and RANK == -1,
        )
        
        # Set depth-specific hyperparameters from trainer args
        model.args.update({
            "imgsz": self.args.imgsz,
            "depth_min": self.data.get("depth_min", 0.0),
            "depth_max": self.data.get("depth_max", 1.0),
            "lambda_l1": self.args.get("lambda_l1", 1.0),
            "lambda_si": self.args.get("lambda_si", 0.5),
            "lambda_grad": self.args.get("lambda_grad", 0.1),
        })
        
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """
        Return DepthValidator instance for model validation.
        
        Returns:
            (DepthValidator): Depth validation instance.
        """
        self.loss_names = "l1_loss", "si_loss", "grad_loss"
        from ultralytics.models.yolo.depth import DepthValidator
        
        return DepthValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss and size."""
        return ("\n" + "%11s" * (3 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Size",
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled depth training loss items.
        
        Args:
            loss_items (torch.Tensor, optional): Tensor of loss values (l1_loss, si_loss, grad_loss).
            prefix (str): Prefix for loss labels ('train' or 'val').
        
        Returns:
            (dict): Dictionary with labelled loss items if loss_items is provided, else list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert to float
            return dict(zip(keys, loss_items))
        else:
            return keys

    def plot_training_samples(self, batch, ni):
        """
        Plot training samples with their depth map annotations.
        
        Args:
            batch (dict): Batch data containing images and depth maps.
            ni (int): Batch index number.
        """
        # Get depth range for visualization
        depth_min = self.data.get("depth_min", 0.0)
        depth_max = self.data.get("depth_max", 255.0)
        
        # Denormalize GT depth to original range for visualization
        import torch
        gt_depth = batch.get("depth")
        if gt_depth is not None:
            if isinstance(gt_depth, torch.Tensor):
                gt_depth = gt_depth.clone()
            if depth_max > depth_min:
                gt_depth = gt_depth * (depth_max - depth_min) + depth_min
        
        plot_images(
            labels={
                "img": batch["img"],
                "depths": gt_depth,
                "im_file": batch.get("im_file", []),
            },
            paths=batch.get("im_file"),
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
            depth_range=(depth_min, depth_max),  # Fixed range for consistent visualization
        )

    def plot_metrics(self):
        """Plot training metrics from the results CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)