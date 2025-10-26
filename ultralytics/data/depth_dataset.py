# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.utils import LOGGER, colorstr


class DepthDataset(Dataset):
    """
    Simple dataset for depth estimation tasks.
    
    Loads RGB images and corresponding depth maps for monocular depth estimation.
    Supports both image and depth map resizing to target size.
    
    Attributes:
        img_path (Path): Path to images directory.
        depth_path (Path): Path to depth maps directory (.npy files).
        imgsz (int): Target image size for training.
        augment (bool): Whether to apply augmentations.
        samples (list): List of (image_file, depth_file) tuples.
    
    Examples:
        >>> dataset = DepthDataset(img_path="path/to/images", depth_path="path/to/depths")
        >>> sample = dataset[0]  # Returns dict with 'img', 'depth', 'im_file'
    """
    
    def __init__(
        self,
        img_path: str | Path,
        depth_path: str | Path,
        imgsz: int = 640,
        augment: bool = False,
    ):
        """
        Initialize DepthDataset.
        
        Args:
            img_path (str | Path): Path to images directory.
            depth_path (str | Path): Path to depth maps directory (*.npy files).
            imgsz (int): Target image size. Default: 640.
            augment (bool): Whether to apply augmentations. Default: False.
        
        Raises:
            ValueError: If no image-depth pairs found.
        """
        self.img_path = Path(img_path)
        self.depth_path = Path(depth_path)
        self.imgsz = imgsz
        self.augment = augment
        
        # Get all image files (jpg and png)
        img_files = sorted(self.img_path.glob("*.jpg")) + sorted(self.img_path.glob("*.png"))
        
        # Verify depth files exist and create sample pairs
        self.samples = []
        for img_file in img_files:
            depth_file = self.depth_path / f"{img_file.stem}.npy"
            if depth_file.exists():
                self.samples.append((img_file, depth_file))
        
        if not self.samples:
            raise ValueError(
                f"No image-depth pairs found. "
                f"Images: {self.img_path}, Depths: {self.depth_path}"
            )
        
        LOGGER.info(
            f"Loaded {len(self.samples)} depth estimation samples from {colorstr('bold', self.img_path)}"
        )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample (image + depth map).
        
        Args:
            idx (int): Sample index.
        
        Returns:
            (dict): Dictionary containing:
                - 'img' (torch.Tensor): Image tensor (3, H, W) in range [0, 1]
                - 'depth' (torch.Tensor): Depth map tensor (1, H, W)
                - 'im_file' (str): Path to image file
        
        Raises:
            FileNotFoundError: If image or depth file cannot be loaded.
        """
        img_file, depth_file = self.samples[idx]
        
        # Load image (BGR)
        img = cv2.imread(str(img_file))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_file}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load depth map (.npy file)
        try:
            depth = np.load(str(depth_file)).astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load depth map: {depth_file}. Error: {e}")
        
        # Ensure depth is 2D (H, W)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        elif depth.ndim != 2:
            raise ValueError(f"Invalid depth shape: {depth.shape}. Expected (H, W) or (H, W, 1)")
        
        # Apply augmentations
        if self.augment:
            img, depth = self._apply_augmentations(img, depth)
        
        # Resize to target size
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensors and normalize
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() / 255.0
        depth = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).float()
        
        return {
            "img": img,  # (3, H, W) in [0, 1]
            "depth": depth,  # (1, H, W)
            "im_file": str(img_file),
        }
    
    def _apply_augmentations(self, img: np.ndarray, depth: np.ndarray) -> tuple:
        """
        Apply augmentations to image and depth map.
        
        Args:
            img (np.ndarray): Input image (H, W, 3).
            depth (np.ndarray): Input depth map (H, W).
        
        Returns:
            (tuple): Augmented (img, depth) pair.
        """
        h, w = img.shape[:2]
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            depth = cv2.flip(depth, 1)
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random rotation (small angle)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            depth = cv2.warpAffine(depth, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return img, depth
    
    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Collate batch of samples.
        
        Args:
            batch (list): List of samples from __getitem__.
        
        Returns:
            (dict): Batched data with stacked tensors.
        """
        return {
            "img": torch.stack([sample["img"] for sample in batch]),  # (B, 3, H, W)
            "depth": torch.stack([sample["depth"] for sample in batch]),  # (B, 1, H, W)
            "im_file": [sample["im_file"] for sample in batch],
        }
