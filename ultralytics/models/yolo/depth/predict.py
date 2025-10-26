# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class DepthPredictor(BasePredictor):
    """
    Predictor class for depth estimation.
    
    This class extends BasePredictor to perform monocular depth estimation on images,
    converting model outputs (depth maps) into Results objects for visualization and analysis.
    
    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO depth estimation model.
    
    Example:
    ```python
        from ultralytics.models.yolo.depth import DepthPredictor
        
        args = dict(model='yolo11n-depth.pt', source='image.jpg')
        predictor = DepthPredictor(overrides=args)
        predictor.predict_cli()
    ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize DepthPredictor with custom configurations.
        
        Args:
            cfg (dict): Default configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "depth"

    def postprocess(self, preds, img, orig_imgs):
        """
        Post-process depth predictions and return Results objects.
        
        Args:
            preds (torch.Tensor): Predicted depth maps from the model, shape (B, 1, H, W) or (B, C, H, W).
            img (torch.Tensor): Preprocessed input images.
            orig_imgs (list | np.ndarray): Original input images.
        
        Returns:
            (list[Results]): List of Results objects containing depth maps.
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # Get depth range for denormalization (from model args or defaults)
        depth_min = getattr(self.model, 'args', {}).get('depth_min', 0.0)
        depth_max = getattr(self.model, 'args', {}).get('depth_max', 255.0)

        results = []
        for i, depth_map in enumerate(preds):
            orig_img = orig_imgs[i]
            orig_shape = orig_img.shape[:2]
            
            # Handle different depth map dimensions
            if depth_map.dim() == 3:
                # If shape is (C, H, W), ensure C=1 and squeeze it
                if depth_map.shape[0] != 1:
                    depth_map = depth_map[:1, :, :]  # Take first channel if multiple
                depth_map = depth_map.squeeze(0)  # Remove channel dimension
            
            # Resize depth map to original image size if needed
            if depth_map.shape != orig_shape:
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(0).unsqueeze(0),
                    size=orig_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            
            # Denormalize depth from [0, 1] to original range [depth_min, depth_max]
            # This ensures consistent visualization with val_batch_pred.jpg
            if depth_max > depth_min:
                depth_map = depth_map * (depth_max - depth_min) + depth_min
            
            # Convert to numpy
            depth_map_np = depth_map.detach().cpu().numpy()
            
            # Get image path
            img_path = self.batch[0][i] if self.batch and self.batch[0] else ""
            
            # Create Results object with depth data
            results.append(
                Results(
                    orig_img=orig_img,
                    path=img_path,
                    names={0: "depth"},
                    depth=depth_map_np,
                )
            )
        
        return results