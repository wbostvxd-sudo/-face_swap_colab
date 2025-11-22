"""
Advanced face blending and mixing system
Professional face integration techniques
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import ndimage


class AdvancedFaceBlender:
    """Advanced face blending system with multiple techniques"""
    
    def __init__(self):
        self.blending_techniques = {
            'feather': self._feather_blending,
            'poisson': self._poisson_blending,
            'seamless': self._seamless_clone,
            'multiband': self._multiband_blending,
            'gaussian': self._gaussian_blending,
            'linear': self._linear_blending
        }
    
    def blend_face(
        self,
        base_image: np.ndarray,
        swapped_face: np.ndarray,
        mask: np.ndarray,
        technique: str = 'feather',
        parameters: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Blend a swapped face into base image
        
        Args:
            base_image: Base image where face will be pasted
            swapped_face: Already processed face to paste
            mask: Binary mask of face area
            technique: Blending technique to use
            parameters: Additional parameters for technique
        """
        if technique not in self.blending_techniques:
            technique = 'feather'
        
        if parameters is None:
            parameters = {}
        
        return self.blending_techniques[technique](
            base_image, swapped_face, mask, parameters
        )
    
    def _feather_blending(
        self,
        base: np.ndarray,
        face: np.ndarray,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Feather blending - smooth blend with gradient"""
        feather_radius = params.get('feather_radius', 10)
        
        # Create mask with gradient
        mask_float = mask.astype(np.float32) / 255.0
        smooth_mask = cv2.GaussianBlur(mask_float, (0, 0), feather_radius)
        smooth_mask = np.clip(smooth_mask * 2, 0, 1)
        
        # Apply blend
        if len(base.shape) == 3:
            mask_3d = np.stack([smooth_mask] * 3, axis=2)
        else:
            mask_3d = smooth_mask
        
        result = base.copy().astype(np.float32)
        face_float = face.astype(np.float32)
        
        result = result * (1 - mask_3d) + face_float * mask_3d
        return result.astype(np.uint8)
    
    def _poisson_blending(
        self,
        base: np.ndarray,
        face: np.ndarray,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Poisson blending using seamless clone"""
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return base
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        center = (cx, cy)
        
        flags = params.get('flags', cv2.NORMAL_CLONE)
        return cv2.seamlessClone(face, base, mask, center, flags)
    
    def _seamless_clone(
        self,
        base: np.ndarray,
        face: np.ndarray,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Improved seamless cloning"""
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return base
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        center = (cx, cy)
        
        return cv2.seamlessClone(face, base, mask, center, cv2.MIXED_CLONE)
    
    def _multiband_blending(
        self,
        base: np.ndarray,
        face: np.ndarray,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Simplified multiband blending"""
        levels = params.get('levels', 3)
        
        # Create Laplacian pyramids
        base_pyr = [base.astype(np.float32)]
        face_pyr = [face.astype(np.float32)]
        mask_pyr = [mask.astype(np.float32) / 255.0]
        
        # Build pyramids
        for i in range(levels - 1):
            base_pyr.append(cv2.pyrDown(base_pyr[-1]))
            face_pyr.append(cv2.pyrDown(face_pyr[-1]))
            mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))
        
        # Blend at each level
        result_pyr = []
        for i in range(levels - 1, -1, -1):
            if i == levels - 1:
                mask_3d = np.stack([mask_pyr[i]] * 3, axis=2) if len(base_pyr[i].shape) == 3 else mask_pyr[i]
                blended = base_pyr[i] * (1 - mask_3d) + face_pyr[i] * mask_3d
                result_pyr.append(blended)
            else:
                expanded = cv2.pyrUp(result_pyr[-1], dstsize=(base_pyr[i].shape[1], base_pyr[i].shape[0]))
                mask_3d = np.stack([mask_pyr[i]] * 3, axis=2) if len(base_pyr[i].shape) == 3 else mask_pyr[i]
                blended = base_pyr[i] * (1 - mask_3d) + face_pyr[i] * mask_3d
                result_pyr.append(blended)
        
        result = np.clip(result_pyr[-1], 0, 255).astype(np.uint8)
        return result
    
    def _gaussian_blending(
        self,
        base: np.ndarray,
        face: np.ndarray,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Blending using gaussian filter"""
        sigma = params.get('sigma', 5.0)
        
        smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) / 255.0
        
        if len(base.shape) == 3:
            mask_3d = np.stack([smooth_mask] * 3, axis=2)
        else:
            mask_3d = smooth_mask
        
        result = base.astype(np.float32) * (1 - mask_3d) + face.astype(np.float32) * mask_3d
        return result.astype(np.uint8)
    
    def _linear_blending(
        self,
        base: np.ndarray,
        face: np.ndarray,
        mask: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Simple linear blending"""
        alpha = params.get('alpha', 0.8)
        
        mask_norm = (mask.astype(np.float32) / 255.0) * alpha
        
        if len(base.shape) == 3:
            mask_3d = np.stack([mask_norm] * 3, axis=2)
        else:
            mask_3d = mask_norm
        
        result = base.astype(np.float32) * (1 - mask_3d) + face.astype(np.float32) * mask_3d
        return result.astype(np.uint8)


def create_oval_face_mask(bbox: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Create an oval mask for face"""
    mask = np.zeros(image_size, dtype=np.uint8)
    
    x1, y1, x2, y2 = bbox.astype(int)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    width = x2 - x1
    height = y2 - y1
    
    # Create ellipse
    axes = (width // 2, height // 2)
    cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
    
    return mask


def create_smooth_face_mask(bbox: np.ndarray, image_size: Tuple[int, int], smooth_radius: int = 20) -> np.ndarray:
    """Create a mask with smoothed edges"""
    mask = create_oval_face_mask(bbox, image_size)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (smooth_radius * 2 + 1, smooth_radius * 2 + 1), 0)
    
    return mask

