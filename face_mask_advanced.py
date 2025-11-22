"""
Advanced Face Masking System
Multiple masking techniques for precise face region control
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class FaceMaskRegion(Enum):
    FACE = "face"
    EYEBROWS = "eyebrows"
    EYES = "eyes"
    NOSE = "nose"
    MOUTH = "mouth"
    CHEEKS = "cheeks"
    FOREHEAD = "forehead"
    CHIN = "chin"
    FULL = "full"


class AdvancedFaceMasker:
    """Advanced face masking with multiple techniques"""
    
    def __init__(self):
        self.mask_types = {
            'box': self._create_box_mask,
            'oval': self._create_oval_mask,
            'landmark_based': self._create_landmark_mask,
            'region_based': self._create_region_mask,
            'feather': self._create_feather_mask,
            'gaussian': self._create_gaussian_mask
        }
    
    def create_mask(
        self,
        image: np.ndarray,
        face_bbox: np.ndarray,
        mask_type: str = 'oval',
        parameters: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Create face mask
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            mask_type: Type of mask to create
            parameters: Additional parameters
        
        Returns:
            Binary mask
        """
        if parameters is None:
            parameters = {}
        
        if mask_type not in self.mask_types:
            mask_type = 'oval'
        
        return self.mask_types[mask_type](image, face_bbox, parameters)
    
    def _create_box_mask(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Create rectangular box mask"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Add padding if specified
        padding = params.get('padding', 0)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask
    
    def _create_oval_mask(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Create oval/elliptical mask"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox.astype(int)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Adjust size with padding
        padding = params.get('padding', 0)
        axes = (width // 2 + padding, height // 2 + padding)
        
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        return mask
    
    def _create_landmark_mask(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Create mask based on facial landmarks"""
        # Simplified version - would need actual landmark detection
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Create approximate face shape using landmarks positions
        # This is simplified - real implementation would use detected landmarks
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Approximate face shape (oval with adjustments)
        points = np.array([
            [center_x, y1],  # Top
            [x1, center_y - height // 4],  # Left cheek
            [x1 + width // 4, y2],  # Left chin
            [center_x, y2 + height // 10],  # Bottom chin
            [x2 - width // 4, y2],  # Right chin
            [x2, center_y - height // 4],  # Right cheek
        ], np.int32)
        
        cv2.fillPoly(mask, [points], 255)
        
        # Smooth edges
        blur_radius = params.get('blur_radius', 10)
        mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        return mask
    
    def _create_region_mask(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Create mask for specific face regions"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox.astype(int)
        
        regions = params.get('regions', [FaceMaskRegion.FACE])
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        for region in regions:
            if isinstance(region, str):
                region = FaceMaskRegion(region)
            
            if region == FaceMaskRegion.FACE or region == FaceMaskRegion.FULL:
                # Full face
                cv2.ellipse(mask, (center_x, center_y), 
                           (width // 2, height // 2), 0, 0, 360, 255, -1)
            elif region == FaceMaskRegion.EYES:
                # Eyes region
                eye_y = center_y - height // 4
                cv2.ellipse(mask, (center_x - width // 4, eye_y),
                           (width // 8, height // 12), 0, 0, 360, 255, -1)
                cv2.ellipse(mask, (center_x + width // 4, eye_y),
                           (width // 8, height // 12), 0, 0, 360, 255, -1)
            elif region == FaceMaskRegion.MOUTH:
                # Mouth region
                mouth_y = center_y + height // 4
                cv2.ellipse(mask, (center_x, mouth_y),
                           (width // 6, height // 12), 0, 0, 360, 255, -1)
            elif region == FaceMaskRegion.NOSE:
                # Nose region
                cv2.ellipse(mask, (center_x, center_y),
                           (width // 12, height // 3), 0, 0, 360, 255, -1)
        
        return mask
    
    def _create_feather_mask(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Create mask with feathered edges"""
        # Create base oval mask
        base_mask = self._create_oval_mask(image, bbox, params)
        
        # Apply feathering
        feather_radius = params.get('feather_radius', 20)
        mask = cv2.GaussianBlur(base_mask, 
                               (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        
        return mask
    
    def _create_gaussian_mask(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """Create mask with gaussian falloff"""
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        x1, y1, x2, y2 = bbox.astype(int)
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Create gaussian mask
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = np.exp(-((x - center_x)**2 / (2 * (width / 3)**2) + 
                       (y - center_y)**2 / (2 * (height / 3)**2)))
        
        # Normalize to 0-255
        mask = (mask * 255).astype(np.uint8)
        
        return mask
    
    def create_occlusion_mask(
        self,
        image: np.ndarray,
        face_bbox: np.ndarray
    ) -> np.ndarray:
        """Create mask for occluded face regions"""
        # Simplified occlusion detection
        mask = self._create_oval_mask(image, face_bbox, {})
        
        # Analyze image for occlusions (simplified)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = face_bbox.astype(int)
        face_region = gray[y1:y2, x1:x2]
        
        if face_region.size > 0:
            # Detect dark regions (potential occlusions)
            _, dark_mask = cv2.threshold(face_region, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Expand occlusion mask to full image
            full_occlusion = np.zeros_like(mask)
            full_occlusion[y1:y2, x1:x2] = dark_mask
            
            # Subtract occlusions from face mask
            mask = cv2.subtract(mask, full_occlusion)
        
        return mask
    
    def combine_masks(self, masks: List[np.ndarray], operation: str = 'union') -> np.ndarray:
        """Combine multiple masks"""
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        result = masks[0].copy()
        
        for mask in masks[1:]:
            if operation == 'union':
                result = cv2.bitwise_or(result, mask)
            elif operation == 'intersection':
                result = cv2.bitwise_and(result, mask)
            elif operation == 'subtract':
                result = cv2.subtract(result, mask)
        
        return result


def create_face_mask_simple(
    image: np.ndarray,
    bbox: np.ndarray,
    mask_type: str = 'oval'
) -> np.ndarray:
    """Simple function to create face mask"""
    masker = AdvancedFaceMasker()
    return masker.create_mask(image, bbox, mask_type)

