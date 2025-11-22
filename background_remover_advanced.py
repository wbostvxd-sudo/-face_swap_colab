"""
Advanced Background Removal System
Multiple techniques for background removal
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class AdvancedBackgroundRemover:
    """Advanced background removal with multiple techniques"""
    
    def __init__(self):
        self.techniques = {
            'grabcut': self._grabcut_removal,
            'threshold': self._threshold_removal,
            'edge_based': self._edge_based_removal,
            'color_based': self._color_based_removal,
            'watershed': self._watershed_removal
        }
    
    def remove_background(
        self,
        image: np.ndarray,
        technique: str = 'grabcut',
        parameters: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove background from image
        
        Args:
            image: Input image
            technique: Removal technique
            parameters: Additional parameters
        
        Returns:
            Tuple of (foreground, mask)
        """
        if parameters is None:
            parameters = {}
        
        if technique not in self.techniques:
            technique = 'grabcut'
        
        return self.techniques[technique](image, parameters)
    
    def _grabcut_removal(self, img: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """GrabCut algorithm for background removal"""
        # Initialize mask
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle (can be improved with face detection)
        height, width = img.shape[:2]
        rect = params.get('rect', (int(width*0.1), int(height*0.1), 
                                   int(width*0.8), int(height*0.8)))
        
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 
                   params.get('iterations', 5), cv2.GC_INIT_WITH_RECT)
        
        # Create mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask
        result = img * mask2[:, :, np.newaxis]
        
        return result, mask2 * 255
    
    def _threshold_removal(self, img: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Threshold-based background removal"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        threshold = params.get('threshold', 127)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Invert if needed
        if params.get('invert', False):
            mask = cv2.bitwise_not(mask)
        
        # Apply mask
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return result, mask
    
    def _edge_based_removal(self, img: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Edge-based background removal"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, params.get('low_threshold', 50), 
                         params.get('high_threshold', 150))
        
        # Dilate edges
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=params.get('iterations', 3))
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return result, mask
    
    def _color_based_removal(self, img: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Color-based background removal"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color range (default: remove green screen)
        lower = np.array(params.get('lower_color', [40, 50, 50]))
        upper = np.array(params.get('upper_color', [80, 255, 255]))
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)  # Invert to keep foreground
        
        # Apply mask
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return result, mask
    
    def _watershed_removal(self, img: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Watershed algorithm for background removal"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(img, markers)
        
        # Create mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers > 1] = 255
        
        # Apply mask
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return result, mask
    
    def replace_background(
        self,
        image: np.ndarray,
        new_background: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Replace background with new image"""
        # Resize background to match image
        bg_resized = cv2.resize(new_background, (image.shape[1], image.shape[0]))
        
        # Normalize mask
        mask_normalized = mask.astype(np.float32) / 255.0
        if len(mask_normalized.shape) == 2:
            mask_normalized = np.stack([mask_normalized] * 3, axis=2)
        
        # Blend
        result = image.astype(np.float32) * mask_normalized + \
                 bg_resized.astype(np.float32) * (1 - mask_normalized)
        
        return result.astype(np.uint8)


def remove_background_simple(image: np.ndarray, method: str = 'grabcut') -> np.ndarray:
    """Simple background removal function"""
    remover = AdvancedBackgroundRemover()
    result, _ = remover.remove_background(image, method)
    return result

