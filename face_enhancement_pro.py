"""
Professional face enhancement system
Multiple algorithms and advanced techniques
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import ndimage


class ProFaceEnhancer:
    """Professional face enhancement system with multiple techniques"""
    
    def __init__(self):
        self.available_techniques = {
            'advanced_sharpen': self._advanced_sharpen,
            'noise_reduction': self._noise_reduction,
            'color_balance': self._color_balance,
            'smart_smooth': self._smart_smooth,
            'detail_enhance': self._detail_enhance,
            'illumination_correction': self._illumination_correction,
            'contrast_enhance': self._contrast_enhance,
            'bilateral_filter': self._bilateral_filter
        }
    
    def apply_enhancement(
        self,
        image: np.ndarray,
        technique: str = 'advanced_sharpen',
        intensity: float = 0.5,
        extra_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Apply enhancement using specified technique
        
        Args:
            image: Input image
            technique: Name of technique to apply
            intensity: Enhancement intensity (0-1)
            extra_params: Additional parameters specific to technique
        """
        if technique not in self.available_techniques:
            return image
        
        if extra_params is None:
            extra_params = {}
        
        return self.available_techniques[technique](image, intensity, extra_params)
    
    def _advanced_sharpen(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Advanced sharpening using unsharp masking"""
        img_float = img.astype(np.float32)
        
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(img_float, (0, 0), 1.0 + intensity * 2)
        unsharp_mask = img_float - gaussian
        
        # Apply with controlled intensity
        result = img_float + unsharp_mask * intensity * 2
        
        # Normalize and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def _noise_reduction(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Noise reduction using multiple techniques"""
        h = int(10 * intensity)
        h_color = int(10 * intensity)
        
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h_color, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
    
    def _color_balance(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Automatic color balance"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.0 + intensity, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Recombine
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Blend with original according to intensity
        return cv2.addWeighted(img, 1 - intensity, result, intensity, 0)
    
    def _smart_smooth(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Smart smoothing preserving edges"""
        d = int(5 + intensity * 10)
        sigma_color = 50 + intensity * 50
        sigma_space = 50 + intensity * 50
        
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    def _detail_enhance(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Detail enhancement using high frequency filters"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply enhancement only in edge regions
        if len(img.shape) == 3:
            result = img.copy()
            for i in range(3):
                channel = result[:, :, i]
                channel[edges > 0] = np.clip(channel[edges > 0] * (1 + intensity), 0, 255)
                result[:, :, i] = channel
            return result
        else:
            result = img.copy()
            result[edges > 0] = np.clip(result[edges > 0] * (1 + intensity), 0, 255)
            return result
    
    def _illumination_correction(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Automatic illumination correction"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply adaptive gamma correction
        gamma = 1.0 - intensity * 0.3
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l_corrected = cv2.LUT(l, table)
        
        # Recombine
        lab = cv2.merge([l_corrected, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return cv2.addWeighted(img, 1 - intensity * 0.5, result, intensity * 0.5, 0)
    
    def _contrast_enhance(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Adaptive contrast enhancement"""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0 + intensity * 2, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=1.0 + intensity * 2, tileGridSize=(8, 8))
            result = clahe.apply(img)
        
        return cv2.addWeighted(img, 1 - intensity, result, intensity, 0)
    
    def _bilateral_filter(self, img: np.ndarray, intensity: float, params: Dict) -> np.ndarray:
        """Advanced bilateral filter"""
        d = int(5 + intensity * 15)
        sigma_color = 75
        sigma_space = 75
        
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    def apply_enhancement_to_face_region(
        self,
        image: np.ndarray,
        face_bbox: np.ndarray,
        technique: str = 'advanced_sharpen',
        intensity: float = 0.5,
        blend_factor: float = 0.8
    ) -> np.ndarray:
        """
        Apply enhancement only to face region
        
        Args:
            image: Full image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            technique: Enhancement technique to apply
            intensity: Enhancement intensity
            blend_factor: Blend factor with original (0-1)
        """
        result = image.copy()
        x1, y1, x2, y2 = face_bbox.astype(int)
        
        # Ensure indices are within image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return result
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return result
        
        # Apply enhancement
        enhanced_region = self.apply_enhancement(face_region, technique, intensity)
        
        # Blend with original
        blended_region = cv2.addWeighted(
            face_region, 1 - blend_factor,
            enhanced_region, blend_factor,
            0
        )
        
        # Paste back
        result[y1:y2, x1:x2] = blended_region
        
        return result
    
    def apply_multiple_enhancements(
        self,
        image: np.ndarray,
        techniques: List[str],
        intensities: List[float]
    ) -> np.ndarray:
        """Apply multiple enhancement techniques in sequence"""
        result = image.copy()
        
        for technique, intensity in zip(techniques, intensities):
            if technique in self.available_techniques:
                result = self.apply_enhancement(result, technique, intensity)
        
        return result


def enhance_faces_in_image(
    image: np.ndarray,
    faces: List,
    technique: str = 'advanced_sharpen',
    intensity: float = 0.5
) -> np.ndarray:
    """
    Apply enhancement to all faces in an image
    
    Args:
        image: Input image
        faces: List of detected faces
        technique: Enhancement technique
        intensity: Enhancement intensity
    """
    enhancer = ProFaceEnhancer()
    result = image.copy()
    
    for face in faces:
        if hasattr(face, 'bbox'):
            bbox = face.bbox
            result = enhancer.apply_enhancement_to_face_region(
                result, bbox, technique, intensity
            )
    
    return result

