"""
Face Classification System
Classify faces by gender, age, race, and other attributes
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class Race(Enum):
    WHITE = "white"
    BLACK = "black"
    ASIAN = "asian"
    INDIAN = "indian"
    MIDDLE_EASTERN = "middle_eastern"
    LATINO_HISPANIC = "latino_hispanic"
    UNKNOWN = "unknown"


class FaceClassifier:
    """Classify faces by various attributes"""
    
    def __init__(self):
        self.age_ranges = [
            (0, 2, "baby"),
            (3, 9, "child"),
            (10, 19, "teen"),
            (20, 29, "twenties"),
            (30, 39, "thirties"),
            (40, 49, "forties"),
            (50, 59, "fifties"),
            (60, 69, "sixties"),
            (70, 79, "seventies"),
            (80, 100, "elderly")
        ]
    
    def classify_face_simple(
        self,
        face_image: np.ndarray,
        face_bbox: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Simple face classification using heuristics
        
        Args:
            face_image: Face image or full image
            face_bbox: Bounding box if full image provided
        
        Returns:
            Dictionary with classification results
        """
        # Extract face region if bbox provided
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox.astype(int)
            face_region = face_image[y1:y2, x1:x2]
        else:
            face_region = face_image
        
        if face_region.size == 0:
            return {
                'gender': Gender.UNKNOWN.value,
                'age_range': 'unknown',
                'age_estimate': 0,
                'race': Race.UNKNOWN.value,
                'confidence': 0.0
            }
        
        # Simple heuristics-based classification
        # Note: This is a simplified version. Real implementation would use ML models
        
        # Analyze face region
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Estimate age based on skin texture and features
        age_estimate = self._estimate_age(gray)
        age_range = self._get_age_range(age_estimate)
        
        # Estimate gender (simplified - would need ML model for accuracy)
        gender = self._estimate_gender(face_region)
        
        # Estimate race (simplified - would need ML model for accuracy)
        race = self._estimate_race(face_region)
        
        return {
            'gender': gender,
            'age_range': age_range,
            'age_estimate': age_estimate,
            'race': race,
            'confidence': 0.6  # Low confidence for heuristic methods
        }
    
    def _estimate_age(self, gray_face: np.ndarray) -> int:
        """Estimate age based on texture analysis"""
        # Analyze skin texture
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # More texture = older (simplified heuristic)
        if laplacian_var < 100:
            return np.random.randint(0, 20)  # Young
        elif laplacian_var < 200:
            return np.random.randint(20, 40)  # Adult
        elif laplacian_var < 300:
            return np.random.randint(40, 60)  # Middle-aged
        else:
            return np.random.randint(60, 80)  # Elderly
    
    def _get_age_range(self, age: int) -> str:
        """Get age range category"""
        for min_age, max_age, label in self.age_ranges:
            if min_age <= age <= max_age:
                return label
        return "unknown"
    
    def _estimate_gender(self, face_region: np.ndarray) -> str:
        """Estimate gender (simplified heuristic)"""
        # This is a placeholder - real implementation needs ML model
        # Analyze facial features (jawline, cheekbones, etc.)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Simple heuristic based on face shape
        height, width = gray.shape
        aspect_ratio = height / width if width > 0 else 1.0
        
        # More square faces often associated with males (very simplified)
        if aspect_ratio > 1.3:
            return Gender.FEMALE.value
        else:
            return Gender.MALE.value
    
    def _estimate_race(self, face_region: np.ndarray) -> str:
        """Estimate race (simplified heuristic)"""
        # This is a placeholder - real implementation needs ML model
        # Analyze skin tone and facial features
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        # Very simplified heuristic (not accurate)
        if avg_val < 100:
            return Race.BLACK.value
        elif avg_hue < 20:
            return Race.ASIAN.value
        elif avg_hue < 30:
            return Race.WHITE.value
        else:
            return Race.UNKNOWN.value
    
    def classify_multiple_faces(
        self,
        image: np.ndarray,
        faces: List
    ) -> List[Dict]:
        """Classify multiple faces in an image"""
        classifications = []
        
        for face in faces:
            if hasattr(face, 'bbox'):
                classification = self.classify_face_simple(image, face.bbox)
                classifications.append(classification)
        
        return classifications


def get_face_statistics(classifications: List[Dict]) -> Dict:
    """Get statistics from face classifications"""
    if not classifications:
        return {}
    
    genders = [c.get('gender', 'unknown') for c in classifications]
    ages = [c.get('age_estimate', 0) for c in classifications]
    races = [c.get('race', 'unknown') for c in classifications]
    
    return {
        'total_faces': len(classifications),
        'gender_distribution': {g: genders.count(g) for g in set(genders)},
        'average_age': np.mean(ages) if ages else 0,
        'age_range': (min(ages), max(ages)) if ages else (0, 0),
        'race_distribution': {r: races.count(r) for r in set(races)}
    }

