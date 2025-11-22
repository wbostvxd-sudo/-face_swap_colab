"""
Advanced face detection and analysis module
Multi-model detection system with enhanced strategies
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import insightface
from insightface.app import FaceAnalysis


class MultiModelFaceDetector:
    """Face detector with support for multiple models"""
    
    def __init__(self, base_model='buffalo_l', detection_size=(320, 320)):
        """
        Initialize multi-model detector
        
        Args:
            base_model: Base InsightFace model
            detection_size: Detection size (smaller = faster)
        """
        self.base_model = base_model
        self.detection_size = detection_size
        self.app = FaceAnalysis(name=base_model)
        self.app.prepare(ctx_id=0, det_size=detection_size)
        self.detection_cache = {}
    
    def detect_faces(self, image: np.ndarray, use_cache: bool = True) -> List:
        """Detect faces with optional cache"""
        if use_cache:
            img_hash = hash(image.tobytes())
            if img_hash in self.detection_cache:
                return self.detection_cache[img_hash]
        
        faces = self.app.get(image)
        
        if use_cache:
            self.detection_cache[img_hash] = faces
        
        return faces
    
    def get_face_statistics(self, image: np.ndarray) -> Dict:
        """Get detailed statistics of detected faces"""
        faces = self.detect_faces(image)
        
        if not faces:
            return {
                'total': 0,
                'areas': [],
                'positions': [],
                'scores': []
            }
        
        statistics = {
            'total': len(faces),
            'areas': [],
            'positions': [],
            'scores': [],
            'centers': []
        }
        
        for face in faces:
            bbox = face.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            statistics['areas'].append(area)
            statistics['positions'].append({
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3])
            })
            statistics['centers'].append({'x': center_x, 'y': center_y})
            
            if hasattr(face, 'det_score'):
                statistics['scores'].append(float(face.det_score))
            else:
                statistics['scores'].append(1.0)
        
        return statistics
    
    def filter_faces_by_size(self, image: np.ndarray, min_size: float = 0.01) -> List:
        """Filter faces by minimum size (as percentage of image area)"""
        faces = self.detect_faces(image)
        if not faces:
            return []
        
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * min_size
        
        filtered_faces = []
        for face in faces:
            bbox = face.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area >= min_area:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def find_dominant_face(self, image: np.ndarray) -> Optional:
        """Find the most dominant face (largest and centered)"""
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2
        
        best_face = None
        best_score = -1
        
        for face in faces:
            bbox = face.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance to center
            dist_center = np.sqrt(
                (center_x - image_center_x)**2 + 
                (center_y - image_center_y)**2
            )
            
            # Combined score: large area and close to center
            score = area / (1 + dist_center * 0.01)
            
            if score > best_score:
                best_score = score
                best_face = face
        
        return best_face
    
    def clear_cache(self):
        """Clear detection cache"""
        self.detection_cache.clear()


class SmartFaceSelector:
    """Smart face selector with multiple criteria"""
    
    def __init__(self, detector: MultiModelFaceDetector):
        self.detector = detector
    
    def select_by_criteria(
        self,
        image: np.ndarray,
        criteria: str = 'largest',
        confidence_threshold: float = 0.5
    ) -> List:
        """
        Select faces according to criteria
        
        Args:
            image: Input image
            criteria: 'largest', 'smallest', 'center', 'left', 'right', 'top', 'bottom', 'all'
            confidence_threshold: Minimum confidence threshold
        """
        faces = self.detector.detect_faces(image)
        
        # Filter by confidence
        if confidence_threshold > 0:
            faces = [
                f for f in faces 
                if hasattr(f, 'det_score') and f.det_score >= confidence_threshold
            ]
        
        if not faces:
            return []
        
        if criteria == 'all':
            return faces
        elif criteria == 'largest':
            return [self._get_largest(faces)]
        elif criteria == 'smallest':
            return [self._get_smallest(faces)]
        elif criteria == 'center':
            return [self._get_center(image, faces)]
        elif criteria == 'left':
            return [self._get_left(faces)]
        elif criteria == 'right':
            return [self._get_right(faces)]
        elif criteria == 'top':
            return [self._get_top(faces)]
        elif criteria == 'bottom':
            return [self._get_bottom(faces)]
        
        return faces
    
    def _get_largest(self, faces: List):
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        return faces[np.argmax(areas)]
    
    def _get_smallest(self, faces: List):
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        return faces[np.argmin(areas)]
    
    def _get_center(self, image: np.ndarray, faces: List):
        center_x = image.shape[1] / 2
        center_y = image.shape[0] / 2
        distances = []
        for f in faces:
            bbox = f.bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            distances.append(dist)
        return faces[np.argmin(distances)]
    
    def _get_left(self, faces: List):
        return sorted(faces, key=lambda f: f.bbox[0])[0]
    
    def _get_right(self, faces: List):
        return sorted(faces, key=lambda f: f.bbox[0], reverse=True)[0]
    
    def _get_top(self, faces: List):
        return sorted(faces, key=lambda f: f.bbox[1])[0]
    
    def _get_bottom(self, faces: List):
        return sorted(faces, key=lambda f: f.bbox[1], reverse=True)[0]


def draw_face_analysis(image: np.ndarray, faces: List, show_info: bool = True) -> np.ndarray:
    """Draw detailed face analysis on image"""
    result = image.copy()
    
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for idx, face in enumerate(faces):
        color = colors[idx % len(colors)]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        if show_info:
            # Face information
            area = (x2 - x1) * (y2 - y1)
            info = f"F{idx}: {area:.0f}px"
            
            if hasattr(face, 'det_score'):
                info += f" ({face.det_score:.2f})"
            
            cv2.putText(result, info, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(result, (center_x, center_y), 3, color, -1)
    
    return result

