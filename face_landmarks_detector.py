"""
Face Landmarks Detection System
Detect and work with facial landmarks (68 points)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import dlib
import os


class FaceLandmarksDetector:
    """Detect and work with facial landmarks"""
    
    def __init__(self):
        self.predictor = None
        self.detector = None
        self.landmark_model_path = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize dlib face detector and landmark predictor"""
        try:
            # Try to use dlib if available
            self.detector = dlib.get_frontal_face_detector()
            
            # Try to load shape predictor (would need model file)
            # For now, use simplified landmark estimation
            self.predictor = None
        except:
            # Fallback to OpenCV-based detection
            self.detector = None
            self.predictor = None
    
    def detect_landmarks_68(
        self,
        image: np.ndarray,
        face_bbox: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Detect 68 facial landmarks
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
        
        Returns:
            Array of 68 landmark points or None
        """
        if face_bbox is None:
            # Detect face first
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self._detect_faces_simple(gray)
            if not faces:
                return None
            face_bbox = faces[0]
        
        # Estimate landmarks (simplified version)
        landmarks = self._estimate_landmarks_68(image, face_bbox)
        return landmarks
    
    def _detect_faces_simple(self, gray: np.ndarray) -> List[np.ndarray]:
        """Simple face detection"""
        # Use OpenCV Haar Cascade as fallback
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert to [x1, y1, x2, y2] format
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append(np.array([x, y, x + w, y + h]))
        
        return bboxes
    
    def _estimate_landmarks_68(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """Estimate 68 facial landmarks (simplified)"""
        x1, y1, x2, y2 = bbox.astype(int)
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Simplified landmark positions based on face proportions
        landmarks = np.zeros((68, 2), dtype=np.float32)
        
        # Jaw line (points 0-16)
        jaw_y = y2 - height * 0.1
        for i in range(17):
            t = i / 16.0
            x = x1 + width * (0.1 + t * 0.8)
            landmarks[i] = [x, jaw_y]
        
        # Right eyebrow (points 17-21)
        eyebrow_y = center_y - height * 0.25
        for i in range(5):
            x = center_x - width * (0.15 - i * 0.05)
            landmarks[17 + i] = [x, eyebrow_y]
        
        # Left eyebrow (points 22-26)
        for i in range(5):
            x = center_x + width * (0.1 + i * 0.05)
            landmarks[22 + i] = [x, eyebrow_y]
        
        # Nose (points 27-35)
        nose_tip_y = center_y
        landmarks[30] = [center_x, nose_tip_y]  # Nose tip
        landmarks[27] = [center_x, center_y - height * 0.15]  # Nose top
        landmarks[31] = [center_x, center_y + height * 0.1]  # Nose bottom
        landmarks[33] = [center_x - width * 0.05, center_y]  # Left nostril
        landmarks[35] = [center_x + width * 0.05, center_y]  # Right nostril
        
        # Right eye (points 36-41)
        eye_y = center_y - height * 0.1
        eye_width = width * 0.15
        eye_center_x = center_x - width * 0.2
        for i, angle in enumerate([0, 60, 120, 180, 240, 300]):
            angle_rad = np.radians(angle)
            x = eye_center_x + eye_width * 0.4 * np.cos(angle_rad)
            y = eye_y + eye_width * 0.3 * np.sin(angle_rad)
            landmarks[36 + i] = [x, y]
        
        # Left eye (points 42-47)
        eye_center_x = center_x + width * 0.2
        for i, angle in enumerate([0, 60, 120, 180, 240, 300]):
            angle_rad = np.radians(angle)
            x = eye_center_x + eye_width * 0.4 * np.cos(angle_rad)
            y = eye_y + eye_width * 0.3 * np.sin(angle_rad)
            landmarks[42 + i] = [x, y]
        
        # Mouth (points 48-67)
        mouth_y = center_y + height * 0.2
        mouth_width = width * 0.3
        for i in range(20):
            t = i / 19.0
            if i < 12:  # Outer mouth
                angle = 180 * t
                angle_rad = np.radians(angle)
                x = center_x + mouth_width * 0.5 * np.cos(angle_rad)
                y = mouth_y + mouth_width * 0.3 * np.sin(angle_rad)
            else:  # Inner mouth
                angle = 180 * ((i - 12) / 7.0)
                angle_rad = np.radians(angle)
                x = center_x + mouth_width * 0.3 * np.cos(angle_rad)
                y = mouth_y + mouth_width * 0.15 * np.sin(angle_rad)
            landmarks[48 + i] = [x, y]
        
        return landmarks
    
    def get_landmark_regions(self, landmarks: np.ndarray) -> Dict:
        """Get different face regions from landmarks"""
        return {
            'jaw': landmarks[0:17],
            'right_eyebrow': landmarks[17:22],
            'left_eyebrow': landmarks[22:27],
            'nose': landmarks[27:36],
            'right_eye': landmarks[36:42],
            'left_eye': landmarks[42:48],
            'mouth_outer': landmarks[48:60],
            'mouth_inner': landmarks[60:68]
        }
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1
    ) -> np.ndarray:
        """Draw landmarks on image"""
        result = image.copy()
        
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(result, (x, y), 2, color, thickness)
        
        # Draw connections
        connections = [
            (0, 16),  # Jaw
            (17, 21),  # Right eyebrow
            (22, 26),  # Left eyebrow
            (27, 30), (30, 33), (30, 35),  # Nose
            (36, 41),  # Right eye
            (42, 47),  # Left eye
            (48, 59),  # Outer mouth
            (60, 67)   # Inner mouth
        ]
        
        for start, end in connections:
            pt1 = tuple(landmarks[start].astype(int))
            pt2 = tuple(landmarks[end].astype(int))
            cv2.line(result, pt1, pt2, color, 1)
        
        return result
    
    def align_face_by_landmarks(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align face using landmarks"""
        # Get eye centers
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, 
                                (image.shape[1], image.shape[0]))
        
        # Transform landmarks
        landmarks_homogeneous = np.hstack([landmarks, np.ones((68, 1))])
        aligned_landmarks = (rotation_matrix @ landmarks_homogeneous.T).T
        
        return aligned, aligned_landmarks


def detect_landmarks_simple(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Simple landmark detection function"""
    detector = FaceLandmarksDetector()
    return detector.detect_landmarks_68(image, bbox)

