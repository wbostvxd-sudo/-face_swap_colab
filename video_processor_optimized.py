"""
Optimized video processor with advanced techniques
Efficient resource management and intelligent processing
"""

import cv2
import numpy as np
import os
from typing import Optional, Dict, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid


class VideoResourceManager:
    """Resource manager for video processing"""
    
    def __init__(self):
        self.capture_pool: Dict[str, cv2.VideoCapture] = {}
        self.writer_pool: Dict[str, cv2.VideoWriter] = {}
        self.frame_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 100
    
    def get_capture(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """Get or create video capture from pool"""
        if video_path not in self.capture_pool:
            capture = cv2.VideoCapture(video_path)
            if capture.isOpened():
                self.capture_pool[video_path] = capture
                return capture
            return None
        return self.capture_pool.get(video_path)
    
    def release_capture(self, video_path: str):
        """Release video capture"""
        if video_path in self.capture_pool:
            self.capture_pool[video_path].release()
            del self.capture_pool[video_path]
    
    def cleanup_all(self):
        """Cleanup all resources"""
        for capture in self.capture_pool.values():
            capture.release()
        for writer in self.writer_pool.values():
            writer.release()
        self.capture_pool.clear()
        self.writer_pool.clear()
        self.frame_cache.clear()


class IntelligentVideoProcessor:
    """Video processor with intelligent optimizations"""
    
    def __init__(self):
        self.resource_manager = VideoResourceManager()
        self.statistics = {
            'frames_processed': 0,
            'total_time': 0,
            'average_fps': 0
        }
    
    def process_video_parallel(
        self,
        input_path: str,
        output_path: str,
        processing_function: Callable,
        parameters: Dict,
        frame_skip: int = 1,
        max_workers: int = 4,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process video in parallel with optimizations
        
        Args:
            input_path: Input video path
            output_path: Output video path
            processing_function: Function that processes a frame
            parameters: Parameters for processing function
            frame_skip: Process every N frames
            max_workers: Number of parallel threads
            progress_callback: Callback to report progress
        """
        start_time = time.time()
        
        # Get video information
        capture = self.resource_manager.get_capture(input_path)
        if capture is None:
            return {'error': 'Could not open video'}
        
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read all frames to process
        frames_to_process = []
        frame_indices = []
        current_frame = 0
        
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            if current_frame % frame_skip == 0:
                frames_to_process.append((current_frame, frame.copy()))
                frame_indices.append(current_frame)
            
            current_frame += 1
        
        self.resource_manager.release_capture(input_path)
        
        if not frames_to_process:
            return {'error': 'No frames found to process'}
        
        # Process frames in parallel
        processed_frames = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_frame_wrapper,
                    frame_data,
                    processing_function,
                    parameters
                ): idx
                for idx, frame_data in enumerate(frames_to_process)
            }
            
            completed = 0
            for future in as_completed(futures):
                try:
                    frame_num, processed_frame = future.result()
                    processed_frames[frame_num] = processed_frame
                    completed += 1
                    
                    if progress_callback:
                        progress = completed / len(frames_to_process)
                        progress_callback(progress, f"Processing: {completed}/{len(frames_to_process)}")
                except Exception as e:
                    print(f"Error processing frame: {e}")
        
        # Interpolate missing frames if skipped
        if frame_skip > 1:
            for idx in range(current_frame):
                if idx not in processed_frames:
                    closest_idx = min(frame_indices, key=lambda x: abs(x - idx))
                    processed_frames[idx] = processed_frames[closest_idx].copy()
        
        # Save video
        self._save_video(output_path, processed_frames, fps, width, height)
        
        # Update statistics
        total_time = time.time() - start_time
        self.statistics['frames_processed'] = len(processed_frames)
        self.statistics['total_time'] = total_time
        self.statistics['average_fps'] = len(processed_frames) / total_time if total_time > 0 else 0
        
        return {
            'success': True,
            'output_path': output_path,
            'statistics': self.statistics.copy()
        }
    
    def _process_frame_wrapper(
        self,
        frame_data: Tuple[int, np.ndarray],
        function: Callable,
        parameters: Dict
    ) -> Tuple[int, np.ndarray]:
        """Wrapper to process a frame"""
        frame_num, frame = frame_data
        processed_frame = function(frame, **parameters)
        return frame_num, processed_frame
    
    def _save_video(
        self,
        output_path: str,
        frames: Dict[int, np.ndarray],
        fps: float,
        width: int,
        height: int
    ):
        """Save processed frames as video"""
        # Create temporary directory for frames
        temp_dir = os.path.join("temp", f"video_{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save frames as JPEG
        for frame_num in sorted(frames.keys()):
            cv2.imwrite(
                os.path.join(frames_dir, f"{frame_num:08d}.jpg"),
                frames[frame_num],
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
        
        # Use ffmpeg to create video (requires ffmpeg installed)
        import subprocess
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%08d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            temp_video
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Move to final path
            import shutil
            shutil.move(temp_video, output_path)
        except Exception as e:
            print(f"Error creating video: {e}")
        finally:
            # Cleanup temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass


def estimate_processing_time(
    total_frames: int,
    video_fps: float,
    frame_skip: int = 1,
    max_workers: int = 4
) -> Dict:
    """Estimate processing time"""
    frames_to_process = total_frames // frame_skip
    time_per_frame = 0.1  # Conservative estimate
    
    parallel_time = (frames_to_process / max_workers) * time_per_frame
    sequential_time = frames_to_process * time_per_frame
    
    return {
        'frames_to_process': frames_to_process,
        'estimated_parallel_time': parallel_time,
        'estimated_sequential_time': sequential_time,
        'speedup': sequential_time / parallel_time if parallel_time > 0 else 1
    }

