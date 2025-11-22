"""
Batch Processing System
Process multiple images/videos in batch
"""

import os
import cv2
import numpy as np
from typing import List, Callable, Dict, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchProcessor:
    """Process multiple files in batch"""
    
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def process_images_batch(
        self,
        input_dir: str,
        output_dir: str,
        processing_function: Callable,
        function_params: Optional[Dict] = None,
        max_workers: int = 4,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process multiple images in batch
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            processing_function: Function to process each image
            function_params: Parameters for processing function
            max_workers: Number of parallel workers
            progress_callback: Progress callback
        """
        if function_params is None:
            function_params = {}
        
        # Get all image files
        image_files = self._get_image_files(input_dir)
        
        if not image_files:
            return {'error': 'No image files found'}
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'processed': 0,
            'failed': 0,
            'total': len(image_files),
            'errors': []
        }
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_image,
                    img_path, output_dir, processing_function, function_params
                ): img_path
                for img_path in image_files
            }
            
            completed = 0
            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    success = future.result()
                    if success:
                        results['processed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(str(img_path))
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / len(image_files))
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{img_path}: {str(e)}")
        
        return results
    
    def _process_single_image(
        self,
        image_path: str,
        output_dir: str,
        processing_function: Callable,
        params: Dict
    ) -> bool:
        """Process a single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Process
            result = processing_function(image, **params)
            
            # Save
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, result)
            
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files from directory"""
        image_files = []
        for ext in self.supported_image_formats:
            image_files.extend(Path(directory).glob(f'*{ext}'))
            image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        return [str(f) for f in image_files]
    
    def _get_video_files(self, directory: str) -> List[str]:
        """Get all video files from directory"""
        video_files = []
        for ext in self.supported_video_formats:
            video_files.extend(Path(directory).glob(f'*{ext}'))
            video_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        return [str(f) for f in video_files]
    
    def process_videos_batch(
        self,
        input_dir: str,
        output_dir: str,
        processing_function: Callable,
        function_params: Optional[Dict] = None,
        max_workers: int = 2,  # Lower for videos (more resource intensive)
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process multiple videos in batch
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            processing_function: Function to process each video
            function_params: Parameters for processing function
            max_workers: Number of parallel workers
            progress_callback: Progress callback
        """
        if function_params is None:
            function_params = {}
        
        # Get all video files
        video_files = self._get_video_files(input_dir)
        
        if not video_files:
            return {'error': 'No video files found'}
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'processed': 0,
            'failed': 0,
            'total': len(video_files),
            'errors': []
        }
        
        # Process videos (can be resource intensive)
        completed = 0
        for video_path in video_files:
            try:
                filename = os.path.basename(video_path)
                output_path = os.path.join(output_dir, filename)
                
                # Process video
                result_path = processing_function(video_path, output_path, **function_params)
                
                if result_path and os.path.exists(result_path):
                    results['processed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(str(video_path))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed / len(video_files))
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{video_path}: {str(e)}")
        
        return results
    
    def create_processing_report(self, results: Dict) -> str:
        """Create a text report of processing results"""
        report = f"""
Batch Processing Report
=======================
Total files: {results.get('total', 0)}
Processed successfully: {results.get('processed', 0)}
Failed: {results.get('failed', 0)}
Success rate: {(results.get('processed', 0) / results.get('total', 1) * 100):.1f}%

"""
        if results.get('errors'):
            report += "Errors:\n"
            for error in results['errors'][:10]:  # Show first 10 errors
                report += f"  - {error}\n"
            if len(results['errors']) > 10:
                report += f"  ... and {len(results['errors']) - 10} more errors\n"
        
        return report


def process_folder_batch(
    input_folder: str,
    output_folder: str,
    process_func: Callable,
    file_type: str = 'images'
) -> Dict:
    """Simple batch processing function"""
    processor = BatchProcessor()
    
    if file_type == 'images':
        return processor.process_images_batch(input_folder, output_folder, process_func)
    else:
        return processor.process_videos_batch(input_folder, output_folder, process_func)

