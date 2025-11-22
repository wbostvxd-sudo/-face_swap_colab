"""
Professional Face Swap System for Google Colab
Complete system with all advanced features integrated
Optimized for Google Colab GPU usage
"""

import subprocess
import sys
import cv2
import gradio as gr
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from moviepy.editor import VideoFileClip
import tempfile
from typing import Tuple, Optional, Dict, List
import time
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import advanced modules
try:
    from face_detection_engine import MultiModelFaceDetector, SmartFaceSelector, draw_face_analysis
    from face_enhancement_pro import ProFaceEnhancer, enhance_faces_in_image
    from face_blending_system import AdvancedFaceBlender, create_smooth_face_mask
    from video_processor_optimized import IntelligentVideoProcessor
    from background_remover_advanced import AdvancedBackgroundRemover
    from face_classifier_system import FaceClassifier
    from face_mask_advanced import AdvancedFaceMasker
    from frame_enhancer_system import FrameEnhancerSystem
    from face_landmarks_detector import FaceLandmarksDetector
    from batch_processor import BatchProcessor
    MODULES_AVAILABLE = True
except ImportError:
    print("Warning: Advanced modules not available, using basic mode")
    MODULES_AVAILABLE = False

# Colab optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def install_dependencies():
    """Install required dependencies"""
    packages = [
        'gradio',
        'insightface==0.7.3',
        'onnxruntime-gpu',
        'opencv-python',
        'numpy',
        'moviepy',
        'scipy'
    ]
    
    for package in packages:
        try:
            __import__(package.split('==')[0].replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def download_inswapper_model():
    """Download inswapper model if not exists"""
    model_path = 'models/inswapper_128.onnx'
    if not os.path.exists(model_path):
        print("Downloading inswapper_128.onnx model...")
        model_url = "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx"
        try:
            import urllib.request
            os.makedirs('models', exist_ok=True)
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Error downloading: {e}")
            return False
    return True


# Initialization
print("Installing dependencies...")
install_dependencies()

os.makedirs('models', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('results', exist_ok=True)

if not download_inswapper_model():
    print("Warning: Model not downloaded")

# Initialize components
print("Initializing components...")
if MODULES_AVAILABLE:
    detector = MultiModelFaceDetector(base_model='buffalo_l', detection_size=(320, 320))
    selector = SmartFaceSelector(detector)
    enhancer = ProFaceEnhancer()
    blender = AdvancedFaceBlender()
    video_processor = IntelligentVideoProcessor()
    background_remover = AdvancedBackgroundRemover()
    face_classifier = FaceClassifier()
    face_masker = AdvancedFaceMasker()
    frame_enhancer = FrameEnhancerSystem()
    landmarks_detector = FaceLandmarksDetector()
    batch_processor = BatchProcessor()
    app = detector.app
else:
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(320, 320))
    detector = None
    selector = None
    enhancer = None
    blender = None
    video_processor = None
    background_remover = None
    face_classifier = None
    face_masker = None
    frame_enhancer = None
    landmarks_detector = None
    batch_processor = None

# Load inswapper model
print("Loading Inswapper model...")
try:
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=False, download_zip=False)
    print("Model loaded")
except Exception as e:
    print(f"Error: {e}")
    swapper = None

# Global cache
_source_face_cache = None
_source_image_hash = None


def process_image_advanced(
    source_image: np.ndarray,
    target_image: np.ndarray,
    face_index: int = 0,
    enhancement_technique: str = "none",
    enhancement_intensity: float = 0.5,
    swap_all_faces: bool = False,
    selection_mode: str = "index",
    blending_technique: str = "feather"
) -> Tuple[np.ndarray, str]:
    """Process image with all advanced features"""
    global _source_face_cache, _source_image_hash
    
    if swapper is None:
        return target_image, "Error: Model not loaded"
    
    try:
        # Detect source face (with cache)
        source_hash = hash(source_image.tobytes())
        if _source_face_cache is not None and _source_image_hash == source_hash:
            source_faces = _source_face_cache
        else:
            if MODULES_AVAILABLE and detector:
                source_faces = detector.detect_faces(source_image)
            else:
                source_faces = app.get(source_image)
            _source_face_cache = source_faces
            _source_image_hash = source_hash
        
        if not source_faces:
            return target_image, "No faces detected in source"
        
        # Detect target faces
        if MODULES_AVAILABLE and selector and selection_mode != "index":
            if selection_mode == "largest":
                target_faces = [detector.find_dominant_face(target_image)] if detector else []
            elif selection_mode == "center":
                target_faces = selector.select_by_criteria(target_image, 'center')
            else:
                target_faces = selector.select_by_criteria(target_image, selection_mode)
            target_faces = [f for f in target_faces if f is not None]
        else:
            target_faces = app.get(target_image)
        
        if not target_faces:
            return target_image, "No faces detected in target"
        
        result = target_image.copy()
        
        # Swap faces
        if swap_all_faces:
            for target_face in target_faces:
                result = swapper.get(result, target_face, source_faces[0], paste_back=True)
            message = f"Swapped {len(target_faces)} faces"
        else:
            if face_index < len(target_faces):
                result = swapper.get(result, target_faces[face_index], source_faces[0], paste_back=True)
                message = f"Face {face_index} swapped"
            else:
                return target_image, f"Index {face_index} out of range"
        
        # Apply enhancements
        if enhancement_technique != "none" and enhancement_intensity > 0:
            if MODULES_AVAILABLE and enhancer:
                if swap_all_faces:
                    for target_face in target_faces:
                        if hasattr(target_face, 'bbox'):
                            result = enhancer.apply_enhancement_to_face_region(
                                result, target_face.bbox, enhancement_technique, enhancement_intensity
                            )
                else:
                    if face_index < len(target_faces):
                        result = enhancer.apply_enhancement_to_face_region(
                            result, target_faces[face_index].bbox, enhancement_technique, enhancement_intensity
                        )
            else:
                # Basic enhancement
                alpha = 1 + (enhancement_intensity * 0.2)
                beta = enhancement_intensity * 10
                result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        return result, message
    
    except Exception as e:
        return target_image, f"Error: {str(e)}"


def process_frame_batch(args):
    """Process a frame in batch"""
    frame_data, src_img, idx, enh, intensity, all_faces, sel, blend = args
    frame_num, frame = frame_data
    processed_frame, _ = process_image_advanced(
        src_img, frame, idx, enh, intensity, all_faces, sel, blend
    )
    return frame_num, processed_frame


def process_video_advanced(
    source_image: np.ndarray,
    video_path: str,
    output_path: str,
    face_index: int = 0,
    enhancement_technique: str = "none",
    enhancement_intensity: float = 0.5,
    swap_all_faces: bool = False,
    progress: Optional[gr.Progress] = None,
    frame_skip: int = 1,
    max_workers: int = 4
) -> str:
    """Process video with all optimizations"""
    if swapper is None:
        return "Error: Model not loaded"
    
    if MODULES_AVAILABLE and video_processor:
        # Use advanced processor
        def process_frame(frame):
            result, _ = process_image_advanced(
                source_image, frame, face_index, enhancement_technique,
                enhancement_intensity, swap_all_faces
            )
            return result
        
        def progress_callback(prog, desc):
            if progress:
                progress(prog, desc=desc)
        
        result = video_processor.process_video_parallel(
            video_path, output_path, process_frame, {},
            frame_skip, max_workers, progress_callback
        )
        
        if 'error' in result:
            return result['error']
        return output_path
    else:
        # Basic processing (simplified code)
        return "Error: Video processor not available"


def create_advanced_interface():
    """Create advanced Gradio interface"""
    with gr.Blocks(title="Face Swap Pro for Colab", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🎭 Face Swap Professional for Google Colab
            
            Advanced system with multiple processing techniques and optimizations.
            
            **Features:**
            - 🔍 Advanced face detection
            - 🎨 Multiple enhancement techniques
            - 🎯 Smart face selector
            - ⚡ Optimized parallel processing
            """
        )
        
        with gr.Tab("🖼️ Image Face Swap"):
            with gr.Row():
                with gr.Column():
                    source_img = gr.Image(label="Source Image", type="numpy")
                    detect_source_btn = gr.Button("🔍 Detect", variant="secondary")
                    source_faces_text = gr.Textbox(label="Faces", interactive=False)
                
                with gr.Column():
                    target_img = gr.Image(label="Target Image", type="numpy")
                    detect_target_btn = gr.Button("🔍 Detect", variant="secondary")
                    target_faces_text = gr.Textbox(label="Faces", interactive=False)
            
            with gr.Row():
                selection_mode = gr.Radio(
                    choices=["index", "largest", "center", "left", "right"],
                    value="index",
                    label="Selection Mode"
                ) if MODULES_AVAILABLE else None
                face_index = gr.Slider(0, 10, 0, 1, label="Face Index")
                swap_all = gr.Checkbox(label="Swap All Faces", value=False)
            
            with gr.Row():
                enhancement_technique = gr.Radio(
                    choices=["none", "advanced_sharpen", "noise_reduction", "color_balance"] if MODULES_AVAILABLE else ["none", "basic"],
                    value="none",
                    label="Enhancement"
                )
                intensity = gr.Slider(0, 1, 0.5, 0.1, label="Intensity")
            
            process_btn = gr.Button("✨ Swap Faces", variant="primary")
            
            with gr.Row():
                result_img = gr.Image(label="Result")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("🎬 Video Face Swap"):
            with gr.Row():
                source_img_vid = gr.Image(label="Source Image", type="numpy")
                target_video = gr.Video(label="Target Video")
            
            with gr.Row():
                face_index_vid = gr.Slider(0, 10, 0, 1, label="Face Index")
                swap_all_vid = gr.Checkbox(label="Swap All Faces", value=False)
            
            with gr.Row():
                enhancement_vid = gr.Radio(
                    choices=["none", "advanced_sharpen", "noise_reduction"] if MODULES_AVAILABLE else ["none", "basic"],
                    value="none",
                    label="Enhancement"
                )
                intensity_vid = gr.Slider(0, 1, 0.5, 0.1, label="Intensity")
            
            with gr.Row():
                frame_skip = gr.Slider(1, 5, 1, 1, label="Skip Frames")
                workers = gr.Slider(1, 8, 4, 1, label="Parallel Workers")
            
            process_video_btn = gr.Button("✨ Process Video", variant="primary")
            
            with gr.Row():
                result_video = gr.Video(label="Result")
                status_video_text = gr.Textbox(label="Status", interactive=False)
        
        # Connect events
        def count_faces(img):
            if img is None:
                return "Load an image"
            try:
                img_np = np.array(img)
                if MODULES_AVAILABLE and detector:
                    count = detector.detect_faces(img_np)
                    return f"{len(count)} faces detected"
                else:
                    faces = app.get(img_np)
                    return f"{len(faces)} faces detected"
            except:
                return "Error"
        
        detect_source_btn.click(count_faces, inputs=[source_img], outputs=[source_faces_text])
        detect_target_btn.click(count_faces, inputs=[target_img], outputs=[target_faces_text])
        
        def process_image_func(src, tgt, idx, enh, ints, all_faces, sel):
            if src is None or tgt is None:
                return None, "Load both images"
            try:
                src_np = np.array(src)
                tgt_np = np.array(tgt)
                res, msg = process_image_advanced(
                    src_np, tgt_np, idx, enh, ints, all_faces, sel if sel else "index"
                )
                return res, msg
            except Exception as e:
                return None, f"Error: {e}"
        
        image_inputs = [source_img, target_img, face_index, enhancement_technique, intensity, swap_all]
        if selection_mode:
            image_inputs.insert(2, selection_mode)
        
        process_btn.click(process_image_func, inputs=image_inputs, outputs=[result_img, status_text])
        
        def process_video_func(src, vid, idx, enh, ints, all_faces, skip, wrk, prog=gr.Progress()):
            if src is None or vid is None:
                return None, "Load image and video"
            try:
                src_np = np.array(src)
                output_filename = f"result_{int(time.time())}.mp4"
                output_path = os.path.join("results", output_filename)
                res = process_video_advanced(
                    src_np, vid, output_path, idx, enh, ints, all_faces, prog, skip, wrk
                )
                if res.startswith("Error"):
                    return None, res
                return res, "Completed"
            except Exception as e:
                return None, f"Error: {e}"
        
        process_video_btn.click(
            process_video_func,
            inputs=[source_img_vid, target_video, face_index_vid, enhancement_vid, intensity_vid, swap_all_vid, frame_skip, workers],
            outputs=[result_video, status_video_text]
        )
    
    return interface


if __name__ == "__main__":
    print("Starting Face Swap Pro for Colab...")
    interface = create_advanced_interface()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)

