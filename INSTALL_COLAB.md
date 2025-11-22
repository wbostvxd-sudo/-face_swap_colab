# 🚀 Installation Guide for Google Colab

## Quick Setup (Copy & Paste)

### Step 1: Upload Files
Upload all files from `face_swap_colab_pro` folder to your Colab session.

### Step 2: Run This Code
```python
# Install dependencies (automatic)
!pip install -q gradio insightface==0.7.3 onnxruntime-gpu opencv-python numpy moviepy scipy

# Install ffmpeg (for video processing)
!apt-get install -y ffmpeg > /dev/null 2>&1

# Run the application
!python face_swap_colab_main.py
```

## Alternative: One-Cell Setup

```python
# Complete setup in one cell
import os
import subprocess
import sys

# Install packages
packages = ['gradio', 'insightface==0.7.3', 'onnxruntime-gpu', 
            'opencv-python', 'numpy', 'moviepy', 'scipy']
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Install ffmpeg
os.system('apt-get install -y ffmpeg > /dev/null 2>&1')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Run application
exec(open('face_swap_colab_main.py').read())
```

## File Structure in Colab

Make sure your Colab session has this structure:
```
/content/
├── face_swap_colab_main.py
├── face_detection_engine.py
├── face_enhancement_pro.py
├── face_blending_system.py
├── video_processor_optimized.py
├── requirements.txt
└── README_COLAB.md
```

## Verification

After running, you should see:
- ✅ "Installing dependencies..."
- ✅ "Model downloaded successfully" (first time)
- ✅ "Model loaded"
- ✅ "Starting Face Swap Pro for Colab..."
- ✅ A Gradio interface URL

## Troubleshooting

### If model doesn't download:
```python
import urllib.request
os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx',
    'models/inswapper_128.onnx'
)
```

### If ffmpeg error:
```python
!apt-get update
!apt-get install -y ffmpeg
```

### If memory issues:
```python
# Clear cache
import gc
gc.collect()

# Restart runtime: Runtime > Restart runtime
```

## GPU Check

Verify GPU is enabled:
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Ready to Use! 🎉

Once the interface loads, you can:
1. Swap faces in images
2. Process videos with face swap
3. Use advanced enhancement techniques
4. Select faces intelligently

Enjoy! 🎭

