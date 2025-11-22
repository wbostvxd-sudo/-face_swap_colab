# Face Swap Pro for Google Colab

Complete professional face swap system optimized for Google Colab GPU.

## 🚀 Quick Start in Google Colab

### Method 1: Direct Upload
1. Upload all files to your Colab session
2. Run:
```python
!python face_swap_colab_main.py
```

### Method 2: From GitHub (if uploaded)
```python
!git clone https://github.com/your-repo/face_swap_colab_pro.git
%cd face_swap_colab_pro
!python face_swap_colab_main.py
```

## 📁 Project Structure

```
face_swap_colab_pro/
├── face_swap_colab_main.py          # Main application file
├── face_detection_engine.py          # Advanced face detection
├── face_enhancement_pro.py           # Professional enhancement
├── face_blending_system.py           # Advanced blending
├── video_processor_optimized.py      # Optimized video processing
├── requirements.txt                  # Dependencies
└── README_COLAB.md                  # This file
```

## 🎯 Features

### Advanced Face Detection
- Multi-model detection system
- Smart face selection by criteria
- Face statistics and analysis
- Detection caching for performance

### Professional Enhancement
- 8 enhancement techniques:
  - Advanced sharpening
  - Noise reduction
  - Color balance
  - Smart smoothing
  - Detail enhancement
  - Illumination correction
  - Contrast enhancement
  - Bilateral filtering

### Advanced Blending
- 6 blending techniques:
  - Feather blending
  - Poisson blending
  - Seamless cloning
  - Multiband blending
  - Gaussian blending
  - Linear blending

### Optimized Video Processing
- Parallel frame processing
- Resource pooling
- Frame interpolation
- Performance statistics

## 💻 Usage

### Image Face Swap
1. Upload source image (face to transfer)
2. Upload target image
3. Select face index or use smart selection
4. Choose enhancement technique
5. Click "Swap Faces"

### Video Face Swap
1. Upload source image
2. Upload target video
3. Configure processing options:
   - Frame skip (1=all frames, 2=every 2 frames, etc.)
   - Parallel workers (1-8)
4. Click "Process Video"

## ⚡ Performance Tips

### For Long Videos
- Use `frame_skip=2` or `3` for faster processing
- Reduce `workers` to 2-3 if RAM limited

### For Maximum Quality
- Use `frame_skip=1` (process all frames)
- Use `workers=4-6` depending on GPU

### For Powerful GPU
- Increase `workers` to 6-8
- Keep `frame_skip=1` for maximum quality

## 🔧 Installation in Colab

The script automatically installs dependencies. If manual installation needed:

```python
!pip install -r requirements.txt
```

## 📝 Notes

- Optimized for Google Colab GPU
- Automatic model download on first run
- Temporary files cleaned automatically
- Thread-safe for Colab environment

## 🐛 Troubleshooting

### Model Download Error
Manually download from:
`https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx`
Place in `models/inswapper_128.onnx`

### FFmpeg Error
In Colab, run:
```python
!apt-get install -y ffmpeg
```

### Memory Issues
- Reduce `workers` to 2-3
- Use `frame_skip=2` or more
- Clear cache: Restart runtime

## 📄 License

Based on open source code. See individual files for license details.

## 🙏 Credits

- **Inswapper**: Face swap model
- **InsightFace**: Face detection
- **Gradio**: User interface

