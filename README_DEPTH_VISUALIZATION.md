# 🎨 Enhanced Depth Visualization for FastScene

Hệ thống trực quan hóa bản đồ độ sâu nâng cao với nhiều chế độ hiển thị và organized output management.

## 🚀 Tính năng chính

### ✨ **Multi-Colormap Support**

- **Scientific**: `viridis`, `plasma`, `inferno`, `magma`
- **High Contrast**: `jet`, `turbo`
- **Custom**: Near=Red, Mid=Green, Far=Blue

### 📊 **Normalization Modes**

- **Percentile**: Robust, loại bỏ outliers (5%-95%)
- **MinMax**: Full range scaling
- **Fixed**: Cố định range (0-10m)

### 📁 **Organized Output Structure**

```
output_progressive/
├── rgb_images/           # RGB inpainted images
├── depth_maps/          # Raw depth data (.npy)
├── depth_visualizations/# Colored depth maps with colorbar
├── side_by_side/        # RGB vs Depth comparisons
├── progress_tracking/   # Processing metadata (JSON)
└── metadata/           # Overall process info
```

### 🎯 **Visualization Types**

- **Single Depth**: Colorized depth map với colorbar
- **Side-by-Side**: RGB và Depth cạnh nhau
- **Difference Maps**: So sánh hai depth maps
- **Progress Tracking**: Metadata và statistics

## 📦 Installation & Setup

### 1. Dependencies

```bash
pip install matplotlib opencv-python pillow numpy torch torchvision
```

### 2. Project Structure

```
spacify/
├── utils/
│   └── depth_visualizer.py      # Core visualization classes
├── pro_inpaint_enhanced.py      # Enhanced inpainting pipeline
├── demo_depth_visualization.py  # Test & demo script
└── README_DEPTH_VISUALIZATION.md
```

## 🎮 Quick Start

### 1. Test Depth Visualization

```bash
# Chạy demo để test tất cả features
python demo_depth_visualization.py
```

Output:

- `demo_depth_visualization/` - Colormap comparisons
- `demo_output_organized/` - Organized output example

### 2. Enhanced Progressive Inpainting

```bash
# Basic usage
python pro_inpaint_enhanced.py --enable_depth_vis --depth_colormap viridis

# Advanced options
python pro_inpaint_enhanced.py \
    --enable_depth_vis \
    --depth_colormap custom \
    --output_dir my_experiment_001
```

### 3. Custom Integration

```python
from utils.depth_visualizer import DepthVisualizer, OutputManager

# Setup
visualizer = DepthVisualizer(colormap='viridis', normalize_mode='percentile')
output_manager = OutputManager('my_output')

# Create depth visualization
depth_vis = visualizer.create_depth_visualization(
    depth_map=your_depth_array,
    save_path='depth_colored.png',
    title='My Depth Map'
)

# Create side-by-side comparison
combined = visualizer.create_side_by_side(
    rgb_image=your_rgb_array,
    depth_map=your_depth_array,
    save_path='rgb_vs_depth.png'
)
```

## ⚙️ Configuration Options

### Enhanced Progressive Inpainting Args

```bash
--enable_depth_vis          # Bật depth visualization (default: True)
--depth_colormap COLORMAP   # Chọn colormap (default: viridis)
--output_dir DIR           # Output directory (default: output_progressive)
```

### Available Colormaps

| Colormap  | Best For      | Description                               |
| --------- | ------------- | ----------------------------------------- |
| `viridis` | Scientific    | Perceptually uniform, colorblind-friendly |
| `plasma`  | Scientific    | High contrast, purple-pink-yellow         |
| `inferno` | Scientific    | Black-red-yellow, good for heat maps      |
| `jet`     | High contrast | Classic rainbow, maximum contrast         |
| `turbo`   | High contrast | Improved rainbow, better than jet         |
| `magma`   | Dark themes   | Black-purple-white                        |
| `custom`  | Intuitive     | Near=Red, Mid=Green, Far=Blue             |

### Normalization Modes

| Mode         | Description     | Use Case                 |
| ------------ | --------------- | ------------------------ |
| `percentile` | 5%-95% range    | Robust, removes outliers |
| `minmax`     | Full data range | Preserve all detail      |
| `fixed`      | Fixed 0-10m     | Compare across scenes    |

## 📈 Performance Tips

### 1. **Caching Enabled**

- Depth computations tự động cached
- Spherical coordinates cached cho speed

### 2. **Batch Processing**

- Process multiple images đồng thời
- Optimized memory usage

### 3. **Selective Saving**

```python
config.save_depth_raw = False      # Skip .npy files
config.save_side_by_side = False   # Skip comparisons
config.enable_progress_tracking = False  # Skip metadata
```

## 🔧 Advanced Usage

### Custom Colormap

```python
# Define custom colormap function
def my_custom_colormap(normalized_depth):
    colored = np.zeros((*normalized_depth.shape, 3))
    # Your custom color logic here
    return (colored * 255).astype(np.uint8)

# Use trong visualizer
visualizer.apply_colormap = my_custom_colormap
```

### Progress Monitoring

```python
# Read processing metadata
import json
with open('output_progressive/progress_tracking/progress_001.json') as f:
    metadata = json.load(f)

print(f"Step {metadata['step']}: {metadata['processing_time']:.2f}s")
print(f"Depth range: {metadata['depth_stats']['min']:.3f} - {metadata['depth_stats']['max']:.3f}")
```

### Batch Analysis

```python
# Analyze all depth maps in directory
import glob
import numpy as np

depth_files = glob.glob('output_progressive/depth_maps/*.npy')
all_stats = []

for file in depth_files:
    depth = np.load(file)
    stats = {
        'file': file,
        'min': depth.min(),
        'max': depth.max(),
        'mean': depth.mean(),
        'std': depth.std()
    }
    all_stats.append(stats)

# Visualize statistics
import pandas as pd
df = pd.DataFrame(all_stats)
print(df.describe())
```

## 🎯 Output Examples

### 1. Single Depth Visualization

- **Input**: Raw depth array (512×1024)
- **Output**: Colorized depth với colorbar và statistics
- **Format**: PNG với high DPI (150)

### 2. Side-by-Side Comparison

- **Input**: RGB image + Depth map
- **Output**: Horizontal concatenation với separate colorbars
- **Use**: Visual quality assessment

### 3. Organized Structure

```
my_experiment/
├── rgb_images/
│   ├── step_001_x_rgb.png
│   ├── step_002_x_rgb.png
│   └── ...
├── depth_visualizations/
│   ├── step_001_x_depth_vis.png
│   ├── step_002_x_depth_vis.png
│   └── ...
└── metadata/
    └── final_report.json
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **Memory Issues**

```python
# Reduce batch size
config.batch_size = 2

# Disable caching
config.enable_caching = False

# Skip heavy visualizations
config.save_side_by_side = False
```

#### 2. **Slow Processing**

```python
# Use faster normalization
config.depth_normalize_mode = 'minmax'

# Reduce visualization quality
depth_visualizer.create_depth_visualization(..., dpi=100)
```

#### 3. **Disk Space**

```python
# Save only essentials
config.save_depth_raw = False
config.enable_progress_tracking = False
```

## 📊 Performance Benchmarks

| Configuration              | Time/Step | Memory | Disk/Step |
| -------------------------- | --------- | ------ | --------- |
| **Full** (all features)    | ~15s      | 4GB    | 25MB      |
| **Fast** (RGB + depth vis) | ~8s       | 2GB    | 8MB       |
| **Minimal** (RGB only)     | ~5s       | 1GB    | 2MB       |

## 🔄 Integration với Existing Pipeline

### Replace Original

```python
# Before
from pro_inpaint import progressive_inpaint_optimized

# After
from pro_inpaint_enhanced import progressive_inpaint_enhanced
```

### Gradual Migration

```python
# Keep both versions
from pro_inpaint import progressive_inpaint_optimized as original
from pro_inpaint_enhanced import progressive_inpaint_enhanced as enhanced

# Use based on needs
if need_depth_visualization:
    result = enhanced(rgb, depth)
else:
    result = original(rgb, depth)
```

## 🎉 Next Steps

1. **Run Demo**: `python demo_depth_visualization.py`
2. **Test Enhanced**: `python pro_inpaint_enhanced.py`
3. **Customize**: Modify colormaps/settings theo nhu cầu
4. **Integrate**: Replace trong existing workflow

---

## 💡 Pro Tips

- **Scientific Work**: Dùng `viridis` + `percentile`
- **Presentations**: Dùng `jet` + `minmax` cho high contrast
- **Debugging**: Dùng `custom` + side-by-side để detect issues
- **Performance**: Disable unnecessary features cho speed
- **Storage**: Use `.npy` cho raw data, `.png` cho visualization

**Enjoy enhanced depth visualization! 🎨✨**
