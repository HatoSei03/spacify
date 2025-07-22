# 🚀 Quick Start - Enhanced Depth Visualization

## ⚡ Super Quick Start (2 minutes)

### 1. Test Demo (No models needed)

```bash
python run_enhanced.py --demo
```

→ Creates sample depth visualizations trong `demo_depth_visualization/`

### 2. Quick Test (Requires models)

```bash
# Fast test với viridis colormap
python run_enhanced.py --quick-test fast

# Scientific test với plasma colormap
python run_enhanced.py --quick-test scientific

# Full test với custom colormap
python run_enhanced.py --quick-test full
```

### 3. Full Enhanced Inpainting

```bash
# Basic - RGB + Depth visualization
python run_enhanced.py --input inpaint_data/demo.png

# Advanced - Custom colormap và output folder
python run_enhanced.py --input inpaint_data/demo.png --colormap custom --output-dir my_experiment
```

## 📁 What You Get

### Demo Output (`demo_depth_visualization/`)

```
demo_depth_visualization/
├── colormap_comparison.png      # All colormaps side-by-side
├── normalization_comparison.png # Different normalization modes
├── depth_viridis.png           # Individual colormap examples
├── depth_plasma.png
├── combined_viridis.png        # RGB vs Depth comparisons
└── ...
```

### Enhanced Inpainting Output (`output_enhanced/`)

```
output_enhanced/
├── rgb_images/                 # 🖼️ Inpainted RGB images
│   ├── step_001_x_rgb.png
│   └── ...
├── depth_visualizations/       # 🎨 Colored depth maps
│   ├── step_001_x_depth_vis.png
│   └── ...
├── side_by_side/              # 📊 RGB vs Depth comparisons
│   ├── step_001_x_combined.png
│   └── ...
├── depth_maps/                # 💾 Raw depth data (.npy)
├── progress_tracking/         # 📈 Processing metadata
└── final_report.json         # 📋 Complete summary
```

## 🎨 Colormap Preview

| Colormap  | Best For         | Visual Style                   |
| --------- | ---------------- | ------------------------------ |
| `viridis` | Scientific work  | Purple → Blue → Green → Yellow |
| `plasma`  | High contrast    | Purple → Pink → Yellow         |
| `inferno` | Heat maps        | Black → Red → Yellow           |
| `jet`     | Maximum contrast | Blue → Green → Yellow → Red    |
| `custom`  | Intuitive        | Near=Red, Mid=Green, Far=Blue  |

## ⚙️ Common Options

```bash
# Different colormaps
--colormap viridis     # Scientific (default)
--colormap custom      # Intuitive near/far colors
--colormap jet         # High contrast

# Output control
--output-dir my_test   # Custom output folder
--disable-vis          # Skip depth visualization (faster)

# Comparison
--compare output1 output2  # Compare two experiments
```

## 🔧 Troubleshooting

### ❌ "Missing dependency"

```bash
pip install -r requirements_enhanced.txt
```

### ❌ "Missing pretrained models"

Download và đặt vào:

- `pretrained_models/EGformer_pretrained.pkl`
- `pretrained_models/G0185000.pt`
- `models/RealESRGAN_x2plus.pth`

### ❌ "Out of memory"

```bash
# Disable heavy features
python run_enhanced.py --input image.png --disable-vis
```

### ❌ "Too slow"

```bash
# Use fast preset
python run_enhanced.py --quick-test fast
```

## 📊 Performance Guide

| Mode                | Time  | Memory | Disk/Step | Quality   |
| ------------------- | ----- | ------ | --------- | --------- |
| `--demo`            | 30s   | 1GB    | 50MB      | Preview   |
| `--quick-test fast` | 5min  | 2GB    | 100MB     | Good      |
| `--quick-test full` | 15min | 4GB    | 300MB     | Excellent |
| Full Enhanced       | 20min | 6GB    | 500MB     | Perfect   |

## 💡 Pro Tips

### For Quick Testing

```bash
# Just test visualization
python demo_depth_visualization.py

# Quick inpainting test
python run_enhanced.py --quick-test fast
```

### For Research

```bash
# Scientific colormap với detailed tracking
python run_enhanced.py --input image.png --colormap viridis --output-dir experiment_001
```

### For Presentations

```bash
# High contrast colormap
python run_enhanced.py --input image.png --colormap jet --output-dir presentation
```

### For Debugging

```bash
# Custom colormap shows depth intuitive
python run_enhanced.py --input image.png --colormap custom
```

## 🎯 Next Steps

1. **Test**: `python run_enhanced.py --demo`
2. **Learn**: Check `demo_depth_visualization/` results
3. **Use**: Run với your own images
4. **Compare**: Try different colormaps
5. **Integrate**: Use trong your research workflow

---

**Need help?** Check `README_DEPTH_VISUALIZATION.md` for details!
