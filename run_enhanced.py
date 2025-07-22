#!/usr/bin/env python3
"""
Quick launcher for enhanced depth visualization system
Easy-to-use interface với presets và batch processing
"""

import argparse
import os
import sys
import time
from pathlib import Path

def setup_environment():
    """Setup environment và check dependencies"""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install torch torchvision matplotlib pillow numpy opencv-python")
        return False

def check_pretrained_models():
    """Check if pretrained models exist"""
    required_models = [
        'pretrained_models/EGformer_pretrained.pkl',
        'pretrained_models/G0185000.pt', 
        'models/RealESRGAN_x2plus.pth'
    ]
    
    missing = []
    for model in required_models:
        if not os.path.exists(model):
            missing.append(model)
    
    if missing:
        print("⚠️  Missing pretrained models:")
        for model in missing:
            print(f"   • {model}")
        print("📥 Please download required models first")
        return False
    
    print("✅ All pretrained models found")
    return True

def run_demo():
    """Run depth visualization demo"""
    print("🎨 Running depth visualization demo...")
    
    try:
        from demo_depth_visualization import main as demo_main
        demo_main()
        print("✅ Demo completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def run_enhanced_inpainting(args):
    """Run enhanced progressive inpainting"""
    print("🚀 Starting enhanced progressive inpainting...")
    
    # Build command
    cmd_args = [
        '--enable_depth_vis' if args.enable_vis else '--no-enable_depth_vis',
        '--depth_colormap', args.colormap,
        '--output_dir', args.output_dir
    ]
    
    # Add model paths if specified
    if args.checkpoint:
        cmd_args.extend(['--checkpoint_path', args.checkpoint])
    
    print(f"🔧 Configuration:")
    print(f"   • Depth Visualization: {'ON' if args.enable_vis else 'OFF'}")
    print(f"   • Colormap: {args.colormap}")
    print(f"   • Output Directory: {args.output_dir}")
    print(f"   • Input Image: {args.input_image}")
    
    try:
        # Import and run enhanced version
        sys.path.append('.')
        
        # Update config in enhanced version
        from pro_inpaint_enhanced import config, main as enhanced_main
        config.input_rgb = args.input_image
        config.output_dir = args.output_dir
        config.depth_colormap = args.colormap
        config.enable_depth_visualization = args.enable_vis
        
        result = enhanced_main()
        print(f"✅ Enhanced inpainting completed! Generated {result} images")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced inpainting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test(preset='fast'):
    """Quick test với preset configurations"""
    presets = {
        'fast': {
            'colormap': 'viridis',
            'enable_vis': True,
            'output_dir': 'test_fast'
        },
        'full': {
            'colormap': 'custom', 
            'enable_vis': True,
            'output_dir': 'test_full'
        },
        'scientific': {
            'colormap': 'plasma',
            'enable_vis': True,
            'output_dir': 'test_scientific'
        }
    }
    
    if preset not in presets:
        print(f"❌ Unknown preset: {preset}")
        print(f"Available presets: {list(presets.keys())}")
        return False
    
    config = presets[preset]
    print(f"🧪 Running quick test with '{preset}' preset...")
    
    # Create mock args
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = MockArgs(
        input_image='inpaint_data/demo.png',
        checkpoint=None,
        **config
    )
    
    return run_enhanced_inpainting(args)

def compare_outputs(dir1, dir2):
    """Compare two output directories"""
    print(f"🔍 Comparing outputs: {dir1} vs {dir2}")
    
    try:
        import numpy as np
        from utils.depth_visualizer import DepthVisualizer
        
        # Get depth files from both directories
        depth_dir1 = os.path.join(dir1, 'depth_maps')
        depth_dir2 = os.path.join(dir2, 'depth_maps') 
        
        if not (os.path.exists(depth_dir1) and os.path.exists(depth_dir2)):
            print("❌ Depth directories not found")
            return False
        
        depth_files1 = sorted([f for f in os.listdir(depth_dir1) if f.endswith('.npy')])
        depth_files2 = sorted([f for f in os.listdir(depth_dir2) if f.endswith('.npy')])
        
        if len(depth_files1) != len(depth_files2):
            print(f"⚠️  Different number of files: {len(depth_files1)} vs {len(depth_files2)}")
        
        visualizer = DepthVisualizer()
        comparison_dir = 'output_comparison'
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Compare matching files
        common_files = set(depth_files1) & set(depth_files2)
        
        for filename in sorted(common_files)[:5]:  # Limit to first 5
            print(f"   📊 Comparing {filename}...")
            
            depth1 = np.load(os.path.join(depth_dir1, filename))
            depth2 = np.load(os.path.join(depth_dir2, filename))
            
            # Create difference visualization
            diff_path = os.path.join(comparison_dir, f"diff_{filename.replace('.npy', '.png')}")
            visualizer.create_depth_difference(
                depth1, depth2,
                save_path=diff_path,
                title=f"Difference: {filename}"
            )
        
        print(f"✅ Comparison completed! Results in {comparison_dir}/")
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Depth Visualization Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  python run_enhanced.py --demo
  
  # Quick test
  python run_enhanced.py --quick-test fast
  
  # Full enhanced inpainting
  python run_enhanced.py --input inpaint_data/demo.png --colormap viridis
  
  # Compare two outputs
  python run_enhanced.py --compare output1 output2
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--demo', action='store_true',
                           help='Run depth visualization demo')
    mode_group.add_argument('--quick-test', choices=['fast', 'full', 'scientific'],
                           help='Run quick test with preset')
    mode_group.add_argument('--input', type=str,
                           help='Input image for enhanced inpainting')
    mode_group.add_argument('--compare', nargs=2, metavar=('DIR1', 'DIR2'),
                           help='Compare two output directories')
    
    # Enhanced inpainting options
    parser.add_argument('--colormap', default='viridis',
                       choices=['viridis', 'plasma', 'inferno', 'jet', 'turbo', 'magma', 'custom'],
                       help='Depth colormap (default: viridis)')
    parser.add_argument('--output-dir', default='output_enhanced',
                       help='Output directory (default: output_enhanced)')
    parser.add_argument('--disable-vis', action='store_true',
                       help='Disable depth visualization')
    parser.add_argument('--checkpoint', type=str,
                       help='Custom checkpoint path')
    
    args = parser.parse_args()
    
    print("🚀 FastScene Enhanced Depth Visualization Launcher")
    print("=" * 55)
    
    # Check environment
    if not setup_environment():
        sys.exit(1)
    
    success = False
    
    if args.demo:
        success = run_demo()
    
    elif args.quick_test:
        if not check_pretrained_models():
            sys.exit(1)
        success = quick_test(args.quick_test)
    
    elif args.input:
        if not check_pretrained_models():
            sys.exit(1)
        
        if not os.path.exists(args.input):
            print(f"❌ Input image not found: {args.input}")
            sys.exit(1)
        
        # Set up args for enhanced inpainting
        args.enable_vis = not args.disable_vis
        args.input_image = args.input
        success = run_enhanced_inpainting(args)
    
    elif args.compare:
        success = compare_outputs(args.compare[0], args.compare[1])
    
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()