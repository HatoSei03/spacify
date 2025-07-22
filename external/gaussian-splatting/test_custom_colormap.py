#!/usr/bin/env python3
"""
Quick test for custom colormap fix
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.depth_visualizer import DepthVisualizer
import os

def test_custom_colormap():
    """Test custom colormap functionality"""
    print("🧪 Testing custom colormap fix...")
    
    # Create sample depth map
    depth_map = np.random.rand(100, 200) * 10  # Random depth values 0-10
    
    # Test custom colormap
    try:
        visualizer = DepthVisualizer(colormap='custom', normalize_mode='percentile')
        
        # Test single depth visualization
        print("  📊 Testing single depth visualization...")
        depth_vis = visualizer.create_depth_visualization(
            depth_map,
            save_path='test_custom_single.png',
            title='Test Custom Colormap'
        )
        print("  ✅ Single visualization successful")
        
        # Test RGB image
        rgb_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Test side-by-side
        print("  📊 Testing side-by-side visualization...")
        combined = visualizer.create_side_by_side(
            rgb_image, depth_map,
            save_path='test_custom_combined.png',
            title='Test Custom Side-by-Side'
        )
        print("  ✅ Side-by-side visualization successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Custom colormap test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_colormaps():
    """Test that all colormaps work"""
    print("🎨 Testing all colormaps...")
    
    colormaps = ['viridis', 'plasma', 'inferno', 'jet', 'turbo', 'magma', 'custom']
    depth_map = np.random.rand(50, 100) * 5
    
    success_count = 0
    
    for colormap in colormaps:
        try:
            print(f"  📊 Testing {colormap}...")
            visualizer = DepthVisualizer(colormap=colormap)
            
            # Test basic visualization
            depth_vis = visualizer.create_depth_visualization(
                depth_map,
                save_path=f'test_{colormap}.png',
                title=f'Test {colormap.title()}'
            )
            
            print(f"  ✅ {colormap} successful")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ {colormap} failed: {e}")
    
    print(f"📊 Results: {success_count}/{len(colormaps)} colormaps working")
    return success_count == len(colormaps)

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        'test_custom_single.png',
        'test_custom_combined.png'
    ] + [f'test_{cmap}.png' for cmap in ['viridis', 'plasma', 'inferno', 'jet', 'turbo', 'magma', 'custom']]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  🗑️ Removed {file}")

def main():
    print("🚀 Custom Colormap Fix Test")
    print("=" * 30)
    
    success = True
    
    # Test custom colormap specifically
    if not test_custom_colormap():
        success = False
    
    # Test all colormaps
    if not test_all_colormaps():
        success = False
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    cleanup_test_files()
    
    if success:
        print("\n🎉 All tests passed! Custom colormap fix is working!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    main() 