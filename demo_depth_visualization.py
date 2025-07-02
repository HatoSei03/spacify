#!/usr/bin/env python3
"""
Demo script for testing depth visualization capabilities
Shows different colormaps and visualization modes
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from utils.depth_visualizer import DepthVisualizer, OutputManager

def create_sample_depth_map(height=512, width=1024):
    """Create a realistic panoramic depth map for testing"""
    # Create coordinate grids
    y, x = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    
    # Create various depth patterns
    # Central objects (closer)
    center_object = np.exp(-((x)**2 + (y*2)**2) * 3) * 3
    
    # Side walls (medium distance) 
    left_wall = np.exp(-((x + 0.8)**2) * 20) * 5
    right_wall = np.exp(-((x - 0.8)**2) * 20) * 5
    
    # Floor and ceiling gradients
    floor_gradient = np.maximum(0, (x + 1) * 2)
    ceiling_gradient = np.maximum(0, (-x + 1) * 2)
    
    # Background (far)
    background = np.ones_like(x) * 8
    
    # Combine all elements
    depth_map = background + center_object + left_wall + right_wall + floor_gradient * 0.5 + ceiling_gradient * 0.3
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.1, depth_map.shape)
    depth_map += noise
    
    # Ensure positive values
    depth_map = np.clip(depth_map, 0.1, 15.0)
    
    return depth_map

def create_sample_rgb_image(height=512, width=1024):
    """Create a sample RGB panoramic image"""
    # Create a simple panoramic scene
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient (top half)
    for i in range(height//3):
        intensity = int(135 + (255-135) * (1 - i/(height//3)))
        rgb[i, :, :] = [intensity//2, intensity//2, intensity]  # Blue sky
    
    # Horizon line
    horizon = height//3
    rgb[horizon:horizon+10, :, :] = [100, 150, 100]  # Green horizon
    
    # Floor (bottom half)
    for i in range(horizon+10, height):
        intensity = int(80 + 40 * (i-horizon-10)/(height-horizon-10))
        rgb[i, :, :] = [intensity, intensity//2, intensity//3]  # Brown floor
    
    # Add some objects
    # Central building
    rgb[horizon-50:horizon+100, width//2-100:width//2+100, :] = [120, 120, 120]  # Gray building
    
    # Side objects
    rgb[horizon-20:horizon+50, 100:200, :] = [150, 100, 80]  # Left object
    rgb[horizon-20:horizon+50, width-200:width-100, :] = [150, 100, 80]  # Right object
    
    return rgb

def demo_all_colormaps():
    """Demo all available colormaps"""
    print("🎨 Testing all colormaps...")
    
    # Create sample data
    depth_map = create_sample_depth_map()
    rgb_image = create_sample_rgb_image()
    
    # Create output directory
    demo_dir = "demo_depth_visualization"
    os.makedirs(demo_dir, exist_ok=True)
    
    colormaps = ['viridis', 'plasma', 'inferno', 'jet', 'turbo', 'magma', 'custom']
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Depth Visualization Colormap Comparison', fontsize=16)
    
    for idx, colormap in enumerate(colormaps):
        if idx >= 7:  # Skip if too many colormaps
            break
            
        row = idx // 4
        col = idx % 4
        
        print(f"  📊 Testing colormap: {colormap}")
        
        # Create visualizer
        visualizer = DepthVisualizer(colormap=colormap, normalize_mode='percentile')
        
        # Create visualization
        depth_vis = visualizer.create_depth_visualization(
            depth_map, 
            save_path=os.path.join(demo_dir, f"depth_{colormap}.png"),
            show_colorbar=True,
            title=f"Depth Map - {colormap.title()}"
        )
        
        # Add to comparison plot
        normalized_depth, vmin, vmax = visualizer.normalize_depth(depth_map)
        im = axes[row, col].imshow(normalized_depth, cmap=colormap)
        axes[row, col].set_title(f"{colormap.title()}\n({vmin:.2f} - {vmax:.2f})")
        axes[row, col].axis('off')
        
        # Create side-by-side
        visualizer.create_side_by_side(
            rgb_image, depth_map,
            save_path=os.path.join(demo_dir, f"combined_{colormap}.png"),
            title=f"RGB vs Depth ({colormap.title()})"
        )
    
    # Remove empty subplots
    if len(colormaps) < 8:
        for idx in range(len(colormaps), 8):
            row = idx // 4
            col = idx % 4
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(os.path.join(demo_dir, "colormap_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Colormap comparison saved to {demo_dir}/")

def demo_normalization_modes():
    """Demo different normalization modes"""
    print("📏 Testing normalization modes...")
    
    depth_map = create_sample_depth_map()
    demo_dir = "demo_depth_visualization"
    
    normalization_modes = ['percentile', 'minmax', 'fixed']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Depth Normalization Mode Comparison', fontsize=16)
    
    for idx, norm_mode in enumerate(normalization_modes):
        print(f"  📐 Testing normalization: {norm_mode}")
        
        visualizer = DepthVisualizer(colormap='viridis', normalize_mode=norm_mode)
        normalized_depth, vmin, vmax = visualizer.normalize_depth(depth_map)
        
        im = axes[idx].imshow(normalized_depth, cmap='viridis')
        axes[idx].set_title(f"{norm_mode.title()}\n({vmin:.3f} - {vmax:.3f})")
        axes[idx].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Save individual visualization
        visualizer.create_depth_visualization(
            depth_map,
            save_path=os.path.join(demo_dir, f"depth_norm_{norm_mode}.png"),
            title=f"Normalization: {norm_mode}"
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(demo_dir, "normalization_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

def demo_depth_difference():
    """Demo depth difference visualization"""
    print("🔍 Testing depth difference visualization...")
    
    # Create two similar depth maps
    depth1 = create_sample_depth_map()
    
    # Create second depth map with some differences
    depth2 = depth1.copy()
    depth2[100:200, 300:400] += 2.0  # Add some objects
    depth2[300:400, 600:700] -= 1.5  # Remove some depth
    
    demo_dir = "demo_depth_visualization"
    
    visualizer = DepthVisualizer()
    
    # Create difference visualization
    diff = visualizer.create_depth_difference(
        depth1, depth2,
        save_path=os.path.join(demo_dir, "depth_difference.png"),
        title="Depth Map Difference Analysis"
    )
    
    print(f"  📊 Max difference: {diff.max():.3f}")
    print(f"  📊 Mean difference: {diff.mean():.3f}")

def demo_output_manager():
    """Demo organized output management"""
    print("📁 Testing output management...")
    
    # Create output manager
    output_manager = OutputManager("demo_output_organized")
    
    # Generate some sample data
    for step in range(5):
        for direction in ['x', 'z']:
            paths = output_manager.get_paths(step, direction, 'all')
            
            # Create dummy data
            sample_rgb = create_sample_rgb_image()
            sample_depth = create_sample_depth_map()
            
            # Save organized
            Image.fromarray(sample_rgb).save(paths['rgb'])
            np.save(paths['depth_raw'], sample_depth)
            
            # Create visualizations
            visualizer = DepthVisualizer()
            visualizer.create_depth_visualization(
                sample_depth,
                save_path=paths['depth_vis'],
                title=f"Step {step} - {direction}"
            )
            
            visualizer.create_side_by_side(
                sample_rgb, sample_depth,
                save_path=paths['side_by_side'],
                title=f"Demo Step {step} - {direction}"
            )
            
            # Save metadata
            metadata = {
                'demo': True,
                'step': step,
                'direction': direction,
                'depth_stats': {
                    'min': float(sample_depth.min()),
                    'max': float(sample_depth.max()),
                    'mean': float(sample_depth.mean())
                }
            }
            output_manager.save_metadata(step, direction, metadata)
    
    print("✅ Organized output demo completed!")

def main():
    """Run all demos"""
    print("🚀 FastScene Depth Visualization Demo")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all demos
    demo_all_colormaps()
    demo_normalization_modes()
    demo_depth_difference()
    demo_output_manager()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("🎉 All demos completed successfully!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print("📁 Check the following directories for results:")
    print("   • demo_depth_visualization/")
    print("   • demo_output_organized/")
    print("\n💡 Usage tips:")
    print("   • Use 'viridis' for scientific visualization")
    print("   • Use 'jet' for high contrast")
    print("   • Use 'custom' for intuitive near=red, far=blue")
    print("   • Use 'percentile' normalization for robust results")

if __name__ == "__main__":
    main()