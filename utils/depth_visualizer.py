import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image
import os
from typing import Optional, Tuple, Union

class DepthVisualizer:
    """Advanced depth map visualization utility with multiple colormaps and modes"""
    
    def __init__(self, colormap='viridis', normalize_mode='percentile'):
        """
        Args:
            colormap: 'viridis', 'plasma', 'inferno', 'jet', 'turbo', 'custom'
            normalize_mode: 'percentile', 'minmax', 'fixed'
        """
        self.colormap = colormap
        self.normalize_mode = normalize_mode
        self.supported_colormaps = ['viridis', 'plasma', 'inferno', 'jet', 'turbo', 'magma', 'custom']
        
    def normalize_depth(self, depth_map: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
        """Normalize depth map based on specified mode"""
        depth_clean = depth_map.copy()
        
        if self.normalize_mode == 'percentile':
            if vmin is None:
                vmin = np.percentile(depth_clean, 5)
            if vmax is None:  
                vmax = np.percentile(depth_clean, 95)
        elif self.normalize_mode == 'minmax':
            if vmin is None:
                vmin = depth_clean.min()
            if vmax is None:
                vmax = depth_clean.max()
        elif self.normalize_mode == 'fixed':
            if vmin is None:
                vmin = 0.0
            if vmax is None:
                vmax = 10.0
                
        # Clamp and normalize
        depth_clean = np.clip(depth_clean, vmin, vmax)
        if vmax > vmin:
            depth_clean = (depth_clean - vmin) / (vmax - vmin)
        else:
            depth_clean = np.zeros_like(depth_clean)
            
        return depth_clean, vmin, vmax
    
    def apply_colormap(self, normalized_depth: np.ndarray) -> np.ndarray:
        """Apply colormap to normalized depth"""
        if self.colormap == 'custom':
            # Custom depth colormap: near=red, mid=green, far=blue
            colored = np.zeros((*normalized_depth.shape, 3))
            colored[..., 0] = 1.0 - normalized_depth  # Red for near
            colored[..., 1] = 1.0 - np.abs(normalized_depth - 0.5) * 2  # Green for mid
            colored[..., 2] = normalized_depth  # Blue for far
        else:
            cmap = plt.get_cmap(self.colormap)
            colored = cmap(normalized_depth)[..., :3]  # Remove alpha channel
            
        return (colored * 255).astype(np.uint8)
    
    def create_depth_visualization(self, depth_map: np.ndarray, 
                                 save_path: Optional[str] = None,
                                 show_colorbar: bool = True,
                                 title: str = "Depth Map") -> np.ndarray:
        """Create depth visualization with colorbar"""
        normalized_depth, vmin, vmax = self.normalize_depth(depth_map)
        colored_depth = self.apply_colormap(normalized_depth)
        
        if show_colorbar and save_path:
            # Create figure with colorbar
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            im = ax.imshow(normalized_depth, cmap=self.colormap, vmin=0, vmax=1)
            ax.set_title(f"{title}\n(Range: {vmin:.3f} - {vmax:.3f})")
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Depth Value', rotation=270, labelpad=20)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        if save_path and not show_colorbar:
            Image.fromarray(colored_depth).save(save_path)
            
        return colored_depth
    
    def create_side_by_side(self, rgb_image: np.ndarray, depth_map: np.ndarray, 
                           save_path: str, title: str = "RGB vs Depth") -> np.ndarray:
        """Create side-by-side RGB and depth visualization"""
        depth_vis = self.create_depth_visualization(depth_map, show_colorbar=False)
        
        # Ensure same height
        h = min(rgb_image.shape[0], depth_vis.shape[0])
        rgb_resized = cv2.resize(rgb_image, (rgb_image.shape[1], h))
        depth_resized = cv2.resize(depth_vis, (depth_vis.shape[1], h))
        
        # Concatenate horizontally
        combined = np.hstack([rgb_resized, depth_resized])
        
        # Save with matplotlib for better control
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(rgb_resized)
        ax1.set_title("RGB Image")
        ax1.axis('off')
        
        normalized_depth, vmin, vmax = self.normalize_depth(depth_map)
        im = ax2.imshow(normalized_depth, cmap=self.colormap)
        ax2.set_title(f"Depth Map ({vmin:.3f} - {vmax:.3f})")
        ax2.axis('off')
        
        # Add colorbar to depth subplot
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Depth', rotation=270, labelpad=15)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return combined
    
    def create_depth_difference(self, depth1: np.ndarray, depth2: np.ndarray,
                               save_path: str, title: str = "Depth Difference") -> np.ndarray:
        """Visualize difference between two depth maps"""
        diff = np.abs(depth1 - depth2)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=np.percentile(diff, 95))
        ax.set_title(title)
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Difference', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return diff

class OutputManager:
    """Manage organized output folders and files"""
    
    def __init__(self, base_output_dir: str = "output_progressive"):
        self.base_dir = base_output_dir
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized directory structure"""
        dirs = [
            'rgb_images',
            'depth_maps',
            'depth_visualizations', 
            'side_by_side',
            'progress_tracking',
            'metadata'
        ]
        
        for dir_name in dirs:
            dir_path = os.path.join(self.base_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            
    def get_paths(self, step: int, direction: str, file_type: str) -> dict:
        """Get organized file paths for saving"""
        prefix = f"step_{step:03d}_{direction}"
        
        paths = {
            'rgb': os.path.join(self.base_dir, 'rgb_images', f"{prefix}_rgb.png"),
            'depth_raw': os.path.join(self.base_dir, 'depth_maps', f"{prefix}_depth.npy"),
            'depth_vis': os.path.join(self.base_dir, 'depth_visualizations', f"{prefix}_depth_vis.png"),
            'side_by_side': os.path.join(self.base_dir, 'side_by_side', f"{prefix}_combined.png"),
            'progress': os.path.join(self.base_dir, 'progress_tracking', f"progress_{step:03d}.json")
        }
        
        return paths
    
    def save_metadata(self, step: int, direction: str, metadata: dict):
        """Save processing metadata"""
        import json
        
        paths = self.get_paths(step, direction, 'metadata')
        metadata_path = paths['progress']
        
        # Add timestamp
        import time
        metadata['timestamp'] = time.time()
        metadata['step'] = step
        metadata['direction'] = direction
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)