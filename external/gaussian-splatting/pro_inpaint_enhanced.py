import cv2
import os
import numpy as np
from PIL import Image
import torch
import imageio as io
import json
import time
from scipy.interpolate import griddata
from utils.option import args
from models.egformer import EGDepthModel
from utils.depth_visualizer import DepthVisualizer, OutputManager

import argparse
import importlib
from inpaint import inpaint
from evaluate.depest import depth_est

#--------sr
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

import concurrent.futures
from torch.cuda.amp import autocast

device = 'cuda:0'

# Enhanced configuration
class Config:
    def __init__(self):
        # Input/Output settings
        self.input_rgb = 'inpaint_data/demo.png'
        self.output_dir = 'output_progressive'
        
        # Depth visualization settings
        self.enable_depth_visualization = True
        self.depth_colormap = 'viridis'  # viridis, plasma, inferno, jet, turbo, custom
        self.depth_normalize_mode = 'percentile'  # percentile, minmax, fixed
        self.save_side_by_side = True
        self.save_depth_raw = True
        
        # Processing settings
        self.movement_step = 0.02
        self.total_steps = 12
        self.reverse_at_step = 6
        self.directions = ['x', 'z', 'xz', '-xz']
        
        # Performance settings
        self.batch_size = 4
        self.enable_progress_tracking = True
        self.enable_caching = True

config = Config()

#----------------------------------------depth estimation-EGformer
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, help="Method to be evaluated", default="EGformer")
parser.add_argument("--eval_data", type=str, help="data category to be evaluated", default="Inference")
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--checkpoint_path', type=str, default='pretrained_models/EGformer_pretrained.pkl')
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--multiprocessing_distributed', default=True)
parser.add_argument('--dist-backend', type=str, default="nccl")
parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")

# Add visualization arguments
parser.add_argument('--enable_depth_vis', action='store_true', default=True, 
                   help='Enable depth visualization')
parser.add_argument('--depth_colormap', type=str, default='viridis',
                   choices=['viridis', 'plasma', 'inferno', 'jet', 'turbo', 'custom'],
                   help='Colormap for depth visualization')
parser.add_argument('--output_dir', type=str, default='output_progressive',
                   help='Output directory for results')

args_parsed = parser.parse_args()

# Update config from args
config.enable_depth_visualization = args_parsed.enable_depth_vis
config.depth_colormap = args_parsed.depth_colormap  
config.output_dir = args_parsed.output_dir

# Initialize models
torch.distributed.init_process_group(backend=args_parsed.dist_backend, init_method=args_parsed.dist_url,
                                     world_size=args_parsed.world_size, rank=args_parsed.rank)

print("🚀 Initializing models...")
net = EGDepthModel(hybrid=False)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0], find_unused_parameters=True)
net.load_state_dict(torch.load(args_parsed.checkpoint_path), strict=False)
net.eval()

#-------------------Inpainting AOT-GAN
net_inpa = importlib.import_module('model.' + args.model)
in_model = net_inpa.InpaintGenerator(args).to(device)
args.pre_train = 'pretrained_models/G0185000.pt'
in_model.load_state_dict(torch.load(args.pre_train, map_location=device))
in_model.eval()

#-------------------SR real-GAN
model_path = r'models/RealESRGAN_x2plus.pth'
sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(scale=2, model_path=model_path, model=sr_model, 
                        tile=384, tile_pad=20, pre_pad=20, half=False, device=device)

# Initialize visualization and output management
print("📁 Setting up output directories...")
output_manager = OutputManager(config.output_dir)
depth_visualizer = DepthVisualizer(colormap=config.depth_colormap, 
                                 normalize_mode=config.depth_normalize_mode)

# Cache for performance
_depth_cache = {}
_spherical_cache = None

def depth_completion_optimized(input_rgb):
    """Optimized depth estimation with caching"""
    if config.enable_caching:
        rgb_hash = hash(input_rgb.tobytes())
        if rgb_hash in _depth_cache:
            return _depth_cache[rgb_hash]
    
    with autocast():
        est_depth = depth_est(input_rgb, net)
    
    if config.enable_caching:
        _depth_cache[rgb_hash] = est_depth
    
    return est_depth

def get_spherical_coords(H=512, W=1024):
    """Cache spherical coordinates để tránh tính toán lặp lại"""
    global _spherical_cache
    if _spherical_cache is None:
        _y = np.repeat(np.array(range(W)).reshape(1, W), H, axis=0)
        _x = np.repeat(np.array(range(H)).reshape(1, H), W, axis=0).T
        
        _theta = (1 - 2 * (_x) / H) * np.pi / 2
        _phi = 2 * np.pi * (0.5 - (_y) / W)
        
        axis0 = (np.cos(_theta) * np.cos(_phi)).reshape(H, W, 1)
        axis1 = np.sin(_theta).reshape(H, W, 1)
        axis2 = (-np.cos(_theta) * np.sin(_phi)).reshape(H, W, 1)
        
        _spherical_cache = np.concatenate((axis0, axis1, axis2), axis=2)
    
    return _spherical_cache

def translate(crd, rgb, d, cam=[]):
    """Enhanced translate function với progress tracking"""
    H, W = rgb.shape[0], rgb.shape[1]
    d = np.where(d == 0, -1, d)

    tmp_coord = crd - cam
    new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2))
    new_coord = tmp_coord / new_d.reshape(H, W, 1)

    new_depth = np.zeros(new_d.shape)
    [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]
    idx = np.where(new_d > 0)

    theta = np.zeros(y.shape)
    phi = np.zeros(y.shape)
    x1 = np.zeros(z.shape)
    y1 = np.zeros(z.shape)
    
    # Spherical coordinate conversion
    theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    phi[idx] = np.arctan2(-z[idx], x[idx])

    # Convert to panorama coordinates
    x1[idx] = (0.5 - theta[idx] / np.pi) * H
    y1[idx] = (0.5 - phi[idx] / (2 * np.pi)) * W

    x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')
    img = np.zeros(rgb.shape)
    mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)

    # Depth sorting for occlusion handling
    x = x[mask]
    y = y[mask]
    new_d = new_d[mask]
    rgb = rgb[mask]

    reorder = np.argsort(-new_d)
    x = x[reorder]
    y = y[reorder]
    new_d = new_d[reorder]
    rgb = rgb[reorder]

    # Assign values
    new_depth[x, y] = new_d
    img[x, y] = rgb
    mask = (new_depth != 0).astype(int)
    mask_index = np.argwhere(mask == 0)
    
    return img, new_depth.reshape(H, W, 1), tmp_coord, cam.reshape(1, 1, 3), mask, mask_index

def generate_optimized(input_rgb, input_depth, flag, direction, step_idx):
    """Enhanced generate function with improved caching"""
    H, W = 512, 1024
    
    rgb = input_rgb
    d = input_depth
    
    d_max = np.max(d)
    d = d / d_max if d_max > 0 else d
    d = d.reshape(rgb.shape[0], rgb.shape[1], 1)
    d = np.where(d == 0, 1, d)
    
    # Use cached spherical coordinates
    coord = get_spherical_coords(H, W) * d
    
    # Camera position mapping
    cam_pos_map = {
        'x': np.array([config.movement_step * flag, 0, 0]),
        'z': np.array([0, 0, config.movement_step * flag]),
        'xz': np.array([config.movement_step * flag, 0, config.movement_step * flag]),
        '-xz': np.array([config.movement_step * flag, 0, -config.movement_step * flag])
    }
    cam_pos = cam_pos_map[direction]
    
    img1, d1, _, _, mask1, mask_index = translate(coord, rgb, d, cam_pos)
    d1 = np.squeeze(d1, axis=-1)
    d1 = np.stack((d1, d1, d1), axis=-1)
    
    mask = np.uint8(mask1 * 255)
    img = np.uint8(img1)
    img[mask == 0] = 255
    mask = cv2.bitwise_not(mask)
    
    return mask, img, d1[:, :, 0], mask_index

def save_results_enhanced(rgb_result, depth_result, step_idx, direction, paths, metadata):
    """Enhanced saving with comprehensive visualization"""
    # Save RGB
    Image.fromarray(rgb_result).save(paths['rgb'])
    
    if config.enable_depth_visualization:
        # Save raw depth data
        if config.save_depth_raw:
            np.save(paths['depth_raw'], depth_result)
        
        # Create depth visualization
        depth_vis = depth_visualizer.create_depth_visualization(
            depth_result, 
            save_path=paths['depth_vis'],
            title=f"Step {step_idx} - {direction}"
        )
        
        # Create side-by-side comparison
        if config.save_side_by_side:
            combined = depth_visualizer.create_side_by_side(
                rgb_result, depth_result,
                save_path=paths['side_by_side'],
                title=f"Progressive Inpainting - Step {step_idx} ({direction})"
            )
    
    # Save metadata
    if config.enable_progress_tracking:
        output_manager.save_metadata(step_idx, direction, metadata)

def progressive_inpaint_enhanced(ori_rgb, ori_depth):
    """Enhanced progressive inpainting với comprehensive visualization"""
    print("🎨 Starting enhanced progressive inpainting...")
    
    total_steps = len(config.directions) * config.total_steps
    current_step = 0
    
    # Initialize progress tracking
    progress_data = {
        'total_steps': total_steps,
        'start_time': time.time(),
        'steps_completed': [],
        'processing_times': []
    }
    
    for dir_idx, direction in enumerate(config.directions):
        print(f"\n📐 Processing direction: {direction} ({dir_idx+1}/{len(config.directions)})")
        
        input_rgb = ori_rgb.copy()
        depth = ori_depth.copy()
        flag = 1
        
        for i in range(config.total_steps):
            step_start_time = time.time()
            current_step += 1
            
            # Reverse direction at halfway point
            if i == config.reverse_at_step:
                flag = -1
                input_rgb = ori_rgb.copy()
                depth = ori_depth.copy()
                
            print(f"  🔄 Step {i+1}/{config.total_steps} (Overall: {current_step}/{total_steps})")
            
            # Generate novel view
            mask, img, depth_2d, mask_index = generate_optimized(
                input_rgb, depth, flag, direction, current_step
            )
            
            # Inpaint
            inpainted = inpaint(mask, img, in_model, upsampler)
            
            # Depth estimation
            new_depth = depth_completion_optimized(inpainted)
            
            # Get organized paths
            paths = output_manager.get_paths(current_step, direction, 'all')
            
            # Prepare metadata
            step_time = time.time() - step_start_time
            metadata = {
                'direction': direction,
                'step_in_direction': i + 1,
                'flag': flag,
                'processing_time': step_time,
                'depth_stats': {
                    'min': float(new_depth.min()),
                    'max': float(new_depth.max()),
                    'mean': float(new_depth.mean()),
                    'std': float(new_depth.std())
                }
            }
            
            # Save results with enhanced visualization
            save_results_enhanced(inpainted, new_depth, current_step, direction, paths, metadata)
            
            # Update for next iteration
            input_rgb = inpainted.copy()
            depth = new_depth.copy()
            
            # Update progress tracking
            progress_data['steps_completed'].append(current_step)
            progress_data['processing_times'].append(step_time)
            
            print(f"    ✅ Completed in {step_time:.2f}s")
    
    # Final progress report
    total_time = time.time() - progress_data['start_time']
    avg_time_per_step = np.mean(progress_data['processing_times'])
    
    final_report = {
        'total_processing_time': total_time,
        'average_time_per_step': avg_time_per_step,
        'total_steps_completed': current_step,
        'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_used': {
            'colormap': config.depth_colormap,
            'normalize_mode': config.depth_normalize_mode,
            'directions': config.directions,
            'total_steps_per_direction': config.total_steps
        }
    }
    
    # Save final report
    report_path = os.path.join(config.output_dir, 'final_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n🎉 Progressive inpainting completed!")
    print(f"📊 Total time: {total_time:.2f}s ({avg_time_per_step:.2f}s/step)")
    print(f"📁 Results saved to: {config.output_dir}")
    print(f"📋 Report saved to: {report_path}")
    
    return current_step

def main():
    """Main execution function"""
    print("🚀 FastScene Enhanced Progressive Inpainting")
    print("=" * 50)
    
    # Load initial RGB and estimate depth
    print("📷 Loading input image...")
    rgb = np.array(Image.open(config.input_rgb).convert('RGB'))
    print(f"Image shape: {rgb.shape}")
    
    print("🧠 Estimating initial depth...")
    depth = depth_est(rgb, net)
    print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")
    
    # Save initial visualizations
    if config.enable_depth_visualization:
        print("🎨 Creating initial depth visualization...")
        initial_paths = {
            'rgb': os.path.join(config.output_dir, 'initial_rgb.png'),
            'depth_vis': os.path.join(config.output_dir, 'initial_depth_vis.png'),
            'side_by_side': os.path.join(config.output_dir, 'initial_combined.png'),
            'depth_raw': os.path.join(config.output_dir, 'initial_depth.npy')
        }
        
        Image.fromarray(rgb).save(initial_paths['rgb'])
        np.save(initial_paths['depth_raw'], depth)
        
        depth_visualizer.create_depth_visualization(
            depth, 
            save_path=initial_paths['depth_vis'],
            title="Initial Depth Estimation"
        )
        
        depth_visualizer.create_side_by_side(
            rgb, depth,
            save_path=initial_paths['side_by_side'],
            title="Initial RGB vs Depth"
        )
    
    # Run enhanced progressive inpainting
    num_images = progressive_inpaint_enhanced(ori_rgb=rgb, ori_depth=depth)
    
    print(f"\n✨ Generated {num_images} enhanced image pairs!")
    return num_images

if __name__ == "__main__":
    main()