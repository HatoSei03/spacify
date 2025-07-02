import cv2
import os
import numpy as np
from PIL import Image
import torch
import imageio as io
import json
from scipy.interpolate import griddata
from utils.option import args
from models.egformer import EGDepthModel

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
rgb = 'inpaint_data/demo.png'

mov = 0.02
step = 12
min = 6

#----------------------------------------depth estimation-EGformer
parser = argparse.ArgumentParser()
parser.add_argument("--method",
                    type=str,
                    help="Method to be evaluated",
                    default="EGformer")

parser.add_argument("--eval_data",
                    type=str,
                    help="data category to be evaluated",
                    default="Inference")

parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--checkpoint_path', type=str, default='pretrained_models/EGformer_pretrained.pkl')

parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--multiprocessing_distributed', default=True)
parser.add_argument('--dist-backend', type=str, default="nccl")
parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")

config = parser.parse_args()

torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                     world_size=config.world_size, rank=config.rank)

net = EGDepthModel(hybrid=False)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0], find_unused_parameters=True)
net.load_state_dict(torch.load(config.checkpoint_path), strict=False)
net.eval()

#-------------------Inpainting AOT-GAN

net_inpa = importlib.import_module('model.' + args.model)
in_model = net_inpa.InpaintGenerator(args).to(device)
args.pre_train = 'pretrained_models/G0185000.pt'
in_model.load_state_dict(torch.load(args.pre_train, map_location=device))
in_model.eval()


#-------------------SR real-GAN
model_path = r'models/RealESRGAN_x2plus.pth'
dni_weight = None
sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2)

upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            dni_weight=dni_weight,
            model=sr_model,
            tile=384,
            tile_pad=20,
            pre_pad=20,
            half=False,
            device=device,
        )

# Cache depth estimation results
_depth_cache = {}

def depth_completion_cached(input_rgb):
    # Simple hash for caching
    rgb_hash = hash(input_rgb.tobytes())
    if rgb_hash in _depth_cache:
        return _depth_cache[rgb_hash]
    
    est_depth = depth_est(input_rgb, net)
    _depth_cache[rgb_hash] = est_depth
    return est_depth

def inpaint_image(mask_path, rgb_path):

    mask = mask_path
    rgb = rgb_path
    inpainted = inpaint(mask, rgb, in_model, upsampler)

    return inpainted

def depth_completion_optimized(input_rgb):
    with autocast():
        est_depth = depth_est(input_rgb, net)
    return est_depth


def translate(crd, rgb, d, cam=[]):
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
    # chuyển đổi tọa độ 3D sang tọa độ góc (spherical coordinates)
    theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    phi[idx] = np.arctan2(-z[idx], x[idx])

    # chuyển đổi tọa độ góc sang tọa độ pixel trong ảnh panorama
    x1[idx] = (0.5 - theta[idx] / np.pi) * H  # - 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
    y1[idx] = (0.5 - phi[idx] / (2 * np.pi)) * W  # - 0.5

    x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')

    img = np.zeros(rgb.shape)

    mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)

    # (522270,)
    x = x[mask]
    y = y[mask]
    new_d = new_d[mask]
    rgb = rgb[mask]

    # sắp xếp lại các điểm dựa trên độ sâu giảm dần, để xử lý che khuất
    reorder = np.argsort(-new_d)
    x = x[reorder]
    y = y[reorder]
    new_d = new_d[reorder]
    rgb = rgb[reorder]

    # Assign
    new_depth[x, y] = new_d
    img[x, y] = rgb

    mask = (new_depth != 0).astype(int)

    mask_index = np.argwhere(mask == 0)
    return img, new_depth.reshape(H, W, 1), tmp_coord, cam.reshape(1, 1, 3), mask, mask_index

# Cache cho tọa độ spherical
_spherical_cache = None

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

def generate_optimized(input_rgb, input_depth, flag, dir, first):
    """Phiên bản tối ưu của hàm generate"""
    H, W = 512, 1024
    
    rgb = input_rgb
    d = input_depth
    
    d_max = np.max(d)
    d = d / d_max
    d = d.reshape(rgb.shape[0], rgb.shape[1], 1)
    d = np.where(d == 0, 1, d)
    
    # Sử dụng cache thay vì tính toán lại
    coord = get_spherical_coords(H, W) * d
    
    # Tối ưu cam_pos calculation
    cam_pos_map = {
        'x': np.array([mov * flag, 0, 0]),
        'z': np.array([0, 0, mov * flag]),
        'xz': np.array([mov * flag, 0, mov * flag]),
        '-xz': np.array([mov * flag, 0, -mov * flag])
    }
    cam_pos = cam_pos_map[dir]
    
    img1, d1, _, _, mask1, mask_index = translate(coord, rgb, d, cam_pos)
    d1 = np.squeeze(d1, axis=-1)
    d1 = np.stack((d1, d1, d1), axis=-1)
    
    mask = np.uint8(mask1 * 255)
    img = np.uint8(img1)
    img[mask == 0] = 255
    mask = cv2.bitwise_not(mask)
    
    return mask, img, d1[:, :, 0], mask_index

def progressive_inpaint_optimized(ori_rgb, ori_depth):
    """Phiên bản tối ưu với batch processing"""
    num_inpaint = 0
    
    # Pre-calculate tất cả masks và images cho tất cả directions
    all_masks = []
    all_images = []
    all_depths = []
    
    directions = ['x', 'z', 'xz', '-xz']
    
    for dir_idx, direction in enumerate(directions):
        input_rgb = ori_rgb
        depth = ori_depth
        flag = 1
        
        for i in range(step):
            if i == min:
                flag = -1
                input_rgb = ori_rgb
                depth = ori_depth
                
            mask, img, depth, mask_index = generate_optimized(
                input_rgb, depth, flag, direction, (i == 0 or i == min)
            )
            
            all_masks.append(mask)
            all_images.append(img)
            all_depths.append(depth)
            
            # Batch inpaint thay vì từng cái một
            if len(all_masks) >= 4:  # Process batch of 4
                batch_masks = all_masks[-4:]
                batch_images = all_images[-4:]
                
                # Batch inpaint
                batch_results = []
                for m, im in zip(batch_masks, batch_images):
                    result = inpaint_image(m, im)
                    batch_results.append(result)
                
                # Batch depth estimation
                batch_depths = []
                for result in batch_results:
                    depth_result = depth_completion_optimized(result)
                    batch_depths.append(depth_result)
                
                # Update input cho iteration tiếp theo
                for j, (result, depth_result) in enumerate(zip(batch_results, batch_depths)):
                    idx = len(all_masks) - 4 + j
                    all_images[idx] = result
                    all_depths[idx] = depth_result
                    
                    Image.fromarray(result).save(f'Pano_inpaint/rgb_{num_inpaint}.png')
                    num_inpaint += 1
                    print(f'num_image {num_inpaint}')
    
    return num_inpaint

# load image and depth
rgb = np.array(Image.open(rgb).convert('RGB'))
depth = depth_est(rgb, net)
progressive_inpaint_optimized(ori_rgb=rgb, ori_depth=depth)

