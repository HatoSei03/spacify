import os
import sys
import cv2
import time
from imageio import imread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Equirec2Cube(nn.Module):
    def __init__(self, cube_dim, equ_h, FoV=90.0):
        super().__init__()
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([
            [0, -180.0, 0], #mặt sau
            [90.0, 0, 0], #mặt trên
            [0, 0, 0], #mặt trước
            [0, 90, 0], #mặt phải
            [0, -90, 0], #mặt trái
            [-90, 0, 0] #mặt dưới
        ], np.float32) / 180.0 * np.pi
        # chuyển đổi các góc quay sang ma trận quay
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        grids = self._getCubeGrid()
        
        for i, grid in enumerate(grids):
            self.register_buffer('grid_%d'%i, grid)

    def _getCubeGrid(self):
        # Tính nội tham số của camera
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV) # tiêu cự
        cx = (self.cube_dim - 1) / 2 # tọa độ trung tâm của khối lập phương
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        # ma trận nội của camera
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        #self.grids = []
        grids = []
        for _, R in enumerate(self.R_lst):
            tmp = xyz @ R # Don't need to transpose since we are doing it for points not for camera
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            grids.append(torch.FloatTensor(lonlat[None, ...]))
        
        return grids
    
    def forward(self, batch, mode='bilinear'):
        [_, _, h, w] = batch.shape
        assert h == self.equ_h and w == self.equ_w
        assert mode in ['nearest', 'bilinear', 'bicubic']

        out = []
        for i in range(6):
            grid = getattr(self, 'grid_%d'%i)
            grid = grid.repeat(batch.shape[0], 1, 1, 1)
            sample = F.grid_sample(batch, grid, mode=mode, align_corners=True)
            out.append(sample)
        out = torch.cat(out, dim=0)
        final_out = []
        for i in range(batch.shape[0]):
            final_out.append(out[i::batch.shape[0], ...])
        final_out = torch.cat(final_out, dim=0)
        return final_out


