import cv2
import numpy as np
import torch
import glob
import os
import sys


try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def homography_ipmnorm2g(top_view_region):
    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g


# bev
bev = np.zeros((200, 200, 3))
H, W = 200, 200
top_view_region = np.array([[50, -25], [50, 25], [0, -25], [0, 25]])


# camera parameters
cam_xyz = [-1.5, 0, 2.0]
cam_yaw = 0

width = 320
height = 240
fov = 90

# camera intrinsic
focal = width / (2.0 * np.tan(fov * np.pi / 360.0)) 
K = np.identity(3)
K[0, 0] = K[1, 1] = focal
K[0, 2] = width / 2.0
K[1, 2] = height / 2.0


# get IPM
H_g2cam = np.array(carla.Transform(carla.Location(*cam_xyz),carla.Rotation(yaw=cam_yaw),).get_inverse_matrix())
H_g2cam = np.concatenate([H_g2cam[:3, 0:2], np.expand_dims(H_g2cam[:3, 3],1)], 1)

trans_mat = np.array([[0,1,0], [0,0,-1], [1,0,0]])
temp_mat = np.matmul(trans_mat, H_g2cam)
H_g2im = np.matmul(K, temp_mat)

H_ipmnorm2g = homography_ipmnorm2g(top_view_region)
H_ipmnorm2im = np.matmul(H_g2im, H_ipmnorm2g)

S_im_inv = np.array([[1/float(width), 0, 0], [0, 1/float(height), 0], [0, 0, 1]])
M_ipm2im_norm = np.matmul(S_im_inv, H_ipmnorm2im)



# visualization
M = torch.zeros(1, 3, 3)
M[0]=torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

linear_points_W = torch.linspace(0, 1 - 1/W, W)
linear_points_H = torch.linspace(0, 1 - 1/H, H)

base_grid = torch.zeros(H, W, 3)
base_grid[:, :, 0] = torch.ger(torch.ones(H), linear_points_W)
base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(W))
base_grid[:, :, 2] = 1

grid = torch.matmul(base_grid.view(H * W, 3), M.transpose(1, 2))
lst = grid[:, :, 2:].squeeze(0).squeeze(1).numpy() >= 0
grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])

x_vals = grid[0,:,0].numpy() * width
y_vals = grid[0,:,1].numpy() * height

indicate_x1 = x_vals < width
indicate_x2 = x_vals > 0

indicate_y1 = y_vals < height
indicate_y2 = y_vals > 0

indicate = (indicate_x1 * indicate_x2 * indicate_y1 * indicate_y2 * lst)*1

img = cv2.imread('front_view_camera.png')

for _i in range(H):
	for _j in range(W):
		_idx = _j + _i*W

		_x = int(x_vals[_idx])
		_y = int(y_vals[_idx])
		_indic = indicate[_idx]

		if _indic == 0:
			continue

		bev[_i,_j] = img[_y, _x]

cv2.imwrite('bev_front_view_camera.png', np.uint8(bev))