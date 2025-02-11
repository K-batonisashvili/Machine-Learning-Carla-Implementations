import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev

# Load the front view camera image
original_image = cv2.imread('A:/Robot Learning/HW2/front_view_camera.png')

# Set dimensions for the output birds-eye view image
image_width = 200
image_height = 200
#
# # Define source points (corners of lanes in original image)
# src = np.array([[150, 240], [200, 180], [300, 180], [350, 240]], dtype=np.float32)
#
# # Define destination points (mapped to BEV space)
# dest = np.array([[40, 200], [40, 0], [160, 0], [160, 200]], dtype=np.float32)
#
# # Calculate perspective transform matrix
# H = cv2.getPerspectiveTransform(src, dest)
#
# # Warp the perspective to get the BEV image
# bev_image = cv2.warpPerspective(original_image, H, (image_width, image_height))




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
fov = 55

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

		bev[_i,_j] = original_image[_y, _x]

cv2.imwrite('bev_front_view_camera.png', np.uint8(bev))

bev_image = np.uint8(bev)

# Show the BEV image
cv2.imshow("BEV Image", bev_image)

# Cut the lower part of the BEV image for lane detection
state_image_cut = bev_image[120:, :, :]
cv2.imshow("Cut Image", state_image_cut)

# Convert the cut image to grayscale
gray_state_image = cv2.cvtColor(state_image_cut, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_state_image)

# Find gradients along the x and y axes
gradient_x, gradient_y = np.gradient(gray_state_image)

# Calculate the magnitude of gradients
gradient_sum = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

# Apply threshold to suppress weak gradients
gradient_threshold = 20  # Adjust threshold based on your image
gradient_sum[gradient_sum < gradient_threshold] = 0

# Initialize arrays to store detected peaks
argmaxima = []
for row in range(gradient_sum.shape[0]):
    peaks, _ = find_peaks(gradient_sum[row], distance=3)
    argmaxima.append(peaks)

# Initialize lane boundary points
lane_boundary1_points = np.array([[50, 0]])  # Starting point for lane 1
lane_boundary2_points = np.array([[150, 0]])  # Starting point for lane 2

# Detect lane boundaries using peaks
for row in range(len(argmaxima) - 1):
    current_maxima = argmaxima[row]

    if len(current_maxima) != 0:
        # Calculate distances to previous lane boundary points
        dist_to_boundary_1 = np.abs(current_maxima - lane_boundary1_points[-1, 0])
        dist_to_boundary_2 = np.abs(current_maxima - lane_boundary2_points[-1, 0])

        # Find closest points to current lane boundaries
        closest_to_boundary1_index = np.argmin(dist_to_boundary_1)
        closest_to_boundary2_index = np.argmin(dist_to_boundary_2)

        # Append new points to lane boundary arrays
        new_lane_boundary1_point = [[current_maxima[closest_to_boundary1_index], row + 1]]
        new_lane_boundary2_point = [[current_maxima[closest_to_boundary2_index], row + 1]]

        lane_boundary1_points = np.append(lane_boundary1_points, new_lane_boundary1_point, axis=0)
        lane_boundary2_points = np.append(lane_boundary2_points, new_lane_boundary2_point, axis=0)

# Fit splines to lane boundary points
lane_boundary1_x_values = lane_boundary1_points[:, 0]
lane_boundary1_y_values = lane_boundary1_points[:, 1]
lane_boundary1 = splprep([lane_boundary1_x_values, lane_boundary1_y_values], s=0)

lane_boundary2_x_values = lane_boundary2_points[:, 0]
lane_boundary2_y_values = lane_boundary2_points[:, 1]
lane_boundary2 = splprep([lane_boundary2_x_values, lane_boundary2_y_values], s=0)

# Generate spline points for lane boundaries
t_values = np.linspace(0, 1, 6)
lane1_spline_points = splev(t_values, lane_boundary1[0])
lane2_spline_points = splev(t_values, lane_boundary2[0])

# Plot splines on the image
spline1_points = np.vstack((lane1_spline_points[0], lane1_spline_points[1])).T.reshape((-1, 1, 2))
spline2_points = np.vstack((lane2_spline_points[0], lane2_spline_points[1])).T.reshape((-1, 1, 2))

spline_image = cv2.polylines(state_image_cut, [spline1_points.astype(np.int32)], False, (0, 0, 255), 2)
spline_image = cv2.polylines(spline_image, [spline2_points.astype(np.int32)], False, (0, 0, 255), 2)

# Display the image with lane boundaries
cv2.imshow("Splined Image", spline_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
