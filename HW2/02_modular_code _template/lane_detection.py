import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import cv2
import glob
import os
import sys
import torch

try:
    sys.path.append(glob.glob('A:/Robot Learning/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (  # Kote directory
    #sys.path.append(glob.glob(
        #'C:/Users/bcerv/OneDrive/Documents/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (  # Rob directory
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=120, spline_smoothness=10, gradient_threshold=30, distance_maxima_gradient=3):
        self.car_position = np.array([160,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
    

    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.

        input:
            front_view_image)320x240x3

        output:
            bev_image 320x240x3

        '''
        def homography_ipmnorm2g(top_view_region):
            src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
            H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
            return H_ipmnorm2g

        # bev
        bev = np.zeros((200, 200, 3))
        H, W = 200, 200
        top_view_region = np.array([[50, -25], [50, 25], [0, -25], [0, 25]])

        # camera parameters
        cam_xyz = [0, 0, 2.0]
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
        H_g2cam = np.array(
            carla.Transform(carla.Location(*cam_xyz), carla.Rotation(yaw=cam_yaw), ).get_inverse_matrix())
        H_g2cam = np.concatenate([H_g2cam[:3, 0:2], np.expand_dims(H_g2cam[:3, 3], 1)], 1)

        trans_mat = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
        temp_mat = np.matmul(trans_mat, H_g2cam)
        H_g2im = np.matmul(K, temp_mat)

        H_ipmnorm2g = homography_ipmnorm2g(top_view_region)
        H_ipmnorm2im = np.matmul(H_g2im, H_ipmnorm2g)

        S_im_inv = np.array([[1 / float(width), 0, 0], [0, 1 / float(height), 0], [0, 0, 1]])
        M_ipm2im_norm = np.matmul(S_im_inv, H_ipmnorm2im)

        # visualization
        M = torch.zeros(1, 3, 3)
        M[0] = torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

        linear_points_W = torch.linspace(0, 1 - 1 / W, W)
        linear_points_H = torch.linspace(0, 1 - 1 / H, H)

        base_grid = torch.zeros(H, W, 3)
        base_grid[:, :, 0] = torch.ger(torch.ones(H), linear_points_W)
        base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(W))
        base_grid[:, :, 2] = 1

        grid = torch.matmul(base_grid.view(H * W, 3), M.transpose(1, 2))
        lst = grid[:, :, 2:].squeeze(0).squeeze(1).numpy() >= 0
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])

        x_vals = grid[0, :, 0].numpy() * width
        y_vals = grid[0, :, 1].numpy() * height

        indicate_x1 = x_vals < width
        indicate_x2 = x_vals > 0

        indicate_y1 = y_vals < height
        indicate_y2 = y_vals > 0

        indicate = (indicate_x1 * indicate_x2 * indicate_y1 * indicate_y2 * lst) * 1
        img = cv2.imread('front_view_camera.png')

        for _i in range(H):
            for _j in range(W):
                _idx = _j + _i * W

                _x = int(x_vals[_idx])
                _y = int(y_vals[_idx])
                _indic = indicate[_idx]

                if _indic == 0:
                    continue

                bev[_i, _j] = img[_y, _x]
        bev_image = np.uint8(bev)

        return bev_image


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the image at the front end of the car
        and translate to grey scale

        input:
            state_image_full 320x240x3

        output:
            gray_state_image 320x120x1

        '''
        state_image_cut = state_image_full[120:, :, :]
        gray_state_image = cv2.cvtColor(state_image_cut, cv2.COLOR_BGR2GRAY)
        
        return gray_state_image


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 320x120x1

        output:
            gradient_sum 320x120x1

        '''


        # Find gradients along x and y axes
        grad_x = np.abs(np.gradient(gray_image, axis=1))
        grad_y = np.abs(np.gradient(gray_image, axis=0))

        # Sum the gradients
        gradient_sum = grad_x + grad_y
        gradient_sum[gradient_sum < self.gradient_threshold] = 0

        
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 320x120x1

        output:
            maxima (np.array) 2x Number_maxima

        '''

        argmaxima = []

        for row in range(gradient_sum.shape[0]):
            peaks, _ = find_peaks(gradient_sum[row], distance=self.distance_maxima_gradient)
            argmaxima.append(peaks)

        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 320x120x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 160:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[320,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [320, 240, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        #bev
        bev_image = self.front2bev(state_image_full)
        cv2.imshow("BEV Image", bev_image)

        #grayscale
        gray_image = self.cut_gray(bev_image)
        cv2.imshow("Gray Image", gray_image)

        #edge detection
        gradient_sum = self.edge_detection(gray_image)

        #maxima
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)


        lane_boundary1_points = []
        lane_boundary2_points = []
        lanes_found = False

        #maxima to lane boundaries
        for row in range(len(maxima)):
            current_maxima = maxima[row]
            if len(current_maxima) == 0:
                continue  # Skip if no maxima found


            if row == 0:
                #more than two maxima, sort them by distance from car
                if len(current_maxima) > 2:
                    #two maxima closest to car
                    distances = np.abs(current_maxima - self.car_position[0])
                    closest_indices = np.argsort(distances)[:2]
                    lane_boundary1_points.append((current_maxima[closest_indices[0]], row))
                    lane_boundary2_points.append((current_maxima[closest_indices[1]], row))
                elif len(current_maxima) == 2:
                    lane_boundary1_points.append((current_maxima[0], row))
                    lane_boundary2_points.append((current_maxima[1], row))
                elif len(current_maxima) == 1:
                    lane_boundary1_points.append((current_maxima[0], row))
                    lane_boundary2_points.append((0, row))

            else:
                if lane_boundary1_points and lane_boundary2_points:
                    #last detected points for both boundaries
                    last_boundary1 = lane_boundary1_points[-1][0]
                    last_boundary2 = lane_boundary2_points[-1][0]

                    #distances to current maxima
                    dist_to_boundary1 = np.abs(current_maxima - last_boundary1)
                    dist_to_boundary2 = np.abs(current_maxima - last_boundary2)

                    #Get closest maxima
                    if len(current_maxima) > 0:
                        min_idx1 = np.argmin(dist_to_boundary1)
                        if dist_to_boundary1[min_idx1] < 100:  # Within a threshold distance
                            lane_boundary1_points.append((current_maxima[min_idx1], row))
                    if len(current_maxima) > 1:  #second maxima
                        min_idx2 = np.argmin(dist_to_boundary2)
                        if dist_to_boundary2[min_idx2] < 100:  # Within threshold
                            lane_boundary2_points.append((current_maxima[min_idx2], row))

        #Fit splines to the lane boundaries
        if len(lane_boundary1_points) > 4 and len(lane_boundary2_points) > 4:
            lane_boundary1, _ = splprep(np.array(lane_boundary1_points).T, s=self.spline_smoothness)
            lane_boundary2, _ = splprep(np.array(lane_boundary2_points).T, s=self.spline_smoothness)
            lanes_found = True
        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        #save current lane boundaries as old for next
        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[], target_speed=None):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        bev_image = self.front2bev(state_image_full)

        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))

        plt.gcf().clear()

        plt.imshow(bev_image[::-1])

        plt.plot(lane_boundary1_points_points[0], bev_image.shape[0] - lane_boundary1_points_points[1] - 120, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], bev_image.shape[0] - lane_boundary2_points_points[1] - 120, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], bev_image.shape[0] - waypoints[1] - 120, color='white')

        if target_speed is not None:
            plt.text(10, 20, f"Recommended Speed: {target_speed:.2f} m/s", fontsize=12, color='red')

        plt.axis('off')
        plt.xlim((-0.5,320))
        plt.ylim((-0.5,240))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()

