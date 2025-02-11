import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=120, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
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

        image_width = 320
        image_height = 240

        src = np.array([[162, 238], [194, 194], [287, 194], [320, 238]], dtype=np.float32)
        dest = np.array([[93, 176], [93, 156], [106, 156], [106, 176]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dest)
        bev_image = cv2.warpPerspective(front_view_image, H, (image_width, image_height))

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

        gray_state_image = np.dot(state_image_cut[..., :3], [.3, .6, .1])
        
        return gray_state_image[::-1] 


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

        # Find gradients along x and y axes of
        gradient_x, gradient_y = np.gradient(gray_image)

        # Calculate resultant edge strength of image using magnitude
        gradient_sum = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

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

        # to gray
        gray_state = self.cut_gray(state_image_full)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        print('These are lane boundary 1 points', lane_boundary1_points)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)

            # lane_boundary 1

            # lane_boundary 2

            ################

            for row in range(1, len(maxima)):  # Start from the second row onward

                # Find the maxima in the current row
                current_maxima = maxima[row]

                if len(current_maxima) == 0:
                    continue  # Skip row if no maxima are found

                # Calculate the distance from the current maxima to the last points in each lane boundary
                distance_to_boundary1 = np.abs(current_maxima - lane_boundary1_points[-1, 0])
                distance_to_boundary2 = np.abs(current_maxima - lane_boundary2_points[-1, 0])

                # Find the closest maxima to each lane boundary
                min_idx_boundary1 = np.argmin(distance_to_boundary1)
                min_idx_boundary2 = np.argmin(distance_to_boundary2)

                # Append the closest maxima to the respective lane boundaries
                if distance_to_boundary1[min_idx_boundary1] < 100:  # Threshold to stop if too far
                    lane_boundary1_points = np.vstack([lane_boundary1_points, [current_maxima[min_idx_boundary1], row]])

                if distance_to_boundary2[min_idx_boundary2] < 100:  # Threshold to stop if too far
                    lane_boundary2_points = np.vstack([lane_boundary2_points, [current_maxima[min_idx_boundary2], row]])

                # Remove the assigned maxima to avoid reassigning
                maxima[row] = np.delete(current_maxima, [min_idx_boundary1, min_idx_boundary2])

            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                try:
                    tck1, _ = splprep([lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]],
                                      s=self.spline_smoothness)
                    tck2, _ = splprep([lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]],
                                      s=self.spline_smoothness)
                    lane_boundary1 = tck1
                    lane_boundary2 = tck2
                except Exception as e:
                    print(f"Warning: Spline fitting failed. Using previous splines. Error: {e}")
                    lane_boundary1 = self.lane_boundary1_old
                    lane_boundary2 = self.lane_boundary2_old
                
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        if isinstance(self.lane_boundary1_old, tuple) and isinstance(self.lane_boundary2_old, tuple):
        # evaluate spline for 6 different spline parameters.
            t = np.linspace(0, 1, 6)
            try:
                lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
                lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))

                plt.gcf().clear()
                plt.imshow(state_image_full[::-1])
                plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+320-self.cut_size, linewidth=5, color='orange')
                plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+320-self.cut_size, linewidth=5, color='orange')
                if len(waypoints):
                    plt.scatter(waypoints[0], waypoints[1]+320-self.cut_size, color='white')

                plt.axis('off')
                plt.xlim((-0.5,95.5))
                plt.ylim((-0.5,95.5))
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)
                fig.canvas.flush_events()
            except Exception as e:
                print(f"Error plotting splines: {e}")

        else:
            print("Warning: Invalid spline data, skipping lane plot.")
