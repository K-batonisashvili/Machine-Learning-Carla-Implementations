import deepq
import glob
import os
import sys
import time
import math
import weakref
import time
import random
import numpy as np
from collections import deque
import pygame
import cv2
from graphics import HUD
import action
from reward import generate_route as my_generated_route
from utils import smooth_action

try:
    sys.path.append(glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import logging


class CarlaSimulation:
    def __init__(self):
        self.spectator_width, self.spectator_height = 320, 240
        self.allow_render = True
        self.allow_spectator = True
        self.spectator_camera = None
        self.episode_idx = -2
        self.world = None
        self.reward = None
        self.ego_vehicle = None
        self.ego_cam = None
        self.map = None
        self.cam_bp = None
        self.active_sensors = []
        self.fps = 1
        self.action_space = action.get_action_set()  # NEEDS TO BE CHANGED, I HAVE action.py
        self.observation = None
        # self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(self.spectator_height, self.spectator_width, 3), dtype=np.uint8)
        self.control = carla.VehicleControl()
        self.max_distance = 1000
        self.action_smoothing = .75
        # self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn

    def main(self):

        try:
            client = carla.Client('127.0.0.1', 2000)
            client.set_timeout(10.0)
            self.world = client.get_world()
            self.map = self.world.get_map()

            # --------------
            # Spawn ego vehicle
            # --------------

            ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name', 'ego')
            print('\nEgo role_name is set')
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color', ego_color)
            print('\nEgo color is set')

            spawn_points = self.world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if 0 < number_of_spawn_points:
                random.shuffle(spawn_points)
                ego_transform = spawn_points[0]
                self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
                print('\nEgo is spawned')
            else:
                logging.warning('Could not found any spawn points')

            # --------------
            # PyGame visualization
            # --------------

            if self.allow_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((self.spectator_width, self.spectator_height),
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.spectator_width, self.spectator_height)
                self.hud.set_vehicle(self.ego_vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            # --------------
            # Place spectator on ego spawning
            # --------------

            if self.allow_spectator:
                self.spectator_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
                self.spectator_camera.set_attribute('image_size_x', f'{self.spectator_width}')
                self.spectator_camera.set_attribute('image_size_y', f'{self.spectator_height}')
                self.spectator_camera.set_attribute('fov', '90')
                spectator_location = carla.Location(-5.5, 0, 2.5)
                spectator_transform = carla.Transform(spectator_location)
                self.spectator_sensor = self.world.spawn_actor(self.spectator_camera, spectator_transform, attach_to=self.ego_vehicle)
                self.spectator_sensor.listen(self._set_viewer_image)
                self.active_sensors.append(self.spectator_sensor)
            # --------------
            # Add a RGB camera to ego vehicle.
            # --------------

            self.cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            cam_location = carla.Location(2, 0, 1)
            cam_transform = carla.Transform(cam_location)
            self.cam_bp.set_attribute("image_size_x", str(320))
            self.cam_bp.set_attribute("image_size_y", str(240))
            self.cam_bp.set_attribute("fov", str(90))
            ego_cam = self.world.spawn_actor(self.cam_bp, cam_transform, attach_to=self.ego_vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
            ego_cam.listen(self._set_observation_image)
            self.active_sensors.append(self.ego_cam)

        except RuntimeError as msg:
            pass

        self.reset()

    def vehicle_location(self):
        return self.ego_vehicle.get_location()


    def reset(self):
        print("we are at reset")
        self.cleanup_sensors()
        self.episode_idx += 1
        self.num_routes_completed = -1
        self.current_vehicle_location = self.vehicle_location()

        # Generate a random route
        self.generate_route()


        self.closed = False
        self.terminate = False
        self.success_state = False
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0

        # reset metrics
        self.total_reward = 0.0
        self.previous_location = self.ego_vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.world.tick()

        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)

        return obs

    def render(self):

        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            # "Reward: % 19.2f" % self.last_reward,
            "",
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.allow_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.spectator_camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.height, self.observation.width
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(
            pygame.surfarray.make_surface(self.get_rgb_image(self.observation).swapaxes(0, 1)),
            pos_observation)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list
        # Render to screen
        pygame.display.flip()

    def generate_route(self):
        # Do a soft reset (teleport vehicle)
        self.control.steer = float(0.0)
        self.control.throttle = float(0.0)
        self.ego_vehicle.set_simulate_physics(False)  # Reset the car's physics

        # Generate waypoints along the lap

        spawn_points_list = np.random.choice(self.map.get_spawn_points(), 2, replace=False)
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            print("we are right before rward route")

            self.route_waypoints = my_generated_route(0, 10, map_name='Town01', sampling_resolution=2)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.map.get_spawn_points(), 2, replace=False)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        print("we are at route")
        self.ego_vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.ego_vehicle.set_simulate_physics(True)

    def step(self, action):
        print('we are at step')
        if action is not None:
            # Create new route on route completion
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                self.success_state = True

            throttle, steer = [float(a) for a in action]

            # Perfom action
            self.control.throttle = smooth_action(self.control.throttle, throttle, self.action_smoothing)
            self.control.steer = smooth_action(self.control.steer, steer, self.action_smoothing)

            self.ego_vehicle.apply_control(self.control)

        self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        if self.allow_spectator:
            self.viewer_image = self._get_viewer_image()

        # Get vehicle transform
        transform = self.ego_vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                         self.vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # Did we pass the waypoint?
                waypoint_index += 1  # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),
                                                     self.vector(self.next_waypoint.transform.location),
                                                     self.vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.get_vehicle_lon_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True


        # # Call external reward fn
        # self.last_reward = self.reward_fn(self)
        # self.total_reward += self.last_reward

        self.step_count += 1

        if self.allow_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                pygame.quit()
                if self.world is not None:
                    self.world.destroy()
                self.terminate = True
            self.render()

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }
        print("Rightbefore step ends")
        return self.get_rgb_image(self.observation), self.terminate or self.success_state, info

    def get_vehicle_lon_speed(self):
        carla_velocity_vec3 = self.ego_vehicle.get_velocity()
        vec4 = np.array([carla_velocity_vec3.x,
                         carla_velocity_vec3.y,
                         carla_velocity_vec3.z, 1]).reshape(4, 1)
        carla_trans = np.array(self.ego_vehicle.get_transform().get_matrix())
        carla_trans.reshape(4, 4)
        carla_trans[0:3, 3] = 0.0
        vel_in_vehicle = np.linalg.inv(carla_trans) @ vec4
        return vel_in_vehicle[0]

    def get_rgb_image(self, input):
        # Converting to suitable format for opencv function
        image = np.frombuffer(input.raw_data, dtype=np.uint8)
        image = image.reshape((input.height, input.width, 4))
        image = image[:, :, :3]
        image = image[:, :, ::-1].copy()

        return image

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer
        self.observation_buffer = None
        return obs

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer
        self.viewer_image_buffer = None
        return image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def vector(self, v):
        """ Turn carla Location/Vector3D/Rotation to np.array """
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def distance_to_line(self, A, B, p):
        p[2] = 0
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = self.vector(self.ego_vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(image.transform.get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(image.height)
        image_h = int(image.width)
        fov = float(image.fov)

        image = self.get_rgb_image(image)

        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = self.vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_h, image_w, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image

    def cleanup_sensors(self):
        for sensor in self.active_sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.active_sensors = []


if __name__ == '__main__':

    simulation = CarlaSimulation()
    try:
        simulation.main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_replay.')
