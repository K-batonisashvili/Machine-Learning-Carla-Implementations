from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key

# init carla environement

from lane_detection import LaneDetection
import glob
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pygame
import argparse
import random
from PIL import Image
import pyglet
from pyglet import gl
from pyglet.window import key


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
import manual_control
from manual_control import World, KeyboardControl, HUD

# Define variables
steps = 0
LD_module = LaneDetection()
pygame.init()
pygame.font.init()
carla_world = None

fig = plt.figure()
plt.ion()
plt.show()

try:

    # init carla environment
    client = carla.Client('localhost', 2000)
    client.set_timeout(80.0)  # Increase if needed
    world = client.get_world()

    # Initialize display for pygame
    display = pygame.display.set_mode((320, 240), pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0, 0, 0))
    # pygame.display.flip()

    # Initialize HUD and world objects
    args = argparse.Namespace(rolename='hero', filter='vehicle.*', gamma=2.2)
    hud = HUD(320, 240)

    # --------------
    # Spawn ego vehicle
    # --------------

    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name','ego')
    print('\nEgo role_name is set')
    ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
    ego_bp.set_attribute('color',ego_color)
    print('\nEgo color is set')

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if 0 < number_of_spawn_points:
        random.shuffle(spawn_points)
        ego_transform = spawn_points[0]
        ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
        print('\nEgo is spawned')

    ego_vehicle.set_autopilot(True)

    # --------------
    # Add a RGB camera sensor to ego vehicle.
    # --------------

    cam_bp = None
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x",str(320))
    cam_bp.set_attribute("image_size_y",str(240))
    cam_bp.set_attribute("fov",str(90))
    cam_location = carla.Location(1.5,0,2)
    cam_bp.set_attribute("sensor_tick", str(2.0))
    cam_transform = carla.Transform(cam_location)
    ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)

    image_processing = False

    def cam_callback(image):
        global image_processing
        if not image_processing:  # Only save if not currently processing
            image.save_to_disk('front_view_camera.png')
            print('Image was saved')
            image_processing = True  # Set processing flag to True
    ego_cam.listen(lambda image: cam_callback(image))

    spectator = world.get_spectator()
    world_snapshot = world.wait_for_tick()
    spectator.set_transform(ego_vehicle.get_transform())

    clock = pygame.time.Clock()

    while True:
        clock.tick_busy_loop(60)
        world_snapshot = world.wait_for_tick()
        pygame.display.flip()

        if image_processing:
            img = Image.open("front_view_camera.png")  # Load screenshot
            camera_array = np.array(img)  # Convert to numpy array
            # Perform lane detection
            lane1, lane2 = LD_module.lane_detection(camera_array)
            waypoints = waypoint_prediction(lane1, lane2)
            target_speed = target_speed_prediction(waypoints)
            #
            #Plot detected lanes
            LD_module.plot_state_lane(camera_array, steps, fig, waypoints=waypoints, target_speed=target_speed)
            image_processing = False  # Reset processing flag


        # Check for exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

finally:
    # --------------
    # Stop recording and destroy actors
    # --------------
    client.stop_recorder()
    if ego_vehicle is not None:
        if ego_cam is not None:
            ego_cam.stop()
            ego_cam.destroy()
        ego_vehicle.destroy()
    pygame.quit()