import glob
import os
import sys
import time

try:
    # sys.path.append(glob.glob('A:/Robot Learning/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (  # Kote directory
    sys.path.append(glob.glob('C:/Users/bcerv/OneDrive/Documents/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (  # Rob directory
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
                
import carla

import argparse
import logging
import random

import numpy as np
import pandas as pd

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        world = client.get_world()
        ego_vehicle = None
        ego_cam = None
        ego_col = None
        ego_lane = None
        ego_obs = None
        ego_gnss = None
        ego_imu = None

        # # --------------
        # # Start recording
        # # --------------
        #
        # client.start_recorder('~/tutorial/recorder/recording01.log')
        #
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
        else: 
            logging.warning('Could not found any spawn points')

        # --------------
        # Spectator on ego position
        # --------------
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()
        spectator.set_transform(ego_vehicle.get_transform())

        # --------------
        # Spawn attached RGB camera
        # --------------
        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(320))
        cam_bp.set_attribute("image_size_y",str(240))
        cam_bp.set_attribute("fov",str(105))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        # Initialize dataframe
        global drive_dataframe 
        drive_dataframe = pd.DataFrame(columns=['Timestamp','Throttle','Brake','Steering','Direction','Obstacle detected',
                                                'Lane invasion detected','Collision detected','GNSS measure','IMU measure'])

        # Log file for frame, timestamp, and control inputs
        # log_file_path = 'A:/Robot Learning/HW1/Training_Data/frame_input_log.txt' # Kote directory
        log_file_path = 'C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Training_Data' # Rob directory

        def camera_callback(image):

            global drive_dataframe

            # Capture the frame timestamp
            frame_timestamp = image.timestamp
            
            # Get the current vehicle control inputs
            control = ego_vehicle.get_control()
            throttle = control.throttle
            brake = control.brake
            steering_raw = control.steer

            if steering_raw < -0.01:
                print('Steering less than -0.005, you are going left')
                direction = 'left'
            elif steering_raw > 0.01:
                print("Steering greater than 0.005, you are going right")
                direction = 'right'
            else:
                steering_raw = 0
                print('Steering is in between, going straight')
                direction = 'straight'

            # Save Data To Array
            drive_dataframe = drive_dataframe.append({
                "Timestamp": frame_timestamp,
                "Throttle": throttle,
                "Brake": brake,
                "Steering": steering_raw,
                "Direction": direction
            }, ignore_index = True)

            # image.save_to_disk('A:/Robot Learning/HW1/Training_Data/%.6d.jpg' % image.frame) # Kote directory
            image.save_to_disk('C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Training_Data/%.6d.jpg' % image.frame) # Rob directory

        # Set the camera to listen for frames
        ego_cam.listen(camera_callback)

        # --------------
        # Add Lane invasion sensor to ego vehicle.
        # --------------
        lane_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_location = carla.Location(0,0,0)
        lane_rotation = carla.Rotation(0,0,0)
        lane_transform = carla.Transform(lane_location,lane_rotation)
        ego_lane = world.spawn_actor(lane_bp,lane_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def lane_callback(lane):
            global drive_dataframe
            drive_dataframe = drive_dataframe.append({"Lane invasion detected": lane}, ignore_index = True)
            print("Lane invasion detected:\n"+str(lane)+'\n')
        ego_lane.listen(lambda lane: lane_callback(lane))

        # --------------
        # Add Obstacle sensor to ego vehicle.
        # --------------
        obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')
        obs_bp.set_attribute("only_dynamics",str(True))
        obs_location = carla.Location(0,0,0)
        obs_rotation = carla.Rotation(0,0,0)
        obs_transform = carla.Transform(obs_location,obs_rotation)
        ego_obs = world.spawn_actor(obs_bp,obs_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def obs_callback(obs):
            global drive_dataframe
            drive_dataframe = drive_dataframe.append({"Obstacle detected": obs}, ignore_index=True)
            print("Obstacle detected:\n"+str(obs)+'\n')
        ego_obs.listen(lambda obs: obs_callback(obs))

        # --------------
        # Add collision sensor to ego vehicle.
        # --------------
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        col_location = carla.Location(0,0,0)
        col_rotation = carla.Rotation(0,0,0)
        col_transform = carla.Transform(col_location,col_rotation)
        ego_col = world.spawn_actor(col_bp,col_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def col_callback(colli):
            global drive_dataframe
            drive_dataframe = drive_dataframe.append({"Collision detected": colli}, ignore_index=True)
            print("Collision detected:\n"+str(colli)+'\n')
        ego_col.listen(lambda colli: col_callback(colli))

        # # --------------
        # # Add GNSS sensor to ego vehicle.
        # # --------------
        # gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        # gnss_location = carla.Location(0,0,0)
        # gnss_rotation = carla.Rotation(0,0,0)
        # gnss_transform = carla.Transform(gnss_location,gnss_rotation)
        # gnss_bp.set_attribute("sensor_tick",str(3.0))
        # ego_gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # def gnss_callback(gnss):
        #     global drive_dataframe
        #     drive_dataframe = drive_dataframe.append({
        #         "GNSS measure": gnss}, ignore_index=True)
        #     print("GNSS measure:\n"+str(gnss)+'\n')
        # ego_gnss.listen(lambda gnss: gnss_callback(gnss))

        # --------------
        # Add IMU sensor to ego vehicle.
        # --------------
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_location = carla.Location(0,0,0)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location,imu_rotation)
        imu_bp.set_attribute("sensor_tick",str(3.0))
        ego_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def imu_callback(imu):
            global drive_dataframe
            drive_dataframe = drive_dataframe.append({"IMU measure": imu}, ignore_index=True)
            print("IMU measure:\n"+str(imu)+'\n')
        ego_imu.listen(lambda imu: imu_callback(imu))

        # --------------
        # Enable autopilot for ego vehicle
        # --------------
        
        ego_vehicle.set_autopilot(True)

        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()
            
    finally:
        # --------------
        # Stop recording and destroy actors
        # --------------
        # drive_dataframe.to_csv(r'A:/Robot Learning/HW1/Training_Data/drive_data.csv',index=False) # Kote directory
        drive_dataframe.to_csv(r'C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Training_Data/drive_data.csv',index=False) # Rob directory
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()
            ego_vehicle.destroy()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')