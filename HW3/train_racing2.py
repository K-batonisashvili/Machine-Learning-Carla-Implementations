import time

import carla
import numpy as np
import pygame
import random
import math
from deepq import learn
from reward import generate_route, reward_function



class CarlaEnv:
    """CARLA env initialization."""

    def __init__(self, client, route, display=None, spectator_active=True):
        self.client = client
        self.world = client.get_world()
        self.route = route
        self.display = display if spectator_active else None
        self.spectator_active = spectator_active
        self.vehicle = None
        self.spectator_camera = None
        self.rgb_camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.collision = False
        self.lane_invasion = False
        self.previous_distance = float('inf')
        self.route_index = 0
        self.done = False
        self.reward_counter = 0
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.step_count = 0
        self.movement_threshold = 3
        self.movement_check_steps = 100
        self.previous_positions = []

    def reset(self):
        """Reset the env."""
        # Spawn vehicle at the start of the route
        self.destroy()
        self.collision = False
        self.lane_invasion = False
        self.previous_distance = float('inf')
        self.route_index = 0
        self.done = False
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.reward_counter = 0
        self.step_count = 0
        self.step_count_limit = 0
        self.previous_positions = []
        spectator = self.world.get_spectator()

        # Spawn vehicle at the start of the route
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        for attempt in range(10):  # Retry spawning up to 10 times
            random_spawn_point = random.choice(spawn_points)  # Randomize spawn point
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random_spawn_point)
            if self.vehicle is not None:
                break  # Successful spawn

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts.")
        self.world.wait_for_tick()
        self._setup_spectator_camera_behind_car()
        # Attach sensors
        self._attach_collision_sensor()
        self._attach_lane_invasion_sensor()
        self._setup_rgb_camera()
        if self.spectator_active:
            self._setup_spectator_camera()


        rgb_image = self.get_image(self.rgb_camera)
        rgb_image_normalized = rgb_image.astype(np.float32) / 255.0

        return rgb_image_normalized

    def step(self, action):
        """
        Apply an action and return the next state, reward, and done flag.

        Args:
        - action: The control command to apply (throttle, steer, brake).
        """
        steer, throttle, brake = action
        print("This is the action:", action)
        print("!!!!!!!!!!!!!!!!!!!!!PRINTING COLLISION!!!!!!!!!!!!!!!!!!!", self.collision)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)

        # Update route index if vehicle is close to the next waypoint
        current_location = self.vehicle.get_location()
        next_waypoint = self.route[self.route_index][0].transform.location
        distance_to_waypoint = current_location.distance(next_waypoint)

        if distance_to_waypoint < 2.0:
            self.route_index += 1

        # Check if route is completed
        if self.route_index >= len(self.route):
            self.done = True

        # Get speed of the vehicle
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        # Compute reward using the reward function
        current_position = (current_location.x, current_location.y)
        next_waypoint_position = (next_waypoint.x, next_waypoint.y)
        reward = reward_function(
            collision=self.collision,
            speed=speed,
            lane_invasion=self.lane_invasion,
            current_position=current_position,
            previous_distance=self.previous_distance,
            next_waypoint=next_waypoint_position,
        )
        self.reward_counter += reward
        # Update previous distance
        self.previous_distance = distance_to_waypoint

        self.step_count += 1
        self.previous_positions.append(current_position)

        if self.collision:
            self.collision_count += 1
        if self.lane_invasion:
            self.lane_invasion_count += 1

        if self.collision_count >= 1:
            print("Collided more than this many times:", self.collision_count)
            self.done = True
        if self.lane_invasion_count >= 4:
            print("lane invaded more than this many times:", self.lane_invasion_count)
            self.done = True
        if self.reward_counter < -200:
            print("reward less than 200, its:", self.reward_counter)
            self.done = True

        if self.step_count >= self.movement_check_steps:
            self.step_count_limit += self.step_count
            total_movement = sum(
                np.linalg.norm(np.array(self.previous_positions[i]) - np.array(self.previous_positions[i - 1]))
                for i in range(1, len(self.previous_positions))
            )
            if total_movement < self.movement_threshold:
                self.done = True
            self.step_count = 0
            self.previous_positions = []

        if self.step_count_limit > 250:
            self.done = True

        rgb_image = self.get_image(self.rgb_camera)
        rgb_image_normalized = rgb_image.astype(np.float32) / 255.0

        return rgb_image_normalized, reward, self.done, {}

    def seed(self, seed):
        """
        Set the random seed for the environment to ensure reproducibility.

        Args:
        - seed: int, the seed value to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        # If CARLA has any seed-dependent components, set them here.

    def get_state(self):
        """Get the current state of the environment."""
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        next_waypoint = self.route[self.route_index][0].transform.location
        state = np.array([
            location.x, location.y, location.z,
            velocity.x, velocity.y, velocity.z,
            next_waypoint.x, next_waypoint.y, next_waypoint.z
        ])
        return state

    def get_image(self, camera):
        """Capture an image from the camera sensor."""
        image_buffer = []

        def image_callback(data):
            image_buffer.append(data)

        camera.listen(image_callback)

        # Wait for the image to be populated
        timeout = time.time() + 5.0  # 5 seconds timeout
        while not image_buffer and time.time() < timeout:
            time.sleep(0.1)

        if not image_buffer:
            raise RuntimeError("Failed to retrieve image from sensor within timeout period.")

        camera.stop()

        # Convert the image to a NumPy array
        image = image_buffer[0]
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Exclude alpha channel
        return array

    def destroy(self):
        """Clean up the environment."""
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.spectator_camera is not None:
            self.spectator_camera.stop()
            self.spectator_camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
        if self.rgb_camera is not None:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.stop()
            self.lane_invasion_sensor.destroy()

    def _attach_collision_sensor(self):
        """Attach a collision sensor to the vehicle."""
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.vehicle)

        def on_collision(colli):
            self.collision = True
            print("Collision occurred")

        self.collision_sensor.listen(on_collision)

    def _attach_lane_invasion_sensor(self):
        """Attach a lane invasion sensor to the vehicle."""
        blueprint = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.vehicle)

        def on_lane_invasion(laneinvade):
            self.lane_invasion = True
            print("Lane invasion occurred")

        self.lane_invasion_sensor.listen(on_lane_invasion)

    def _setup_rgb_camera(self):
        """Set up a spectator camera to observe the vehicle."""
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '320')
        blueprint.set_attribute('image_size_y', '240')
        blueprint.set_attribute('fov', '90')
        # blueprint.set_attribute("sensor_tick",str(2.0))
        camera_transform = carla.Transform(carla.Location(x=2, y=0, z=1))
        self.rgb_camera = self.world.spawn_actor(blueprint, camera_transform, attach_to=self.vehicle)

    def _setup_spectator_camera_behind_car(self):
        """Set up the spectator camera to observe the vehicle."""
        spectator = self.world.get_spectator()

        # Get the vehicle's current transform (location and rotation)
        vehicle_transform = self.vehicle.get_transform()

        # Calculate the new camera location
        # Move 2 meters behind the vehicle and 5 meters above its base
        offset_distance = 10.0  # Distance behind the vehicle
        height_offset = 5.0  # Height above the vehicle
        vehicle_location = vehicle_transform.location
        forward_vector = vehicle_transform.get_forward_vector()  # Vehicle's forward direction

        # Calculate the new spectator location
        new_location = vehicle_location - forward_vector * offset_distance  # Move backward
        new_location.z += height_offset  # Adjust height

        # Set the spectator's transform with the same rotation as the vehicle
        new_transform = carla.Transform(location=new_location, rotation=vehicle_transform.rotation)
        spectator.set_transform(new_transform)
    def _setup_spectator_camera(self):
        """Set up a spectator camera to observe the vehicle."""
        if not self.spectator_active:
            return

        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '320')
        blueprint.set_attribute('image_size_y', '240')
        blueprint.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=-5, y=0, z=2.5))
        self.spectator_camera = self.world.spawn_actor(blueprint, camera_transform, attach_to=self.vehicle)

        def process_image(image):
            # Convert image to numpy array and display in pygame
            if self.display is not None:
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Ignore alpha channel
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                self.display.blit(surface, (0, 0))
                pygame.display.flip()

        self.spectator_camera.listen(process_image)




def setup_pygame_window():
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((320, 240), pygame.HWSURFACE | pygame.DOUBLEBUF)

def main():
    spectator_active = False
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Generate a random route
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    start_index = random.randint(0, len(spawn_points) - 1)
    end_index = random.randint(0, len(spawn_points) - 1)
    route = generate_route(start_index, end_index)


    # Initialize pygame
    display = None
    if spectator_active:
        pygame.init()
        pygame.font.init()
        display = pygame.display.set_mode((320, 240), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Agent Viewer")

    # Initialize environment
    env = CarlaEnv(client, route, display, spectator_active)
    time.sleep(0.5)

    try:
        # Train the vehicle using the `learn` function
        learn(env,
              lr=1e-4,
              total_timesteps=10000,
              buffer_size=500000,
              exploration_fraction=0.1,
              exploration_final_eps=0.02,
              train_freq=1,
              action_repeat=3,
              batch_size=32,
              learning_starts=1000,
              gamma=0.70,
              target_network_update_freq=500,
              model_identifier='carla_agent')
    finally:
        env.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')
