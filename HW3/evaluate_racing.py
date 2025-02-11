import random

import carla
from deepq import evaluate
from reward import generate_route
from train_racing2 import CarlaEnv


def main():
    """ 
    Evaluate a trained Deep Q-Learning agent.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    start_index = random.randint(0, len(spawn_points) - 1)
    end_index = random.randint(0, len(spawn_points) - 1)
    route = generate_route(start_index, end_index)

    env = CarlaEnv(client, route, display=None, spectator_active=False)

    try:
        evaluate(env, load_path='carla_agent.pt')
    finally:
        env.destroy()
        print("Evaluation completed.")

if __name__ == '__main__':
    main()