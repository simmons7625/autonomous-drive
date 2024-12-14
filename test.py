import carla
import random

host='127.0.0.1'
port=2000

client = carla.Client(host, port)
client.set_timeout(10.0)
world = client.load_world('Town02')
topology = world.get_map().get_topology()
random_pair = random.choice(topology)  # ランダムなペアを選択
spawn_point1 = random_pair[0].transform
spawn_points = world.get_map().get_spawn_points()
spawn_point2 = random.choice(spawn_points)

print(spawn_point1, '\n', spawn_point2)
