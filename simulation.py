import carla
from environment import Agent, Hazard
import random
import time

class Simulation:
    def __init__(self):
        # CARLAサーバーへの接続
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.actors = []
    
    def setup_environment(self):
        # 環境リセット
        self.world.reset_settings()
        
        # 天候設定
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)
    
    def spawn_agent(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # 車両の生成
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actors.append(vehicle)
        return Agent(vehicle)
    
    def spawn_hazard(self):
        blueprint_library = self.world.get_blueprint_library()
        pedestrian_bp = blueprint_library.filter('walker.pedestrian.*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # 歩行者の生成
        pedestrian = self.world.spawn_actor(pedestrian_bp, spawn_point)
        self.actors.append(pedestrian)
        return Hazard(pedestrian)
    
    def cleanup(self):
        for actor in self.actors:
            actor.destroy()

# メイン実行
if __name__ == "__main__":
    simulation = Simulation()
    try:
        simulation.setup_environment()
        
        # 車両エージェントと歩行者の生成
        agent = simulation.spawn_agent()
        hazard = simulation.spawn_hazard()
        
        # シミュレーションループ
        for _ in range(100):
            agent.apply_control(throttle=0.5, steer=random.uniform(-0.1, 0.1))
            time.sleep(0.1)  # 更新間隔
    finally:
        simulation.cleanup()
