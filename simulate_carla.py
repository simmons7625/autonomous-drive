import random
import carla
import numpy as np
import pygame

class CarlaEnv:
    def __init__(self, host='127.0.0.1', port=2000, resolution=(1280, 720)):
        pygame.init()
        self.screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("CARLA Environment")

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # カメラの解像度を設定
        self.resolution = resolution

        # 環境設定
        self.actor_list = []
        self.vehicle_npcs = []
        self.pedestrian_npcs = []
        self.player = None
        self.rgb_image = None
        self.depth_image = None
        self.collision_detected = False
        self.destination = None
        self.previous_distance = None

        # 初期化
        self._setup_environment()

    def _setup_environment(self):
        """環境の初期化（プレイヤー車両、センサー、目的地、NPCの設定）"""
        blueprint_library = self.world.get_blueprint_library()

        # プレイヤー車両をスポーン
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.player is None:
            raise RuntimeError("Failed to spawn player vehicle.")
        self.actor_list.append(self.player)

        # 目的地を設定
        self.destination = random.choice(spawn_points).location
        self.previous_distance = self._calculate_distance_to_destination()

        # 衝突センサーを設定
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.player)
        collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(collision_sensor)

        # RGBカメラを設定
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.resolution[0]))
        camera_bp.set_attribute('image_size_y', str(self.resolution[1]))
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.rgb_camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.player)
        self.rgb_camera.listen(lambda image: self._process_rgb_image(image))
        self.actor_list.append(self.rgb_camera)

    def _on_collision(self, event):
        """衝突イベントの処理"""
        self.collision_detected = True

    def _process_rgb_image(self, image):
        """RGB画像を処理して保存"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        self.rgb_image = array

    def render(self):
        """pygameで画像を描画"""
        if self.rgb_image is not None:
            surface = pygame.surfarray.make_surface(self.rgb_image.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def reset(self):
        """環境をリセット"""
        self.world.tick()  # ワールドを進める
        self.collision_detected = False  # 衝突フラグをリセット
        self.previous_distance = self._calculate_distance_to_destination()
        return self._get_observation()

    def step(self, action):
        """行動を適用して次の状態と報酬を取得"""
        throttle, steer, brake = action
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        self.player.apply_control(control)

        # 次の状態を取得
        self.world.tick()
        obs = self._get_observation()

        # 距離の変化を計算
        current_distance = self._calculate_distance_to_destination()
        distance_delta = self.previous_distance - current_distance
        self.previous_distance = current_distance

        # 報酬を計算
        reward = 0
        # 終了条件
        done = False
        if self.collision_detected:
            reward -= 100  # 衝突ペナルティ
            done = True
        elif self._reached_destination():
            reward += 100  # 目的地到達ボーナス
            done = True
        elif distance_delta > 0:  # 目的地に近づいていれば報酬を与える
            reward += 1

        return obs, reward, done, {}

    def _calculate_distance_to_destination(self):
        """目的地までの距離を計算"""
        player_location = self.player.get_location()
        distance = player_location.distance(self.destination)
        return distance

    def _reached_destination(self):
        """プレイヤーが目的地に到達したかを確認"""
        distance = self._calculate_distance_to_destination()
        return distance < 2.0  # 距離が2m未満なら到達とみなす

    def _get_observation(self):
        """現在の観測を取得"""
        return {
            'rgb': self.rgb_image
        }

    def close(self):
        """環境を終了してリソースを解放"""
        pygame.quit()
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        print("Environment closed.")

# テスト用のエージェント
if __name__ == "__main__":
    env = CarlaEnv()
    obs = env.reset()
    done = False
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = [0.5, 0.0, 0.0]  # throttle=0.5, steer=0.0, brake=0.0
        obs, reward, done, _ = env.step(action)
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        clock.tick(30)  # 30FPSで更新
    env.close()
