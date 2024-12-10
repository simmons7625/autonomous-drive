import random
import carla
import pygame
import numpy as np

class CarlaEnv:
    def __init__(self, host='127.0.0.1', port=2000, resolution=(1280, 720), num_vehicles=10, num_pedestrians=10):
        pygame.init()
        self.screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("CARLA Environment")

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town02')

        # 設定
        self.resolution = resolution
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians

        self.actor_list = []
        self.rgb_image = None
        self.collision_detected = False
        self.previous_distance = None

        # NPC管理用クラス
        self.npc_manager = NPCManager(self.world)

        # プレイヤー車両
        self.player = None
        self.destination = None

        # 初期化
        self._setup_environment()

    def _setup_environment(self):
        """環境全体の初期化"""
        self._spawn_player()
        self._setup_collision_sensor()
        self._setup_rgb_camera()

        # NPCのスポーン
        self.npc_manager.spawn_vehicles(self.num_vehicles, self.actor_list)
        self.npc_manager.spawn_pedestrians(self.num_pedestrians, self.actor_list)

    def _spawn_player(self):
        """プレイヤー車両をスポーン"""
        # 車両をrandomで選択
        # priusで固定
        vehicle_bp = self.world.get_blueprint_library().find('vehicle.toyota.prius')
        topology = self.world.get_map().get_topology()[0]
        spawn_point = topology[0].transform
        self.destination = topology[1].transform
        self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.player is None:
            raise RuntimeError("Failed to spawn player vehicle.")
        self.actor_list.append(self.player)
        self.previous_distance = self._calculate_distance_to_destination()

    def _setup_collision_sensor(self):
        """衝突センサーを設定"""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.player)
        collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(collision_sensor)

    def _setup_rgb_camera(self):
        """RGBカメラを設定"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.resolution[0]))
        camera_bp.set_attribute('image_size_y', str(self.resolution[1]))
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.player)
        rgb_camera.listen(lambda image: self._process_rgb_image(image))
        self.actor_list.append(rgb_camera)

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
        self.world.tick()
        self.collision_detected = False
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

        self.world.tick()
        obs = self._get_observation()

        current_distance = self._calculate_distance_to_destination()
        distance_delta = self.previous_distance - current_distance
        self.previous_distance = current_distance

        reward, done = self._calculate_reward(distance_delta)

        return obs, reward, done, {}

    def _calculate_distance_to_destination(self):
        """目的地までの距離を計算"""
        player_location = self.player.get_location()
        return player_location.distance(self.destination)

    def _calculate_reward(self, distance_delta):
        """報酬と終了条件を計算"""
        reward = 0
        done = False
        if self.collision_detected:
            reward -= 100
            done = True
        elif self._reached_destination():
            reward += 100
            done = True
        elif distance_delta > 0:
            reward += 1
        return reward, done

    def _reached_destination(self):
        """目的地に到達したか確認"""
        return self._calculate_distance_to_destination() < 2.0

    def _get_observation(self):
        """観測を取得"""
        return {'rgb': self.rgb_image}

    def close(self):
        """環境を終了"""
        pygame.quit()
        self.npc_manager.destroy_all()
        for actor in self.actor_list:
            actor.destroy()
        print("Environment closed.")

class NPCManager:
    def __init__(self, world):
        self.world = world
        self.actors = []

    def spawn_vehicles(self, num_vehicles, actor_list):
        """NPC車両をスポーン"""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        for _ in range(num_vehicles):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                self.actors.append(vehicle)
                actor_list.append(vehicle)

    def spawn_pedestrians(self, num_pedestrians, actor_list):
        """NPC歩行者をスポーン"""
        blueprint_library = self.world.get_blueprint_library()
        walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        for _ in range(num_pedestrians):
            walker_bp = random.choice(walker_bp_list)
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            if spawn_point.location:
                walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                if walker:
                    controller = self.world.try_spawn_actor(
                        walker_controller_bp, carla.Transform(), attach_to=walker)
                    if controller:
                        controller.start()
                        controller.go_to_location(self.world.get_random_location_from_navigation())
                        controller.set_max_speed(1.4)
                        self.actors.extend([walker, controller])
                        actor_list.extend([walker, controller])

    def destroy_all(self):
        """すべてのNPCを削除"""
        for actor in self.actors:
            actor.destroy()

