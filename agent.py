import carla
import torch
import torch.nn.functional as F
import numpy as np

class Agent:
    def __init__(self, vehicle, model, replay_buffer, batch_size=16, gamma=0.99, tau=0.001, device='cuda'):
        """
        Agentクラスの初期化

        Args:
            vehicle (carla.Vehicle): CARLAの車両オブジェクト
            model: Attention-based DQNなどのネットワークモデル
            replay_buffer: 経験を保存するバッファ
            batch_size (int): 学習時のバッチサイズ
            gamma (float): 割引率
            tau (float): ターゲットネットワークのソフト更新パラメータ
            device (str): 使用するデバイス（'cuda' または 'cpu'）
        """
        self.vehicle = vehicle
        self.model = model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

    def apply_control(self, throttle=0.5, steer=0.0, brake=0.0):
        """車両の制御を適用"""
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        self.vehicle.apply_control(control)

    def action(self, observation):
        """観測値から行動を選択"""
        # モデルを評価モードに
        self.model.actor.eval()
        with torch.no_grad():
            # 観測値をGPUに移動
            observation = observation.to(self.device)
            # アクターモデルから行動をサンプリング
            mean, log_std = self.model.actor(observation)
            std = torch.exp(log_std)
            epsilon = torch.randn_like(mean, device=self.device)
            action = torch.tanh(mean + std * epsilon)  # 再パラメータ化トリック

        throttle = action[0].item()
        steer = action[1].item()
        brake = 0.0  # 必要に応じて追加

        # throttle < 0 はbrakeとして処理
        if throttle < 0:
            brake = abs(throttle)
            throttle = 0.0

        return throttle, steer, brake

    def train(self, observations, actions, rewards, next_observations, dones):
        """モデルを訓練"""
        # テンソルに変換してGPUに移動
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # モデルを更新
        actor_loss, critic_loss = self.model.update(observations, actions, rewards, next_observations, dones)

        return {"actor_loss": actor_loss, "critic_loss": critic_loss}

    def soft_update_target_network(self):
        self.model.update_target_network(tau=self.tau)

    def save_network(self, path):
        """ネットワークを保存"""
        self.model.save(path)

    def set_network(self, path):
        """ネットワークを読み込み"""
        self.model.load(path)

    def cleanup(self):
        """車両リソースを解放"""
        self.vehicle.destroy()
