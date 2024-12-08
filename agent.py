import carla
import torch
import torch.nn.functional as F
import numpy as np

class Agent:
    def __init__(self, vehicle, model, replay_buffer, batch_size=64, gamma=0.99, tau=0.001):
        """
        Agentクラスの初期化

        Args:
            vehicle (carla.Vehicle): CARLAの車両オブジェクト
            model: Attention-based DQNなどのネットワークモデル
            replay_buffer: 経験を保存するバッファ
            batch_size (int): 学習時のバッチサイズ
            gamma (float): 割引率
            tau (float): ターゲットネットワークのソフト更新パラメータ
        """
        self.vehicle = vehicle
        self.model = model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def apply_control(self, throttle=0.5, steer=0.0):
        """車両の制御を適用"""
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        self.vehicle.apply_control(control)

    def action(self, observation, temperature=1.0):
        # モデルを評価モードに
        self.model.policy_network.eval()
        # 行動価値を計算
        with torch.no_grad():
            action_values = self.model.policy_network(observation)

        # 確率に基づいて行動をサンプリング
        probabilities = F.softmax(action_values / temperature, dim=1).squeeze(0)
        action = torch.multinomial(probabilities, 1).item()
        return action

    def train(self, observations, actions, rewards, next_observations, dones):
        target_q_values = self.model.target_network.predict(next_observations)
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + self.gamma * max_target_q_values * (1 - dones)
        loss = self.model.train(observations, actions, targets)
        return loss

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

