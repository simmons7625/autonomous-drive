import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        """
        リプレイバッファの初期化
        Args:
            capacity (int): 保存する経験の最大数
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, observation, action, reward, next_observation, done):
        """経験を追加"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (observation, action, reward, next_observation, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """経験をランダムにサンプリング"""
        batch = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(next_observations),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)