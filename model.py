import torch
import torch.nn as nn
import torch.optim as optim

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1, pool_size=(2, 2)):
        super(AttentionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)  # プーリング処理
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc2_mean = nn.Linear(hidden_dim, output_dim)  # 平均を出力
        self.fc2_std = nn.Linear(hidden_dim, output_dim)  # 分散を出力

    def forward(self, x):
        # 入力の次元: [batch_size, height, width, channels]
        batch_size, height, width, channels = x.size()

        # プーリング処理で空間次元を縮小 (height, width -> pool_size)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W) に変換
        x = self.pool(x)  # Adaptive Average Pooling 適用
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C) に戻す

        # 次元縮小後の情報取得
        _, pooled_height, pooled_width, _ = x.size()

        # 線形変換と活性化関数
        x = torch.relu(self.fc1(x))

        # Self-attention 用に [batch_size, num_patches, hidden_dim] へ変換
        x = x.view(batch_size, pooled_height * pooled_width, -1)
        attn_output, _ = self.attention(x, x, x)  # [batch_size, num_patches, hidden_dim]

        # 平均と分散を計算
        mean = self.fc2_mean(attn_output).mean(dim=1)
        std = self.fc2_std(attn_output).mean(dim=1)

        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 状態価値を出力
        )

    def forward(self, state):
        return self.net(state)  # スカラー値（V(s)）を出力


class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim, alpha, lr=1e-3, tau=0.005, device='cuda'):
        self.actor = AttentionNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)  # Criticは状態価値を出力
        self.target_critic = Critic(state_dim).to(device)

        self.alpha = alpha  # 温度パラメータ
        self.tau = tau
        self.device = device

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # ターゲットネットワークの初期化
        self.update_target_network(1.0)
    
    def sample_action(self, observation):
        """観測値から行動を選択"""
        # モデルを評価モードに
        self.actor.eval()
        with torch.no_grad():
            # 観測値をGPUに移動
            observation = observation.to(self.device)
            # アクターモデルから行動をサンプリング
            mean, std = self.actor(observation)
            epsilon = torch.randn_like(mean, device=self.device)
            action = torch.tanh(mean + std * epsilon).squeeze(0)

        throttle = action[0].item()
        steer = action[1].item()
        brake = 0.0  # 必要に応じて追加

        # throttle < 0 はbrakeとして処理
        if throttle < 0:
            brake = abs(throttle)
            throttle = 0.0

        return throttle, steer, brake
    
    def compute_actor_loss(self, states):
        """
        アクターモデルの損失を計算
        """
        mean, std = self.actor(states)
        epsilon = torch.randn_like(mean).to(self.device)
        actions = torch.tanh(mean + std * epsilon)  # 再パラメータ化トリック

        # log_prob を計算
        log_probs = -0.5 * (
            ((actions - mean) / (std + 1e-6)) ** 2
            + 2 * torch.log(std + 1e-6)
            + torch.log(torch.tensor(2 * torch.pi).to(self.device))
        )
        log_probs = log_probs.sum(dim=-1)  # アクション次元で合計
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)  # tanh の補正

        # 状態価値 V(s) を利用して損失を計算
        v_values = self.critic(states)
        actor_loss = (self.alpha * log_probs - v_values).mean()

        return actor_loss


    def compute_critic_loss(self, states, rewards, next_states, dones, gamma=0.99):
        """
        状態価値関数の損失を計算
        """
        with torch.no_grad():
            # 次状態の状態価値 V(s') を計算
            next_v_values = self.target_critic(next_states)
            target_v = rewards + (1 - dones) * gamma * next_v_values

        # 現在の状態価値 V(s) を取得
        v_values = self.critic(states)
        critic_loss = self.loss_fn(v_values, target_v)

        return critic_loss

    def update(self, states, actions, rewards, next_states, dones):
        """
        モデルを更新
        """
        # Criticの更新
        critic_loss = self.compute_critic_loss(states, rewards, next_states, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actorの更新
        actor_loss = self.compute_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ターゲットネットワークのソフト更新
        self.update_target_network(self.tau)

        return actor_loss.item(), critic_loss.item()

    def update_target_network(self, tau):
        """
        ターゲットネットワークのパラメータをソフト更新
        """
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)