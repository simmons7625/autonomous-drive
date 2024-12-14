import torch
import torch.nn as nn
import torch.optim as optim

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1):
        super(AttentionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc2_mean = nn.Linear(hidden_dim, output_dim)  # 平均を出力
        self.fc2_std = nn.Linear(hidden_dim, output_dim)  # 分散を出力

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        batch_size, height, width, channels = x.size()
        x = x.view(batch_size, height*width, -1)
        # Self-attention適用
        attn_output, _ = self.attention(x, x, x)  # [batch_size, num_patches, hidden_dim]
        # print(attn_output.shape) #  (batch_size, 9216, 8)
        
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
            nn.Linear(256, 1)
        )

    def forward(self, state):
        x = torch.cat([state], dim=1)
        return self.net(x)

class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim, alpha, lr=1e-3, device='cuda'):
        self.actor = AttentionNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.target_critic = Critic(state_dim).to(device)

        self.alpha = alpha  # 温度パラメータ（スカラー値のまま）
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
            action = torch.tanh(mean + std * epsilon) .squeeze(0)

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
        actions = torch.tanh(mean + std * epsilon)

        # Q値を取得して損失を計算
        q = self.critic(states)
        actor_loss = (self.alpha - q).mean()  # log_probs を使用しない

        return actor_loss

    def compute_critic_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """
        クリティックモデルの損失を計算
        """
        print(states.shape, actions.shape, rewards.shape, next_states.shape, dones)
        with torch.no_grad():
            next_q = self.target_critic(next_states)
            target_q = rewards + (1 - dones) * gamma * next_q  # log_probs を使用しない
        
        q = self.critic(states)
        critic_loss = self.loss_fn(q, target_q)

        return critic_loss

    def update(self, states, actions, rewards, next_states, dones):
        """
        モデルを更新
        """
        # Criticの更新
        critic_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones)
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

    def save(self, path):
        """
        モデルを保存
        """
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")

    def load(self, path):
        """
        モデルを読み込み
        """
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "_critic.pth"))
        self.update_target_network(1.0)
