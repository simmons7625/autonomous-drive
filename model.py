import torch
import torch.nn as nn
import torch.optim as optim


class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Attentionベースのネットワーク構造
        Args:
            input_dim (int): 入力次元数
            hidden_dim (int): 隠れ層の次元数
            output_dim (int): 出力次元数（行動空間の次元数）
        """
        super(AttentionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        フォワードプロパゲーション
        Args:
            x (torch.Tensor): 入力テンソル（[batch_size, input_dim]）
        Returns:
            torch.Tensor: 行動価値（[batch_size, output_dim]）
        """
        x = torch.relu(self.fc1(x))  # 入力層
        x = x.unsqueeze(1)  # Attentionのために次元を拡張（[batch_size, 1, hidden_dim]）
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        attn_output = attn_output.squeeze(1)  # 元の次元に戻す
        x = self.fc2(attn_output)  # 出力層
        return x


class Attention_based_DQN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        self.policy_network = AttentionNetwork(input_dim, hidden_dim, output_dim)
        self.target_network = AttentionNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # ターゲットネットワークを初期化
        self.update_target_network(1.0)

    def predict(self, state):
        """
        行動価値を予測
        Args:
            state (torch.Tensor): 状態（[batch_size, input_dim]）
        Returns:
            torch.Tensor: 行動価値（[batch_size, output_dim]）
        """
        with torch.no_grad():
            return self.policy_network(state)

    def train(self, states, actions, targets):
        """
        モデルを訓練
        Args:
            states (torch.Tensor): 状態（[batch_size, input_dim]）
            actions (torch.Tensor): 行動（[batch_size]）
            targets (torch.Tensor): ターゲット値（[batch_size]）
        Returns:
            float: 損失値
        """
        self.optimizer.zero_grad()
        q_values = self.policy_network(states)  # 行動価値を予測
        action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # 選択した行動の価値
        loss = self.loss_fn(action_q_values, targets)  # 損失を計算
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self, tau):
        """
        ターゲットネットワークのパラメータを更新
        Args:
            tau (float): ソフト更新の割合（1.0でハード更新）
        """
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        """ネットワークの状態を保存"""
        torch.save(self.policy_network.state_dict(), path)

    def load(self, path):
        """ネットワークの状態を読み込む"""
        self.policy_network.load_state_dict(torch.load(path))
        self.update_target_network(1.0)