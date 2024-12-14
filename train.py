import os
import pandas as pd
from model import SAC
from carla_env import CarlaEnv
from replay_buffer import ReplayBuffer
import torch

# ハイパーパラメータを定義
metrics = {
    'capacity': 10000,          # リプレイバッファの容量
    'batch_size': 1,           # バッチサイズ
    'episodes': 1000,           # エピソード数
    'steps_per_episode': 200,   # 各エピソードの最大ステップ数

    'gamma': 0.99,              # 割引率
    'tau': 0.001,               # ターゲットネットワーク更新の割合
    'lr': 1e-3,                 # 学習率
    'input_dim': 3,             # 状態空間次元数
    'hidden_dim': 8,           # 隠れ層ユニット数
    'output_dim': 2,            # 行動空間次元数
    'alpha': 0.2                # エントロピー温度
}

# 記録用CSVを作成
def open_csv():
    os.makedirs('result', exist_ok=True)  # ディレクトリがなければ作成
    df = pd.DataFrame(columns=['Episode', 'Step', 'Reward', 'Actor Loss', 'Critic Loss'])
    df.to_csv('result/training_metrics.csv', index=False)
    return df

# CSVに記録
def log_metrics(episode, step, reward, actor_loss, critic_loss, df):
    # 新しい行を辞書として作成
    new_row = pd.DataFrame([{
        'Episode': episode,
        'Step': step,
        'Reward': reward,
        'Actor Loss': actor_loss,
        'Critic Loss': critic_loss
    }])

    # DataFrameを結合
    df = pd.concat([df, new_row], ignore_index=True)

    # CSVファイルに保存
    df.to_csv('result/training_metrics.csv', index=False)

    return df


def main(metrics):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 記録用のデータフレーム
    df = open_csv()

    # リプレイバッファと環境、エージェントを初期化
    buffer = ReplayBuffer(metrics['capacity'])
    env = CarlaEnv()

    # SACモデルを初期化
    model = SAC(metrics['input_dim'], metrics['output_dim'], metrics['hidden_dim'], metrics['alpha'], metrics['lr'], device)

    # 学習ループ
    for episode in range(metrics['episodes']):
        obs = env.reset()
        total_reward = 0
        actor_loss, critic_loss = 0, 0

        for step in range(metrics['steps_per_episode']):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            # 行動選択
            throttle, steer, brake = model.sample_action(obs_tensor)

            # 環境の次の状態を取得
            next_obs, reward, done, _ = env.step((throttle, steer, brake))

            # 報酬を蓄積
            total_reward += reward

            # リプレイバッファに保存
            buffer.add(obs, (throttle, steer, brake), reward, next_obs, done)

            # 観測更新
            obs = next_obs

            # 学習
            if len(buffer) >= metrics['capacity']:
                batch = buffer.sample(metrics['batch_size'])
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch

                # バッチをデバイスに移動
                obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(device)
                act_batch = torch.tensor(act_batch, dtype=torch.float32).to(device)
                rew_batch = torch.tensor(rew_batch, dtype=torch.float32).to(device)
                next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32).to(device)

                losses = model.update(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)
                actor_loss, critic_loss = losses['actor_loss'], losses['critic_loss']

            if done:
                break

        # エピソードの結果をログ
        df = log_metrics(episode, step, total_reward, actor_loss, critic_loss, df)
        print(f"Episode {episode + 1}/{metrics['episodes']}, Total Reward: {total_reward}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

    # 環境を終了
    env.close()
    print("Training completed.")

if __name__ == "__main__":
    main(metrics)
