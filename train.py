import pandas as pd
from agent import Agent
from carla_env import CarlaEnv
from replay_buffer import ReplayBuffer
from model import Attention_based_DQN  # モデルクラスをインポート
import numpy as np

# ハイパーパラメータを定義
metrics = {
    'capacity': 10000,          # リプレイバッファの容量
    'batch_size': 64,           # バッチサイズ
    'episodes': 1000,           # エピソード数
    'steps_per_episode': 200,   # 各エピソードの最大ステップ数

    'gamma': 0.99,              # 割引率
    'tau': 0.005,               # ターゲットネットワーク更新の割合
    'lr': 1e-3,                 # 学習率
    'input_dim': 10,            # 状態空間次元数（適宜設定）
    'hidden_dim': 128,          # 隠れ層ユニット数
    'output_dim': 2             # 行動空間次元数（例: [throttle, steer_left, steer_right, brake, do_nothing]）
}

# 記録用CSVを作成
def open_csv():
    df = pd.DataFrame(columns=['Episode', 'Step', 'Reward'])
    df.to_csv('training_metrics.csv', index=False)
    return df

# CSVに記録
def log_metrics(episode, step, reward, loss, df):
    df = df.append({'Episode': episode, 'Step': step, 'Reward': reward, 'Loss': loss}, ignore_index=True)
    df.to_csv('training_metrics.csv', index=False)
    return df

def main(metrics):
    # 記録用のデータフレーム
    df = open_csv()

    # リプレイバッファと環境、エージェントを初期化
    buffer = ReplayBuffer(metrics['capacity'])
    env = CarlaEnv()
    
    model = Attention_based_DQN(metrics['input_dim'], metrics['hidden_dim'], metrics['output_dim'], lr=metrics['lr'])
    agent = Agent(env.player, model, buffer, metrics['batch_size'], metrics['gamma'], metrics['tau'])

    # 学習ループ
    for episode in range(metrics['episodes']):
        obs = env.reset()
        total_reward = 0

        for step in range(metrics['steps_per_episode']):
            # ランダムな行動（後でモデルからの行動選択に切り替える
            if step <= metrics['capacity']:
                throttle = np.random.choice(0, 1)
                steer = np.random.choice(-1, 1)
                brake = np.random.choice(0, 1)
            else:
                throttle, steer, brake = agent.action(obs)

            # 行動をエージェントで適用
            # throttle:0 ~ 1, steer = -1.0 ~ 1.0
            agent.apply_control(throttle=throttle, steer=steer, brake=brake)

            # 環境の次の状態を取得
            next_obs, reward, done, _ = env.step((throttle, steer, brake))

            # 報酬を蓄積
            total_reward += reward

            # リプレイバッファに保存
            buffer.add(obs, (throttle, steer, brake), reward, next_obs, done)

            # 観測更新
            obs = next_obs

            # 学習
            if len(buffer) >= metrics['batch_size']:
                batch = buffer.sample(metrics['batch_size'])
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch
                loss = agent.train(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)

            if done:
                break

        # エピソードの結果をログ
        df = log_metrics(episode, step, total_reward, loss, df)
        print(f"Episode {episode + 1}/{metrics['episodes']}, Total Reward: {total_reward}")

    # 環境とエージェントを終了
    env.close()
    agent.cleanup()
    print("Training completed.")

if __name__ == "__main__":
    main(metrics)
