import time
import numpy as np
from carla_env import CarlaEnv

def test_carla_env():
    # 環境を初期化
    env = CarlaEnv(num_vehicles=5, num_pedestrians=5)

    try:
        print("Environment initialized successfully.")

        # 環境のリセット
        obs = env.reset()
        print("Environment reset. Initial observation received.")

        # 初期状態の可視化
        env.render()

        # テストアクションの定義
        actions = [
            (0.5, 0.0, 0.0),  # 前進
            (0.5, 0.1, 0.0),  # 前進 + 右折
            (0.5, -0.1, 0.0), # 前進 + 左折
            (0.0, 0.0, 1.0)   # ブレーキ
        ]

        # 環境ステップのテスト
        for i, action in enumerate(actions):
            print(f"Step {i + 1}: Applying action {action}")
            obs, reward, done, info = env.step(action)

            print(f"Observation shape: {obs['rgb'].shape if obs['rgb'] is not None else 'None'}")
            print(f"Reward: {reward}, Done: {done}")

            env.render()

            if done:
                print("Episode finished!")
                break

            # 簡単なウェイト（シミュレーション進行の可視化）
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 環境を閉じる
        env.close()
        print("Environment closed.")

# テストスクリプトを実行
if __name__ == "__main__":
    test_carla_env()