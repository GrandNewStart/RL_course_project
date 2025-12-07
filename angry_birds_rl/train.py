from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch

from .config import EnvConfig, DQNConfig, TrainConfig
from .environment import AngryBirdsEnv
from .dqn_agent import DQNAgent


def make_env(cfg: EnvConfig) -> AngryBirdsEnv:
    env = AngryBirdsEnv(
        pig_x_range=(cfg.pig_x_min, cfg.pig_x_max),
        pig_y_range=(cfg.pig_y_min, cfg.pig_y_max),
        gravity=cfg.gravity,
        hit_radius=cfg.hit_radius,
        n_angle_bins=cfg.n_angle_bins,
        n_power_bins=cfg.n_power_bins,
    )
    return env


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    train_cfg: TrainConfig,
) -> Tuple[List[float], List[float]]:
    set_seed(train_cfg.seed)

    env = make_env(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        config=dqn_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    all_returns: List[float] = []
    all_losses: List[float] = []

    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.result_dir, exist_ok=True)
    model_path = os.path.join(train_cfg.model_dir, train_cfg.model_name)

    global_step = 0

    for episode in range(dqn_cfg.max_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0.0

        # 실제로는 한 스텝이지만 while 구조로 유지
        while not done:
            epsilon = agent._compute_epsilon()
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push_transition(state, action, reward, next_state, done)
            loss, epsilon = agent.update()

            episode_return += reward
            state = next_state
            global_step += 1

            if loss > 0:
                all_losses.append(loss)

            if done:
                break

        all_returns.append(episode_return)

        if (episode + 1) % train_cfg.log_interval == 0:
            avg_ret = float(np.mean(all_returns[-train_cfg.log_interval:]))
            avg_loss = float(np.mean(all_losses[-train_cfg.log_interval:])) if all_losses else 0.0
            print(
                f"[Episode {episode + 1:4d}] "
                f"avg_return={avg_ret:.3f}, avg_loss={avg_loss:.4f}, "
                f"buffer_size={len(agent.replay_buffer)}, epsilon={epsilon:.3f}"
            )

    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # 학습 곡선 저장
    returns_path = os.path.join(train_cfg.result_dir, "returns.npy")
    losses_path = os.path.join(train_cfg.result_dir, "losses.npy")
    np.save(returns_path, np.array(all_returns, dtype=np.float32))
    np.save(losses_path, np.array(all_losses, dtype=np.float32))
    print(f"Saved returns to {returns_path}")
    print(f"Saved losses to {losses_path}")

    env.close()
    return all_returns, all_losses