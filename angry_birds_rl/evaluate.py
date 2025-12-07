from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from .config import EnvConfig, DQNConfig, TrainConfig
from .environment import AngryBirdsEnv
from .dqn_agent import DQNAgent


def evaluate_policy(
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    train_cfg: TrainConfig,
    n_episodes: int = 50,
) -> Tuple[float, List[float]]:
    env = AngryBirdsEnv(
        pig_x_range=(env_cfg.pig_x_min, env_cfg.pig_x_max),
        pig_y_range=(env_cfg.pig_y_min, env_cfg.pig_y_max),
        gravity=env_cfg.gravity,
        hit_radius=env_cfg.hit_radius,
        n_angle_bins=env_cfg.n_angle_bins,
        n_power_bins=env_cfg.n_power_bins,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        config=dqn_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    model_path = f"{train_cfg.model_dir}/{train_cfg.model_name}"
    ckpt = torch.load(model_path, map_location="cpu")
    agent.online_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(agent.online_net.state_dict())
    agent.online_net.eval()

    returns: List[float] = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            state = next_state
            done = terminated or truncated

        returns.append(ep_ret)

    env.close()
    mean_ret = float(np.mean(returns))
    print(f"Evaluation over {n_episodes} episodes: avg return = {mean_ret:.3f}")
    return mean_ret, returns