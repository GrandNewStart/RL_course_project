from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DQNConfig
from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """
    간단한 MLP 기반 Q(s, a) 근사기.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class DQNAgent:
    config: DQNConfig
    state_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        device_str = (
            self.config.device
            if torch.cuda.is_available()
            else "cpu"
        )
        self.device = torch.device(device_str)

        self.online_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
        )

        self.replay_buffer = ReplayBuffer(self.config.replay_capacity)
        self._step_count: int = 0

    # epsilon 스케줄
    def _compute_epsilon(self) -> float:
        if self._step_count >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end

        start = self.config.epsilon_start
        end = self.config.epsilon_end
        ratio = self._step_count / float(self.config.epsilon_decay_steps)
        return start + (end - start) * ratio

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """epsilon-greedy 정책."""
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def update(self) -> Tuple[float, float]:
        """
        DQN 업데이트 1스텝 수행.
        return: (loss, epsilon)
        """
        if len(self.replay_buffer) < self.config.train_start_size:
            epsilon = self._compute_epsilon()
            return 0.0, epsilon

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.config.batch_size)

        self._step_count += 1
        epsilon = self._compute_epsilon()

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        # Q(s, a)
        q_values = self.online_net(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # target = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            max_next_q, _ = torch.max(next_q, dim=1)
            targets = rewards_t + self.config.gamma * max_next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # target network 동기화
        if self._step_count % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item()), float(epsilon)

    def save(self, path: str) -> None:
        payload = {
            "model_state": self.online_net.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.online_net.load_state_dict(ckpt["model_state"])
        self.target_net.load_state_dict(self.online_net.state_dict())