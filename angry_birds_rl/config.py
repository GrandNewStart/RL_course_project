from dataclasses import dataclass


@dataclass
class EnvConfig:
    # 타겟(돼지) 위치 범위
    pig_x_min: float = 5.0
    pig_x_max: float = 20.0
    pig_y_min: float = 0.0
    pig_y_max: float = 5.0

    # 물리 파라미터
    gravity: float = 9.81
    hit_radius: float = 1.0

    # 액션 이산화(각도, 파워)
    n_angle_bins: int = 12
    n_power_bins: int = 8


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_update_interval: int = 500
    train_start_size: int = 1_000
    max_episodes: int = 2_000

    # 이 환경은 한 번 쏘면 끝이므로 1로 두되,
    # 일반성을 위해 남겨둠.
    max_steps_per_episode: int = 1

    # epsilon-greedy 탐색
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    hidden_dim: int = 128
    device: str = "cuda"  # 사용 가능한 경우에만 cuda


@dataclass
class TrainConfig:
    seed: int = 42
    log_interval: int = 50
    model_dir: str = "checkpoints"
    model_name: str = "dqn_angry_birds.pt"
    result_dir: str = "results"