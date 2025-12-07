from angry_birds_rl.config import EnvConfig, DQNConfig, TrainConfig
from angry_birds_rl.train import train_dqn
from angry_birds_rl.plot_results import plot_training_curves


def main():
    env_cfg = EnvConfig()
    dqn_cfg = DQNConfig()
    train_cfg = TrainConfig()

    train_dqn(env_cfg, dqn_cfg, train_cfg)
    plot_training_curves(train_cfg, window=50, show=False)


if __name__ == "__main__":
    main()