from angry_birds_rl.config import EnvConfig, DQNConfig, TrainConfig
from angry_birds_rl.evaluate import evaluate_policy


def main():
    env_cfg = EnvConfig()
    dqn_cfg = DQNConfig()
    train_cfg = TrainConfig()

    evaluate_policy(env_cfg, dqn_cfg, train_cfg, n_episodes=100)


if __name__ == "__main__":
    main()