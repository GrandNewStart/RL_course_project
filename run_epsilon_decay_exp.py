from dataclasses import replace
import os
import csv

from angry_birds_rl.config import EnvConfig, DQNConfig, TrainConfig
from angry_birds_rl.train import train_dqn
from angry_birds_rl.evaluate import evaluate_policy


BASE_ENV_CFG = EnvConfig()
BASE_DQN_CFG = DQNConfig()
BASE_TRAIN_CFG = TrainConfig(model_dir="checkpoints", result_dir="results")


def run_single_experiment(
    exp_name: str,
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    train_cfg: TrainConfig,
) -> float:
    """
    하나의 설정으로 학습 + 평가를 수행하고 avg return을 반환.
    결과는 exp_name별로 폴더/모델 이름을 분리해서 저장.
    """
    # exp별로 폴더/파일 이름 분리
    model_dir = os.path.join(train_cfg.model_dir, exp_name)
    result_dir = os.path.join(train_cfg.result_dir, exp_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    train_cfg_local = replace(
        train_cfg,
        model_dir=model_dir,
        result_dir=result_dir,
        model_name="dqn.pt",
    )

    print(f"\n=== [START] {exp_name} ===")
    train_dqn(env_cfg, dqn_cfg, train_cfg_local)
    avg_ret, _ = evaluate_policy(env_cfg, dqn_cfg, train_cfg_local, n_episodes=100)
    print(f"=== [END] {exp_name} | avg_return={avg_ret:.3f} ===\n")

    return avg_ret


def save_results(csv_path: str, rows):
    header = ["exp_name", "epsilon_decay_steps", "hidden_dim", "seed", "avg_return"]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    env_cfg = BASE_ENV_CFG
    base_dqn = BASE_DQN_CFG
    base_train = BASE_TRAIN_CFG

    results = []

    # ==============================================================
    # (1) epsilon decay 하이퍼파라미터 비교 실험
    # ==============================================================

    eps_list = [1000, 3000, 10000, 50000]  # 비교하고 싶은 epsilon decay 후보들

    for eps_decay in eps_list:
        # DQN 설정 변경
        dqn_cfg = replace(
            base_dqn,
            epsilon_decay_steps=eps_decay,  # 변경되는 부분
            hidden_dim=128                  # 고정
        )

        # seed는 고정
        train_cfg = replace(
            base_train,
            seed=42
        )

        exp_name = f"eps{eps_decay}"
        avg_ret = run_single_experiment(exp_name, env_cfg, dqn_cfg, train_cfg)

        # CSV용 기록
        results.append([
            exp_name,
            eps_decay,
            dqn_cfg.hidden_dim,
            train_cfg.seed,
            avg_ret
        ])

    save_results("exp_results_epsilon.csv", results)

    print("\n=== 모든 epsilon decay 실험 종료 ===")