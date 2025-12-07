from dataclasses import replace
import os
import csv
import numpy as np

from angry_birds_rl.config import EnvConfig, DQNConfig, TrainConfig
from angry_birds_rl.train import train_dqn
from angry_birds_rl.evaluate import evaluate_policy


# 기본 설정
BASE_ENV_CFG = EnvConfig()
BASE_DQN_CFG = DQNConfig()
BASE_TRAIN_CFG = TrainConfig(
    model_dir="checkpoints",
    result_dir="results",
)

def run_single_experiment(
    exp_name: str,
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    train_cfg: TrainConfig,
) -> float:
    """
    하나의 실험 설정으로 학습 + 평가 수행.
    실험별 폴더에 결과 저장.
    """
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


# ============================
# 🔥 MAIN — seed 실험
# ============================
if __name__ == "__main__":
    env_cfg = BASE_ENV_CFG
    base_dqn = BASE_DQN_CFG
    base_train = BASE_TRAIN_CFG

    # =====================================================
    # (3) seed 변경 실험
    # =====================================================
    results = []

    # 실험에 사용할 seed 목록
    seeds = [42, 777, 2024, 999, 1313]

    # hyperparameter 고정값
    eps_decay = 3000           # best epsilon decay
    hidden_dim = 64            # best hidden_dim (앞 실험 기준)

    avg_returns = []

    for s in seeds:
        dqn_cfg = replace(
            base_dqn,
            epsilon_decay_steps=eps_decay,
            hidden_dim=hidden_dim,
        )

        train_cfg = replace(
            base_train,
            seed=s,
        )

        exp_name = f"seed{s}_eps{eps_decay}_h{hidden_dim}"
        avg_ret = run_single_experiment(exp_name, env_cfg, dqn_cfg, train_cfg)

        avg_returns.append(avg_ret)
        results.append([
            exp_name,
            eps_decay,
            hidden_dim,
            s,
            avg_ret,
        ])

    save_results("exp_results_seed.csv", results)

    # 신뢰구간 계산
    mean_ret = float(np.mean(avg_returns))
    std_ret = float(np.std(avg_returns, ddof=1))   # sample std (n-1)
    ci_low = mean_ret - 2 * std_ret
    ci_high = mean_ret + 2 * std_ret

    print("\n==================== Seed 실험 결과 요약 ====================")
    print(f"Seeds          : {seeds}")
    print(f"Avg Return     : {mean_ret:.4f}")
    print(f"Std Dev        : {std_ret:.4f}")
    print(f"95% CI approx  : [{ci_low:.4f}, {ci_high:.4f}]")
    print("============================================================\n")