from dataclasses import replace
import os
import csv

from angry_birds_rl.config import EnvConfig, DQNConfig, TrainConfig
from angry_birds_rl.train import train_dqn
from angry_birds_rl.evaluate import evaluate_policy


# 기본 설정 (공통 베이스)
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
    하나의 설정으로 학습 + 평가를 수행하고 평균 return을 반환.
    exp_name별로 결과 디렉터리를 분리해서 저장한다.
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


if __name__ == "__main__":
    env_cfg = BASE_ENV_CFG
    base_dqn = BASE_DQN_CFG
    base_train = BASE_TRAIN_CFG

    # =====================================================
    # (2) hidden_dim 실험
    # =====================================================
    results = []

    hidden_list = [64, 128, 256]   # 비교할 hidden_dim 후보들
    eps_decay_fixed = 3000         # 위에서 가장 괜찮았던 값 사용 추천

    for h in hidden_list:
        dqn_cfg = replace(
            base_dqn,
            epsilon_decay_steps=eps_decay_fixed,  # 고정
            hidden_dim=h,                         # 변하는 부분
        )
        train_cfg = replace(
            base_train,
            seed=42,                              # 고정
        )

        exp_name = f"hidden{h}_eps{eps_decay_fixed}"
        avg_ret = run_single_experiment(exp_name, env_cfg, dqn_cfg, train_cfg)

        results.append([
            exp_name,
            dqn_cfg.epsilon_decay_steps,
            h,
            train_cfg.seed,
            avg_ret,
        ])

    save_results("exp_results_hidden.csv", results)
    print("\n=== hidden_dim 실험 완료 ===")