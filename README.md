# Angry Birds RL (DQN)

간단한 Angry Birds 스타일 2D 투사체 발사 환경에서 DQN 에이전트를 학습시키는 프로젝트입니다.

## 환경 개요

- 새(bird)는 (0, 0) 에서 발사됩니다.
- 타겟(pig)은 `[pig_x_min, pig_x_max] x [pig_y_min, pig_y_max]` 범위에서 랜덤으로 생성됩니다.
- 에이전트는 **발사 각도(angle)** 와 **발사 힘(power)** 조합을 이산 action 으로 선택합니다.
- 한 번 발사하면 에피소드가 종료되며, 탄착점과 타겟 사이의 거리에 따라 보상을 받습니다.
  - 반경 `hit_radius` 내부에 떨어지면 `+1.0`
  - 그 외에는 거리 비례 페널티 (최대 `-1.0`)

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
// Epsilon Decay 실험
python run_epsilon_decay_exp.py

// Hidden Dimension 실험
python run_hidden_dim_exp.py

// Seed 실험
python run_seed_exp.py
```

## 모델 경로
- [./checkpoints/{실험 이름}/dqn.pt](./checkpoints/)

- 실험 결과에 따른 최적의 모델: [checkpoints/seed42_eps3000_h64/dqn.pt](./checkpoints/seed42_eps3000_h64/dqn.pt)