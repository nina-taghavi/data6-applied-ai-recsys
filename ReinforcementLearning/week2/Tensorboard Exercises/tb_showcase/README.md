# TensorBoard showcase (CartPole-v1, PPO)

Six prepared training runs to explore in TensorBoard, used with the exercise *Reading training curves with TensorBoard*.

## Quick start

```bash
pip install stable-baselines3[extra] tensorboard
tensorboard --logdir tb_logs
```

Open http://localhost:6006.

## Runs

All runs: PPO with SB3 defaults, 50,000 environment steps, evaluation every 5,000 steps (10 deterministic episodes on the original reward). Training and evaluation are seeded, but exact curves can still differ between computers and library versions; the exercise asks about patterns, not exact values.

| Run | What is different |
| --- | --- |
| `lr3e-4_seed0/1/2` | Default learning rate 3e-4, three seeds |
| `lr3e-3_seed0` | Learning rate 3e-3 (10x larger) |
| `lr3e-2_seed0` | Learning rate 3e-2 (100x larger) |
| `centered_seed0` | Modified reward 1 - abs(x)/2.4 per step (stay near the centre), as a `gymnasium.RewardWrapper` |

Logged next to the SB3 defaults:

- `eval/mean_reward`, `eval/mean_ep_length`: `EvalCallback` on the original reward
- `custom/original_ep_return`: episode return on the original reward (+1 per step)
- `custom/mean_abs_cart_position`: mean |x| of the cart per episode
- `custom/mean_abs_pole_angle_deg`: mean |pole angle| per episode
- HPARAMS tab: learning rate, gamma, reward type and seed

## Files

| File | Purpose |
| --- | --- |
| `showcase.py` | Trains the runs (`python showcase.py`, about 3 min on 2 CPU cores; `--only <run>` for one run) |
| `export_plot.py` | Reads event files and saves a figure (and CSV) for a report |
| `tb_logs/` | The finished runs |

Tested with Python 3.13, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15, tensorboard 2.21.
