"""TensorBoard showcase: generate a set of PPO runs on CartPole-v1 to explore in TensorBoard.

    python showcase.py            # all runs (about 3 min on 2 CPU cores)
    python showcase.py --only lr_high_seed0
    tensorboard --logdir tb_logs  # then open http://localhost:6006

Runs (all PPO, 50,000 steps, SB3 defaults unless stated):
    lr3e-4_seed0/1/2        default learning rate, three seeds   -> seed variability
    lr3e-3_seed0            10x larger learning rate             -> faster but less stable?
    lr3e-2_seed0            100x larger learning rate            -> what does instability look like?
    centered_seed0          modified reward (stay near the centre) -> training reward != task reward

Every run logs, besides the SB3 defaults:
    eval/mean_reward, eval/mean_ep_length    deterministic evaluation on the ORIGINAL reward (EvalCallback,
                                             10 episodes every 5,000 steps, seeded evaluation env)
    custom/original_ep_return                episode return on the original reward (+1 per step)
    custom/mean_abs_cart_position            mean |x| of the cart during an episode (task-specific metric)
    custom/mean_abs_pole_angle_deg           mean |angle| of the pole during an episode
    HPARAMS tab: learning rate, gamma, reward type, seed vs final metrics
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "CartPole-v1"
TOTAL_TIMESTEPS = 50_000
LOGDIR = "tb_logs"
EVAL_SEED = 1000  # same evaluation episodes for every run

RUNS = {
    "lr3e-4_seed0": dict(learning_rate=3e-4, seed=0),
    "lr3e-4_seed1": dict(learning_rate=3e-4, seed=1),
    "lr3e-4_seed2": dict(learning_rate=3e-4, seed=2),
    "lr3e-3_seed0": dict(learning_rate=3e-3, seed=0),
    "lr3e-2_seed0": dict(learning_rate=3e-2, seed=0),
    "centered_seed0": dict(learning_rate=3e-4, seed=0, reward="centered"),
}


class CenteredRewardWrapper(gym.RewardWrapper):
    """Reward = 1 - |x| / 2.4 instead of 1: still positive for surviving, less when far from the centre.
    A RewardWrapper only changes the reward the agent sees; observations, termination and info stay the same."""

    def reward(self, reward):
        x = np.float32(self.env.unwrapped.state[0])  # cart position after the step (same value as obs[0])
        return reward * (1 - abs(x) / 2.4)


class CustomMetricsCallback(BaseCallback):
    """Logs episode-level custom metrics to TensorBoard with self.logger.record_mean.
    Values are averaged over the episodes that finish between two log dumps."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.ret, self.x, self.angle, self.steps = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        # HPARAMS tab: which settings belong to this run, and which metrics to compare them on
        self.logger.record(
            "hparams",
            HParam(self.hparams, {"eval/mean_reward": 0.0, "rollout/ep_rew_mean": 0.0}),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self):
        obs = self.locals["new_obs"]
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            o = info["terminal_observation"] if done else obs[i]
            self.ret[i] += 1  # original CartPole reward: +1 per step (the wrapper only changes what PPO sees)
            self.x[i] += abs(o[0])
            self.angle[i] += abs(np.degrees(o[2]))
            self.steps[i] += 1
            if done:
                self.logger.record_mean("custom/original_ep_return", self.ret[i])
                self.logger.record_mean("custom/mean_abs_cart_position", self.x[i] / self.steps[i])
                self.logger.record_mean("custom/mean_abs_pole_angle_deg", self.angle[i] / self.steps[i])
                self.logger.record_mean( "custom/reached_500", float(info.get("TimeLimit.truncated", False )))
                self.ret[i] = self.x[i] = self.angle[i] = self.steps[i] = 0
        return True


def train(name):
    torch.set_num_threads(1)
    cfg = RUNS[name]
    env = gym.make(ENV_ID)
    if cfg.get("reward") == "centered":
        env = CenteredRewardWrapper(env)

    model = PPO("MlpPolicy", env, learning_rate=cfg["learning_rate"], seed=cfg["seed"],
                tensorboard_log=LOGDIR, verbose=0, device="cpu")

    # Evaluation env: original reward, Monitor for correct episode stats, seeded so every
    # regeneration of the logs gives the same eval curves
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    eval_env.seed(EVAL_SEED)
    eval_cb = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=5_000,
                           deterministic=True, verbose=0)
    hparams = {"learning_rate": cfg["learning_rate"], "gamma": model.gamma,
               "reward": cfg.get("reward", "original"), "seed": cfg["seed"]}
    model.learn(TOTAL_TIMESTEPS, callback=[CustomMetricsCallback(hparams), eval_cb], tb_log_name=name)
    return name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None, help="names of runs to train")
    p.add_argument("--jobs", type=int, default=2)
    names = p.parse_args().only or list(RUNS)
    with ProcessPoolExecutor(max_workers=p.parse_args().jobs) as ex:
        for n in ex.map(train, names):
            print("done:", n, flush=True)
