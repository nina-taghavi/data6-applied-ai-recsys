"""Read scalars from TensorBoard event files and make a report-ready figure.

    python export_plot.py --tag eval/mean_reward --runs lr3e-4_seed0 lr3e-4_seed1 lr3e-4_seed2
    python export_plot.py --tag rollout/ep_rew_mean --runs lr3e-4_seed0 lr3e-2_seed0 --csv

Writes <tag>.png (and <tag>.csv with --csv) in the folder figures/.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGDIR = Path("tb_logs")


def load_scalar(run, tag):
    # SB3 adds a suffix _1, _2, ... to tb_log_name; take the first matching folder
    folder = sorted(LOGDIR.glob(f"{run}_*"))[0]
    ea = EventAccumulator(str(folder))
    ea.Reload()
    return pd.DataFrame([(e.step, e.value) for e in ea.Scalars(tag)], columns=["step", "value"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--csv", action="store_true", help="also save the data as CSV")
    a = p.parse_args()

    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    frames = []
    for run in a.runs:
        df = load_scalar(run, a.tag)
        ax.plot(df["step"], df["value"], marker="o", ms=3, label=run)
        frames.append(df.assign(run=run))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(a.tag)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    name = a.tag.replace("/", "_")
    fig.savefig(out / f"{name}.png", dpi=150)
    if a.csv:
        pd.concat(frames).to_csv(out / f"{name}.csv", index=False)
    print("saved", out / f"{name}.png")


if __name__ == "__main__":
    main()
