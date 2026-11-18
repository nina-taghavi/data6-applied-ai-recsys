#!/usr/bin/env python3

import os
import random
import string
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

import deepseek
import gemini

sys.setrecursionlimit(300000)

# ==========================================
# Configuration
# ==========================================
NUM_ITEMS = 50_000
MIN_VAL = 0
MAX_VAL = 1_000_000
MIN_STR_LEN = 5
MAX_STR_LEN = 25
SEED = 42

TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
CHART_FILE = f"comparison_chart_{TIMESTAMP}.png"


def generate_random_numbers(count=NUM_ITEMS, min_v=MIN_VAL, max_v=MAX_VAL, seed=SEED):
    rng = random.Random(seed)
    return [rng.randint(min_v, max_v) for _ in range(count)]


def generate_random_strings(count=NUM_ITEMS, min_l=MIN_STR_LEN, max_l=MAX_STR_LEN, seed=SEED):
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    return [
        "".join(rng.choices(alphabet, k=rng.randint(min_l, max_l)))
        for _ in range(count)
    ]


def get_algorithm_pairs():
    ds_algs = deepseek.get_algorithms()
    gm_algs = gemini.get_algorithms()

    all_names = [
        "3-Way Radix Quicksort",
        "MSD Radix Sort",
        "LSD Radix Sort",
        "Prefix Bucket Sort",
        "Quick Sort (3-Way)",
        "Merge Sort",
        "Heap Sort",
        "Comb Sort",
        "Shell Sort (Knuth)",
        "Cocktail Sort",
    ]

    pairs = []
    for name in all_names:
        ds_func = ds_algs.get(name)
        gm_func = gm_algs.get(name)
        if ds_func and gm_func:
            pairs.append((name, ds_func, gm_func))
    return pairs


def main():
    num_data = generate_random_numbers()
    str_data = generate_random_strings()

    string_only = {
        "3-Way Radix Quicksort",
        "MSD Radix Sort",
        "LSD Radix Sort",
        "Prefix Bucket Sort",
    }

    labels = []
    ds_times = []
    gm_times = []

    for name, ds_func, gm_func in get_algorithm_pairs():
        if name in string_only:
            test_input = str_data
            label = f"{name} [str]"
        elif name == "Cocktail Sort" and len(num_data) > 10_000:
            test_input = num_data[:10_000]
            label = f"{name} [int, 10k]"
        else:
            test_input = num_data
            label = f"{name} [int]"

        expected = sorted(test_input)

        t0 = time.perf_counter()
        ds_out = ds_func(test_input.copy())
        ds_elapsed = time.perf_counter() - t0
        assert ds_out == expected, f"DeepSeek {name} failed!"

        t0 = time.perf_counter()
        gm_out = gm_func(test_input.copy())
        gm_elapsed = time.perf_counter() - t0
        assert gm_out == expected, f"Gemini {name} failed!"

        labels.append(label)
        ds_times.append(ds_elapsed)
        gm_times.append(gm_elapsed)

    # Save chart only
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(base_dir, CHART_FILE)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(labels))
    bar_width = 0.38

    bars1 = ax.bar([i - bar_width / 2 for i in x], ds_times, bar_width, label="DeepSeek", color="#0284c7")
    bars2 = ax.bar([i + bar_width / 2 for i in x], gm_times, bar_width, label="Gemini", color="#10b981")

    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title(f"DeepSeek vs. Gemini — All 10 Algorithms ({NUM_ITEMS:,} Items: Strings & Integers)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}s", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()