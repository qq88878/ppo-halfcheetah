"""
plot_results.py — 从 training_log.csv 读取训练日志并绘制 reward 曲线。

用法：
    python plot_results.py                 # 默认读 training_log.csv
    python plot_results.py my_log.csv      # 指定日志文件
    python plot_results.py --window 20     # 滑动平均窗口（默认 10）

输出：
    reward_curve.png   — 平均 reward 曲线 + 滑动平均（可直接放进 README）
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口环境也可保存图片
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path):
    """读取 CSV：iteration, avg_reward, num_episodes"""
    iterations, rewards, episodes = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iterations.append(int(row["iteration"]))
            rewards.append(float(row["avg_reward"]))
            episodes.append(int(row["num_episodes"]))
    return np.array(iterations), np.array(rewards), np.array(episodes)


def moving_average(x, window):
    """滑动平均：window 个点取均值，不足 window 时取已有点的均值"""
    if len(x) < window:
        window = len(x)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="绘制 PPO 训练 reward 曲线")
    parser.add_argument("log", nargs="?", default="training_log.csv",
                        help="训练日志 CSV 路径（默认 training_log.csv）")
    parser.add_argument("--window", type=int, default=10,
                        help="滑动平均窗口大小（默认 10）")
    parser.add_argument("--out", default="reward_curve.png",
                        help="输出图片路径（默认 reward_curve.png）")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"找不到日志文件 {log_path}。请先运行 train.py 训练，生成 training_log.csv。")

    iterations, rewards, episodes = load_log(log_path)
    if len(iterations) == 0:
        sys.exit("日志为空，请确认 train.py 已正常记录。")

    avg = moving_average(rewards, args.window)
    avg_x = iterations[: len(avg)]  # 卷积后 x 轴与 y 轴对齐

    # ---- 画图 ----
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    # 上图：reward 曲线
    ax = axes[0]
    ax.plot(iterations, rewards, color="#94a3b8", linewidth=0.8,
            alpha=0.6, label="avg reward (per 10 iters)")
    ax.plot(avg_x, avg, color="#2563eb", linewidth=2.2,
            label=f"moving avg (window={args.window})")
    ax.axhline(0, color="#cbd5e1", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Average Reward")
    ax.set_title("PPO on HalfCheetah-v5 — Training Reward")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 下图：累计完成的 episode 数
    ax = axes[1]
    ax.plot(iterations, episodes, color="#10b981", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Episodes")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已保存曲线图：{args.out}")

    # ---- 打印摘要 ----
    print(f"日志点：{len(iterations)} 个（每 10 iteration 一个）")
    print(f"起始 avg reward：{rewards[0]:.1f}")
    print(f"最终 avg reward：{rewards[-1]:.1f}")
    best_idx = int(np.argmax(rewards))
    print(f"最高 avg reward：{rewards[best_idx]:.1f} @ iteration {iterations[best_idx]}")
    print(f"累计完成 episode：{episodes[-1]}")


if __name__ == "__main__":
    main()
