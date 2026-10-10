"""拆分三次训练日志并绘制 旧版 vs tanh修复版 对比图"""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

src = Path("training_log.csv")
text = src.read_text(encoding="utf-8")

# 按表头行切分（追加模式下每段训练都有一个表头）
parts = re.split(r"iteration,avg_reward,num_episodes\n", text)
parts = [p for p in parts if p.strip()]
assert len(parts) == 3, f"期望 3 段，实际 {len(parts)} 段"

def parse(part):
    rows = [l.split(",") for l in part.strip().splitlines() if l.strip()]
    it = np.array([int(r[0]) for r in rows])
    rw = np.array([float(r[1]) for r in rows])
    return it, rw

it1, rw1 = parse(parts[0])  # 旧版（无 tanh）
it2, rw2 = parse(parts[1])  # tanh 首跑（630 崩）
it3, rw3 = parse(parts[2])  # tanh 修复版

# 保存三段独立 CSV
for name, it, rw in [("log_old_no_tanh.csv", it1, rw1),
                     ("log_tanh_crashed.csv", it2, rw2),
                     ("log_tanh_fixed.csv", it3, rw3)]:
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "avg_reward"])
        for a, b in zip(it, rw):
            w.writerow([a, b])

def movavg(x, w=20):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.plot(it1, rw1, color="#94a3b8", lw=0.8, alpha=0.5, label="旧版（无 tanh，raw）")
ax.plot(it3, rw3, color="#c7d2fe", lw=0.8, alpha=0.5, label="tanh 修复版（raw）")
ax.plot(it1[:len(movavg(rw1))], movavg(rw1), color="#64748b", lw=2.0, label="旧版（滑均 20）")
ax.plot(it3[:len(movavg(rw3))], movavg(rw3), color="#4f46e5", lw=2.6, label="tanh 修复版（滑均 20）")
ax.axvline(630, color="#f43f5e", ls="--", lw=1.2, alpha=0.8)
ax.text(640, 500, "tanh 首跑在此崩溃(NaN)\n→ 修复 std下限+clamp+梯度裁剪", fontsize=10, color="#f43f5e")
ax.axhline(0, color="#cbd5e1", lw=0.8, ls="--")
ax.set_xlabel("Iteration")
ax.set_ylabel("Average Reward")
ax.set_title("HalfCheetah-v5 · PPO 训练对比：旧版 vs tanh 修复版")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("reward_curve_compare.png", dpi=150)

best1, best3 = rw1.max(), rw3.max()
print(f"旧版：start {rw1[0]:.1f} → end {rw1[-1]:.1f}，峰值 {best1:.1f}@{it1[rw1.argmax()]}")
print(f"tanh修复版：start {rw3[0]:.1f} → end {rw3[-1]:.1f}，峰值 {best3:.1f}@{it3[rw3.argmax()]}")
print(f"峰值提升：{best3 - best1:+.1f}（{(best3/best1 - 1)*100:+.1f}%）")
print("已保存 reward_curve_compare.png + 三段独立 CSV")
