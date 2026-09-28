"""
gae.py - GAE（Generalized Advantage Estimation）
把 buffer 里的 rewards + values 加工成：
  advantages → 训练 Actor（"这个动作比平均好多少"）
  returns    → 训练 Critic（未来累计回报估计）
"""
import numpy as np

def compute_gae(buffer, gamma=0.99, lam=0.95):
    """buffer: 装满的 RolloutBuffer"""
    rollout_size = buffer.size
    advantages = np.zeros(rollout_size, dtype=np.float32)
    returns = np.zeros(rollout_size, dtype=np.float32)

    last_gae = 0.0   # 从尾部往前滚的累计优势

    for t in reversed(range(rollout_size)):
        # 1) 确定"下一步价值"：episode 边界处视为 0（不能跨剧集折现）
        if t == rollout_size - 1 or buffer.dones[t]:
            next_value = 0.0
            last_gae = 0.0          # 跨 episode 重置，别把上一段带进来
        else:
            next_value = buffer.values[t + 1]

        # 2) TD 误差：这一步"现实比预测好/差多少"
        delta = buffer.rewards[t] + gamma * next_value - buffer.values[t]

        # 3) GAE 核心：当前优势 = δ_t + γλ × 后面累计的优势
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

        # 4) returns = advantage + value（Critic 的训练目标）
        returns[t] = advantages[t] + buffer.values[t]

    return advantages, returns