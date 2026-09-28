import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_size=2048, obs_dim=17, act_dim=6):
        # 一次性开好 6 个大数组
        self.obs       = np.zeros((rollout_size, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((rollout_size, act_dim), dtype=np.float32)
        self.rewards   = np.zeros(rollout_size, dtype=np.float32)
        self.dones     = np.zeros(rollout_size, dtype=np.float32)   # 终止标记
        self.log_probs = np.zeros(rollout_size, dtype=np.float32)   # 旧策略 log_prob
        self.values    = np.zeros(rollout_size, dtype=np.float32)   # Critic 预测 V(s)
        self.ptr = 0    # 写指针：下一个存哪
        self.size = 0   # 已存条数

    def store(self, obs, action, reward, done, log_prob, value):
        """收集循环每走一步调用一次"""
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1
        self.size += 1

    def is_full(self):
        """收集满 2048 步了吗"""
        return self.size >= len(self.obs)

    def clear(self):
        """PPO 更新完必须清空（on-policy：旧数据一次性用完就扔）"""
        self.ptr = 0
        self.size = 0